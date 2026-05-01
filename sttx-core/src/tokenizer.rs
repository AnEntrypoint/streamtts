use std::collections::HashMap;
use std::num::NonZeroUsize;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{linear, Linear, Module, VarBuilder, VarMap};
use lru::LruCache;

use crate::obs;
use serde_json::json;

pub const DYNAMIC_VOCAB_OFFSET: u32 = 1 << 30;

pub struct Hypernetwork {
    fc1: Linear,
    fc2: Linear,
    hidden_size: usize,
}

impl Hypernetwork {
    pub fn new(hidden_size: usize, vb: VarBuilder) -> Result<Self> {
        let fc1 = linear(2 * hidden_size, hidden_size, vb.pp("fc1"))?;
        let fc2 = linear(hidden_size, hidden_size, vb.pp("fc2"))?;
        Ok(Self { fc1, fc2, hidden_size })
    }

    pub fn forward(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let cat = Tensor::cat(&[a, b], a.rank() - 1)?;
        let h = self.fc1.forward(&cat)?.gelu()?;
        let out = self.fc2.forward(&h)?;
        Ok(out)
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}

pub struct DynamicTokenizer {
    bigram_counts: HashMap<(u32, u32), u32>,
    window: std::collections::VecDeque<(u32, u32)>,
    window_capacity: usize,
    promote_threshold: u32,
    next_dynamic_id: u32,
    pair_to_id: HashMap<(u32, u32), u32>,
    embed_cache: LruCache<u32, Tensor>,
    pub merges_total: u64,
    pub promotions_total: u64,
}

impl DynamicTokenizer {
    pub fn new(window_capacity: usize, lru_capacity: usize, promote_threshold: u32) -> Self {
        Self {
            bigram_counts: HashMap::new(),
            window: std::collections::VecDeque::with_capacity(window_capacity),
            window_capacity,
            promote_threshold,
            next_dynamic_id: DYNAMIC_VOCAB_OFFSET,
            pair_to_id: HashMap::new(),
            embed_cache: LruCache::new(NonZeroUsize::new(lru_capacity).unwrap()),
            merges_total: 0,
            promotions_total: 0,
        }
    }

    pub fn observe(&mut self, ids: &[u32]) {
        for w in ids.windows(2) {
            let pair = (w[0], w[1]);
            *self.bigram_counts.entry(pair).or_insert(0) += 1;
            if self.window.len() == self.window_capacity {
                let old = self.window.pop_front().unwrap();
                if let Some(c) = self.bigram_counts.get_mut(&old) {
                    *c = c.saturating_sub(1);
                    if *c == 0 {
                        self.bigram_counts.remove(&old);
                    }
                }
            }
            self.window.push_back(pair);
        }
    }

    pub fn merge(&mut self, ids: &[u32]) -> Vec<u32> {
        if ids.len() < 2 {
            return ids.to_vec();
        }
        let mut out = Vec::with_capacity(ids.len());
        let mut i = 0;
        while i < ids.len() {
            if i + 1 < ids.len() {
                let pair = (ids[i], ids[i + 1]);
                if let Some(&dyn_id) = self.pair_to_id.get(&pair) {
                    out.push(dyn_id);
                    self.merges_total += 1;
                    i += 2;
                    continue;
                }
                if self.bigram_counts.get(&pair).copied().unwrap_or(0) >= self.promote_threshold {
                    let dyn_id = self.next_dynamic_id;
                    self.next_dynamic_id += 1;
                    self.pair_to_id.insert(pair, dyn_id);
                    self.promotions_total += 1;
                    obs::info(
                        "tokenizer",
                        json!({
                            "event": "promote",
                            "pair": [pair.0, pair.1],
                            "dyn_id": dyn_id,
                            "promotions_total": self.promotions_total
                        }),
                    );
                    out.push(dyn_id);
                    self.merges_total += 1;
                    i += 2;
                    continue;
                }
            }
            out.push(ids[i]);
            i += 1;
        }
        out
    }

    pub fn dynamic_pair(&self, dyn_id: u32) -> Option<(u32, u32)> {
        self.pair_to_id.iter().find_map(|(p, &id)| if id == dyn_id { Some(*p) } else { None })
    }

    pub fn embed_dynamic(
        &mut self,
        dyn_id: u32,
        base_embed: &dyn Fn(u32) -> Result<Tensor>,
        hyper: &Hypernetwork,
    ) -> Result<Tensor> {
        if let Some(t) = self.embed_cache.get(&dyn_id) {
            return Ok(t.clone());
        }
        let pair = self
            .dynamic_pair(dyn_id)
            .ok_or_else(|| anyhow::anyhow!("unknown dynamic id {dyn_id}"))?;
        let a = base_embed(pair.0)?;
        let b = base_embed(pair.1)?;
        let e = hyper.forward(&a, &b)?;
        self.embed_cache.put(dyn_id, e.clone());
        Ok(e)
    }

    pub fn top_merges(&self, n: usize) -> Vec<((u32, u32), u32, u32)> {
        let mut v: Vec<_> = self
            .pair_to_id
            .iter()
            .map(|(&pair, &id)| {
                let count = self.bigram_counts.get(&pair).copied().unwrap_or(0);
                (pair, id, count)
            })
            .collect();
        v.sort_by(|a, b| b.2.cmp(&a.2));
        v.truncate(n);
        v
    }
}

pub fn make_hypernet(hidden_size: usize, dtype: DType, device: &Device) -> Result<(Hypernetwork, VarMap)> {
    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, dtype, device);
    let h = Hypernetwork::new(hidden_size, vb)?;
    Ok((h, vm))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn promotion_and_merge() {
        let mut t = DynamicTokenizer::new(64, 16, 3);
        let ids = vec![1u32, 2, 1, 2, 1, 2, 5, 6, 7];
        t.observe(&ids);
        let merged = t.merge(&ids);
        assert!(t.promotions_total >= 1, "expected promotions, got {}", t.promotions_total);
        assert!(t.merges_total >= 1);
        assert!(merged.iter().any(|&x| x >= DYNAMIC_VOCAB_OFFSET));
    }

    #[test]
    fn hypernetwork_shape() -> Result<()> {
        let dev = Device::Cpu;
        let (h, _vm) = make_hypernet(32, DType::F32, &dev)?;
        let a = Tensor::zeros(&[1, 32], DType::F32, &dev)?;
        let b = Tensor::zeros(&[1, 32], DType::F32, &dev)?;
        let out = h.forward(&a, &b)?;
        assert_eq!(out.dims(), &[1, 32]);
        Ok(())
    }
}
