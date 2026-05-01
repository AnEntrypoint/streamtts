use anyhow::Result;
use sttx_core::tokenizer::{DynamicTokenizer, DYNAMIC_VOCAB_OFFSET};
use tokenizers::Tokenizer;

pub fn encode_text(tk: &Tokenizer, text: &str) -> Result<Vec<u32>> {
    let enc = tk
        .encode(text, false)
        .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
    Ok(enc.get_ids().to_vec())
}

pub fn observe_and_merge(dyn_tk: &mut DynamicTokenizer, base_ids: &[u32]) -> Vec<u32> {
    dyn_tk.observe(base_ids);
    dyn_tk.merge(base_ids)
}

pub fn split_dynamic(id: u32) -> Option<u32> {
    if id >= DYNAMIC_VOCAB_OFFSET {
        Some(id - DYNAMIC_VOCAB_OFFSET)
    } else {
        None
    }
}

pub fn flatten_for_model(merged: &[u32], dyn_tk: &DynamicTokenizer) -> Vec<u32> {
    let mut out = Vec::with_capacity(merged.len() * 2);
    for &id in merged {
        if id >= DYNAMIC_VOCAB_OFFSET {
            if let Some(pair) = dyn_tk.dynamic_pair(id) {
                out.push(pair.0);
                out.push(pair.1);
                continue;
            }
        }
        out.push(id);
    }
    out
}
