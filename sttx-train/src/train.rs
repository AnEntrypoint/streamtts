use std::path::PathBuf;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{AdamW, Optimizer, ParamsAdamW, VarBuilder, VarMap};
use candle_transformers::models::rwkv_v7::Config;
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde_json::json;
use sttx_core::obs;
use sttx_core::replay::{ReplayBuffer, Record};
use sttx_core::tokenizer::{DynamicTokenizer, Hypernetwork};
use sttx_core::trace::Trace;

use crate::checkpoint::CheckpointMeta;
use crate::model::{fresh_state, LoadedModel};
use crate::tokens;

pub struct TrainConfig {
    pub steps: u64,
    pub max_tokens_per_step: usize,
    pub replay_capacity: usize,
    pub replay_per_step: usize,
    pub replay_weight: f32,
    pub lr: f64,
    pub checkpoint_dir: PathBuf,
    pub checkpoint_every: u64,
    pub seed: u64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            steps: 1000,
            max_tokens_per_step: 256,
            replay_capacity: 1000,
            replay_per_step: 3,
            replay_weight: 0.2,
            lr: 1e-4,
            checkpoint_dir: PathBuf::from("ckpt"),
            checkpoint_every: 100,
            seed: 42,
        }
    }
}

pub struct Trainer {
    pub model: LoadedModel,
    pub trainable: VarMap,
    pub hyper: Hypernetwork,
    pub state_prefix: Vec<Tensor>,
    pub dyn_tk: DynamicTokenizer,
    pub optimizer: AdamW,
    pub replay: ReplayBuffer,
    pub rng: StdRng,
    pub recent_loss: Vec<f32>,
    pub steps: u64,
    pub cfg: TrainConfig,
}

impl Trainer {
    pub fn new(model: LoadedModel, cfg: TrainConfig) -> Result<Self> {
        let trainable = VarMap::new();
        let vb = VarBuilder::from_varmap(&trainable, model.dtype, &model.device);
        let hyper = Hypernetwork::new(model.config.hidden_size, vb.pp("hypernet"))?;

        let state_prefix = build_state_prefix(&model.config, &trainable, &model.device, model.dtype)?;

        let optimizer = AdamW::new(
            trainable.all_vars(),
            ParamsAdamW {
                lr: cfg.lr,
                ..Default::default()
            },
        )?;

        let dyn_tk = DynamicTokenizer::new(512, 256, 3);
        let replay = ReplayBuffer::new(cfg.replay_capacity);
        let rng = StdRng::seed_from_u64(cfg.seed);

        Ok(Self {
            model,
            trainable,
            hyper,
            state_prefix,
            dyn_tk,
            optimizer,
            replay,
            rng,
            recent_loss: Vec::with_capacity(64),
            steps: 0,
            cfg,
        })
    }

    pub fn step_on_trace(&mut self, trace: &Trace) -> Result<f32> {
        let text = trace.corpus_text();
        if text.is_empty() {
            return Ok(0.0);
        }
        let base_ids = tokens::encode_text(&self.model.tokenizer, &text)?;
        if base_ids.len() < 2 {
            return Ok(0.0);
        }
        let merged = tokens::observe_and_merge(&mut self.dyn_tk, &base_ids);
        let model_ids = tokens::flatten_for_model(&merged, &self.dyn_tk);
        let truncated: Vec<u32> = model_ids
            .into_iter()
            .take(self.cfg.max_tokens_per_step + 1)
            .collect();
        if truncated.len() < 2 {
            return Ok(0.0);
        }

        let loss = self.forward_loss(&truncated)?;
        let surprise = loss.to_dtype(DType::F32)?.to_vec0::<f32>()?;
        self.optimizer.backward_step(&loss)?;

        self.replay.add(Record {
            input_ids: truncated[..truncated.len() - 1].to_vec(),
            target_ids: truncated[1..].to_vec(),
            surprise,
        });

        for _ in 0..self.cfg.replay_per_step {
            if self.replay.is_empty() {
                break;
            }
            let sample = {
                let s = self.replay.sample(1, &mut self.rng);
                if s.is_empty() {
                    break;
                }
                let r = s[0];
                let mut combined = r.input_ids.clone();
                combined.push(r.target_ids[r.target_ids.len() - 1]);
                combined
            };
            if sample.len() < 2 {
                continue;
            }
            let r_loss = self.forward_loss(&sample)?;
            let scaled = r_loss.affine(self.cfg.replay_weight as f64, 0.0)?;
            self.optimizer.backward_step(&scaled)?;
        }

        self.steps += 1;
        self.recent_loss.push(surprise);
        if self.recent_loss.len() > 64 {
            self.recent_loss.remove(0);
        }
        obs::info(
            "train",
            json!({
                "event":"step",
                "step": self.steps,
                "loss": surprise,
                "tokens": truncated.len(),
                "merges_total": self.dyn_tk.merges_total,
                "promotions_total": self.dyn_tk.promotions_total
            }),
        );
        Ok(surprise)
    }

    pub fn step_on_ids(&mut self, token_ids: &[u32]) -> Result<f32> {
        if token_ids.len() < 2 {
            return Ok(0.0);
        }
        let truncated: Vec<u32> = token_ids
            .iter()
            .copied()
            .take(self.cfg.max_tokens_per_step + 1)
            .collect();
        let loss = self.forward_loss(&truncated)?;
        let surprise = loss.to_dtype(DType::F32)?.to_vec0::<f32>()?;
        self.optimizer.backward_step(&loss)?;
        self.replay.add(Record {
            input_ids: truncated[..truncated.len() - 1].to_vec(),
            target_ids: truncated[1..].to_vec(),
            surprise,
        });
        self.steps += 1;
        self.recent_loss.push(surprise);
        if self.recent_loss.len() > 64 {
            self.recent_loss.remove(0);
        }
        obs::info(
            "train",
            json!({"event":"step_ids","step": self.steps,"loss": surprise,"tokens": truncated.len()}),
        );
        Ok(surprise)
    }

    pub fn forward_loss(&self, token_ids: &[u32]) -> Result<Tensor> {
        let input = &token_ids[..token_ids.len() - 1];
        let targets = &token_ids[1..];
        let mut state = fresh_state(&self.model.config, &self.model.device, self.model.dtype)?;
        for (layer_idx, prefix) in self.state_prefix.iter().enumerate() {
            state.per_layer[layer_idx].att_kv = prefix.clone();
        }
        let logits_raw = self.model.model.forward_seq(input, &mut state)?;
        let (logits, target_t) = match logits_raw.rank() {
            1 => (
                logits_raw.unsqueeze(0)?,
                Tensor::new(&[*targets.last().unwrap()], &self.model.device)?,
            ),
            2 => (logits_raw, Tensor::new(targets, &self.model.device)?),
            3 => {
                let (_, s, v) = logits_raw.dims3()?;
                (logits_raw.reshape((s, v))?, Tensor::new(targets, &self.model.device)?)
            }
            _ => (logits_raw, Tensor::new(targets, &self.model.device)?),
        };
        let loss = candle_nn::loss::cross_entropy(&logits, &target_t)?;
        Ok(loss)
    }

    pub fn meta(&self, repo_id: String) -> CheckpointMeta {
        CheckpointMeta {
            steps: self.steps,
            recent_loss: self.recent_loss.clone(),
            merges_total: self.dyn_tk.merges_total,
            promotions_total: self.dyn_tk.promotions_total,
            repo_id,
        }
    }
}

pub fn build_state_prefix_pub(cfg: &Config, vm: &VarMap, dev: &Device, dtype: DType) -> Result<Vec<Tensor>> {
    build_state_prefix(cfg, vm, dev, dtype)
}

fn build_state_prefix(cfg: &Config, vm: &VarMap, dev: &Device, dtype: DType) -> Result<Vec<Tensor>> {
    let n_heads = cfg.hidden_size / cfg.head_size;
    let shape = (n_heads, cfg.head_size, cfg.head_size);
    let mut out = Vec::with_capacity(cfg.num_hidden_layers);
    for layer in 0..cfg.num_hidden_layers {
        let t = vm.get(
            shape,
            &format!("state_prefix.{layer}"),
            candle_nn::Init::Const(0.0),
            dtype,
            dev,
        )?;
        out.push(t);
    }
    Ok(out)
}
