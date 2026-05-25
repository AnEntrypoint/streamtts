//! DeepConf-style adaptive latent looping with a Kalman exit gate.
//!
//! Background. The Ouro "Looped Language Models" work scales *compute per token*
//! (a third axis beyond model size and data) by re-feeding a latent vector
//! through the same weights N times, with a learned exit gate. DeepConf (Meta,
//! 2025) replaces "learn when to stop" with "measure the model's own confidence
//! and stop when it's high enough" — no trained gate, so no reward-hacking /
//! loop-collapse to debug. We combine the two: Ouro's loop-during-training
//! structure, DeepConf's confidence signal, and a 2-state Kalman filter
//! ([`sttx_core::kalman`]) to denoise the per-loop confidence and detect the
//! diminishing-returns plateau.
//!
//! RWKV constraint. `rwkv_v7::Model` exposes only `forward_seq`/`forward`, both
//! of which consume token IDs and mutate the recurrent state (advancing
//! `state.pos`); the block stack and head are private, so a true Ouro latent
//! re-feed is not reachable without patching candle-transformers. The reachable
//! RNN analogue is **token feedback on a cloned state**: snapshot the state,
//! then let the model re-read its own predicted token for up to `max_loops`,
//! reading logits each pass. Looping adds compute, not memory — the RWKV state
//! is constant-size — so this fits the 4 GB VRAM budget.

use anyhow::Result;
use candle_core::{DType, Tensor};
use candle_transformers::models::rwkv_v7::{Config, State};
use serde::{Deserialize, Serialize};

use crate::model::fresh_state;

/// Tunable knobs for adaptive looping. Serialized as `loop-config.json` by the
/// `calibrate-loops` subcommand and loaded by the trainer and server. The
/// `Default` impl holds conservative hand values so a missing file is non-fatal.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoopConfig {
    /// Hard cap on loops per token/chunk. `1` reproduces single-pass behavior.
    pub max_loops: usize,
    /// Exit once filtered confidence reaches this (lower group-confidence =
    /// more confident, so we exit when filtered value <= threshold).
    pub conf_threshold: f32,
    /// Exit once |velocity| drops below this (confidence plateaued).
    pub velocity_eps: f32,
    /// Kalman process-noise scale.
    pub kalman_q: f32,
    /// Kalman measurement-noise variance.
    pub kalman_r: f32,
    /// Kalman initial covariance.
    pub kalman_init_var: f32,
    /// top-k for the group-confidence metric.
    pub top_k: usize,
}

impl Default for LoopConfig {
    fn default() -> Self {
        Self {
            max_loops: 1,
            conf_threshold: 0.5,
            velocity_eps: 0.01,
            kalman_q: 0.01,
            kalman_r: 0.1,
            kalman_init_var: 1.0,
            top_k: 5,
        }
    }
}

impl LoopConfig {
    pub fn save(&self, path: &std::path::Path) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Load from disk; returns `Ok(None)` if the file is absent (caller falls
    /// back to `Default`).
    pub fn load(path: &std::path::Path) -> Result<Option<Self>> {
        if !path.exists() {
            return Ok(None);
        }
        let s = std::fs::read_to_string(path)?;
        Ok(Some(serde_json::from_str(&s)?))
    }
}

/// Cross-entropy of a single `[vocab]` logits vector against one target id.
/// Convenience wrapper so callers outside sttx-train (e.g. the CLI's
/// calibrate-loops) need not depend on candle_nn directly.
pub fn cross_entropy_one(logits: &Tensor, target: u32) -> Result<f32> {
    let dev = logits.device();
    let target_t = Tensor::new(&[target], dev)?;
    let ce = candle_nn::loss::cross_entropy(&logits.unsqueeze(0)?, &target_t)?;
    Ok(ce.to_dtype(DType::F32)?.to_vec0::<f32>()?)
}

/// DeepConf group confidence for a single last-token logits vector.
///
/// Returns the negative mean log-probability of the `top_k` most-likely tokens.
/// A peaked (confident) distribution puts large log-probs on a few tokens, so
/// the negated mean is **small**; a flat (uncertain) distribution yields a
/// **large** value. Thus lower = more confident, matching `conf_threshold`'s
/// "<=" exit test.
pub fn group_confidence(logits: &Tensor, top_k: usize) -> Result<f32> {
    // Flatten to a 1-D [vocab] vector regardless of incoming rank.
    let flat = logits.flatten_all()?.to_dtype(DType::F32)?;
    let v: Vec<f32> = flat.to_vec1()?;
    if v.is_empty() {
        return Ok(0.0);
    }
    // log_softmax = x - logsumexp(x), computed in f32 with a max-shift for
    // numerical stability.
    let max = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum_exp: f32 = v.iter().map(|&x| (x - max).exp()).sum();
    let log_z = max + sum_exp.ln();
    let mut log_probs: Vec<f32> = v.iter().map(|&x| x - log_z).collect();
    // Top-k largest log-probs (descending).
    log_probs.sort_unstable_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let k = top_k.clamp(1, log_probs.len());

    // DeepConf confidence = negative log-prob over the top-k, but weighted by the
    // (renormalized) probability mass of those top-k tokens. A peaked
    // distribution concentrates mass on the top token, so the weighted mean is
    // dominated by its near-zero negative-log-prob -> SMALL (confident). A flat
    // distribution spreads weight, pulling in the larger negative-log-probs ->
    // LARGE (uncertain). A plain unweighted top-k mean is wrong: on a one-hot
    // logits it averages in the deeply-negative ranks 2..k and reports the
    // peaked case as *less* confident (the bug the unit test caught).
    let top_logp = &log_probs[..k];
    let weights: Vec<f32> = top_logp.iter().map(|&lp| lp.exp()).collect();
    let wsum: f32 = weights.iter().sum();
    let conf = if wsum > f32::EPSILON {
        top_logp
            .iter()
            .zip(weights.iter())
            .map(|(&lp, &w)| -lp * (w / wsum))
            .sum::<f32>()
    } else {
        -top_logp[0]
    };
    Ok(conf)
}

/// Snapshot of the recurrent state's numeric values, for restore after a loop
/// pass that we don't want to commit. `State` is not `Clone`, but its public
/// `Tensor` fields are (cheap Arc-shared clones).
pub struct StateSnapshot {
    per_layer: Vec<(Tensor, Tensor, Tensor)>,
    pos: usize,
}

impl StateSnapshot {
    pub fn capture(state: &State) -> Self {
        let per_layer = state
            .per_layer
            .iter()
            .map(|l| (l.att_x_prev.clone(), l.att_kv.clone(), l.ffn_x_prev.clone()))
            .collect();
        Self { per_layer, pos: state.pos }
    }

    pub fn restore(&self, state: &mut State) {
        for (l, (axp, kv, fxp)) in state.per_layer.iter_mut().zip(self.per_layer.iter()) {
            l.att_x_prev = axp.clone();
            l.att_kv = kv.clone();
            l.ffn_x_prev = fxp.clone();
        }
        state.pos = self.pos;
    }
}

/// Normalize a slice of (already non-negative) weights to sum to 1. Falls back
/// to uniform weights if the total is ~0.
pub fn normalize_weights(raw: &[f32]) -> Vec<f32> {
    let total: f32 = raw.iter().sum();
    if total <= f32::EPSILON {
        let u = 1.0 / raw.len().max(1) as f32;
        return vec![u; raw.len()];
    }
    raw.iter().map(|&w| w / total).collect()
}

/// Build a fresh state seeded with the trainer's state-prefix tensors. Shared
/// helper so train and serve construct the loop entry-state identically.
pub fn seeded_state(
    config: &Config,
    device: &candle_core::Device,
    dtype: DType,
    state_prefix: &[Tensor],
) -> Result<State> {
    let mut state = fresh_state(config, device, dtype)?;
    for (i, prefix) in state_prefix.iter().enumerate() {
        state.per_layer[i].att_kv = prefix.clone();
    }
    Ok(state)
}

use candle_transformers::models::rwkv_v7::Model;
use sttx_core::kalman::KalmanFilter;

/// Outcome of one adaptive looping pass over a single token position.
pub struct LoopOutcome {
    /// Logits from each realized loop (`len == realized_loops`).
    pub per_loop_logits: Vec<Tensor>,
    /// Per-loop weights from filtered confidence, normalized to sum 1.
    pub weights: Vec<f32>,
    /// How many loops actually ran before the exit gate fired.
    pub realized_loops: usize,
}

/// Run the adaptive latent-refinement loop for a single token whose input id is
/// `token_id`, mutating `state` exactly once (loop 1 commits; refinement loops
/// restore — see `Model::forward_looped`). The DeepConf group-confidence of each
/// loop is fed through a 2-state Kalman filter; we stop early once the filtered
/// confidence is good enough (value `<= conf_threshold`) **and** the velocity has
/// plateaued (`|v| <= velocity_eps`), or at `max_loops`.
///
/// With `max_loops == 1` this is one `forward_looped` pass returning a single
/// logits tensor with weight 1 — identical to the pre-looping `forward` path.
pub fn run_token_loops(
    model: &Model,
    token_id: u32,
    state: &mut State,
    device: &candle_core::Device,
    cfg: &LoopConfig,
) -> Result<LoopOutcome> {
    let max_loops = cfg.max_loops.max(1);
    let input = Tensor::new(&[token_id], device)?.unsqueeze(0)?;
    // forward_looped commits loop-1 state advance and returns up to max_loops logits.
    let all_logits = model.forward_looped(&input, state, &[token_id], max_loops)?;

    // Walk the loops through the Kalman gate, deciding the realized count.
    let mut kf = KalmanFilter::from_params(cfg.kalman_q, cfg.kalman_r, cfg.kalman_init_var);
    let mut realized: Vec<Tensor> = Vec::with_capacity(all_logits.len());
    let mut raw_weights: Vec<f32> = Vec::with_capacity(all_logits.len());

    for (i, logits) in all_logits.iter().enumerate() {
        let c = group_confidence(logits, cfg.top_k)?;
        kf.update(c);
        realized.push(logits.clone());
        // Weight by filtered confidence: lower group-confidence = more confident,
        // so weight = 1 / (1 + filtered) keeps weights positive and rewards
        // confident loops. (Normalized later.)
        let filtered = kf.filtered_confidence();
        raw_weights.push(1.0 / (1.0 + filtered.max(0.0)));

        // Exit test (only after at least one loop, and never on the last):
        let is_last = i + 1 == all_logits.len();
        if !is_last {
            let confident = filtered <= cfg.conf_threshold;
            let plateaued = kf.velocity().abs() <= cfg.velocity_eps;
            if confident && plateaued {
                break;
            }
        }
    }

    let weights = normalize_weights(&raw_weights);
    let realized_loops = realized.len();
    Ok(LoopOutcome {
        per_loop_logits: realized,
        weights,
        realized_loops,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn confidence_low_for_peaked_high_for_uniform() {
        let dev = Device::Cpu;
        // Peaked: one huge logit -> confident -> low group-confidence value.
        let peaked = Tensor::new(&[10.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &dev).unwrap();
        // Uniform: all equal -> uncertain -> high group-confidence value.
        let uniform = Tensor::new(&[1.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], &dev).unwrap();
        let c_peaked = group_confidence(&peaked, 3).unwrap();
        let c_uniform = group_confidence(&uniform, 3).unwrap();
        assert!(c_peaked < c_uniform, "peaked={c_peaked} uniform={c_uniform}");
    }

    #[test]
    fn weights_normalize_to_one() {
        let w = normalize_weights(&[1.0, 2.0, 1.0]);
        let sum: f32 = w.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum={sum}");
    }

    #[test]
    fn zero_weights_fall_back_uniform() {
        let w = normalize_weights(&[0.0, 0.0]);
        assert_eq!(w, vec![0.5, 0.5]);
    }
}
