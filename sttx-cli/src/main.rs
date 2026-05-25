
use std::path::PathBuf;

use anyhow::Result;
use candle_core::{DType, Device};
use clap::{Parser, Subcommand, ValueEnum};
use serde_json::json;
use sttx_ccsniff::CcsniffStream;
use sttx_core::obs;
use sttx_train::checkpoint::{self, CheckpointMeta};
use sttx_train::loop_infer::{group_confidence, LoopConfig};
use sttx_train::model::{self, DEFAULT_MODEL_REPO};
use sttx_train::serve;
use sttx_train::train::{TrainConfig, Trainer};

#[derive(Copy, Clone, Debug, ValueEnum)]
enum DTypeArg {
    Bf16,
    F16,
    F32,
}

impl DTypeArg {
    fn into_dtype(self) -> DType {
        match self {
            DTypeArg::Bf16 => DType::BF16,
            DTypeArg::F16 => DType::F16,
            DTypeArg::F32 => DType::F32,
        }
    }
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum DeviceArg {
    Auto,
    Cpu,
    Cuda,
}

impl DeviceArg {
    fn into_device(self) -> Result<Device> {
        Ok(match self {
            DeviceArg::Cpu => Device::Cpu,
            DeviceArg::Cuda => Device::new_cuda(0)?,
            DeviceArg::Auto => {
                if candle_core::utils::cuda_is_available() {
                    Device::new_cuda(0)?
                } else {
                    Device::Cpu
                }
            }
        })
    }
}

#[derive(Parser)]
#[command(
    name = "streamtts",
    version,
    about = "RWKV-7 streaming trainer and inference server"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    Train {
        #[arg(long, num_args = 1.., default_values_t = Vec::<String>::new())]
        ccsniff_from: Vec<String>,
        /// Read JSONL files produced by ai-data-extraction (`bun run extract:all`).
        /// Each line: {"messages":[{"role":"user|assistant","content":"..."}, ...]}.
        #[arg(long, num_args = 1.., default_values_t = Vec::<String>::new())]
        jsonl_from: Vec<String>,
        #[arg(long, default_value_t = 1000)]
        steps: u64,
        #[arg(long, default_value = "ckpt")]
        checkpoint_dir: PathBuf,
        #[arg(long, default_value_t = 100)]
        checkpoint_every: u64,
        #[arg(long, default_value = DEFAULT_MODEL_REPO)]
        model_repo: String,
        /// Deprecated alias for --device cpu.
        #[arg(long, default_value_t = false)]
        cpu: bool,
        #[arg(long, value_enum, default_value_t = DeviceArg::Auto)]
        device: DeviceArg,
        #[arg(long, value_enum, default_value_t = DTypeArg::Bf16)]
        dtype: DTypeArg,
        /// Maximum tokens of context per training step (truncated BPTT).
        /// Bounded only by memory once chunk-size keeps activations small.
        #[arg(long, default_value_t = 131072)]
        ctx_len: usize,
        /// Activation-bound chunk size for truncated BPTT.
        /// Memory is O(chunk_size), not O(ctx_len). Default 1024 is conservative.
        #[arg(long, default_value_t = 1024)]
        chunk_size: usize,
        #[arg(long, default_value_t = false)]
        api_pairs: bool,
        /// Adaptive looping: max latent-refinement loops per token (1 = off).
        #[arg(long, default_value_t = 1)]
        max_loops: usize,
        /// Adaptive looping: filtered group-confidence exit threshold (lower = more confident).
        #[arg(long, default_value_t = 0.5)]
        conf_threshold: f32,
        /// Load looping params from a loop-config.json (overrides --max-loops/--conf-threshold).
        #[arg(long)]
        loop_config: Option<PathBuf>,
    },
    Serve {
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long, default_value_t = 8080)]
        port: u16,
        #[arg(long, default_value_t = 200)]
        max_tokens: usize,
        #[arg(long, value_enum, default_value_t = DeviceArg::Auto)]
        device: DeviceArg,
        #[arg(long, value_enum, default_value_t = DTypeArg::Bf16)]
        dtype: DTypeArg,
        #[arg(long, default_value_t = 1)]
        max_loops: usize,
        #[arg(long, default_value_t = 0.5)]
        conf_threshold: f32,
        #[arg(long)]
        loop_config: Option<PathBuf>,
    },
    Inspect {
        #[arg(long)]
        checkpoint: PathBuf,
    },
    MergeStats {
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long, default_value_t = 25)]
        top: usize,
    },
    ValidateData {
        #[arg(long, num_args = 1.., default_values_t = Vec::<String>::new())]
        ccsniff_from: Vec<String>,
        #[arg(long, num_args = 1.., default_values_t = Vec::<String>::new())]
        jsonl_from: Vec<String>,
        #[arg(long, default_value_t = false)]
        api_pairs: bool,
    },
    QualityAssert {
        #[arg(long)]
        checkpoint: PathBuf,
        #[arg(long, default_value_t = 20)]
        steps: u64,
        #[arg(long, default_value = DEFAULT_MODEL_REPO)]
        model_repo: String,
        #[arg(long, value_enum, default_value_t = DeviceArg::Auto)]
        device: DeviceArg,
        #[arg(long, value_enum, default_value_t = DTypeArg::Bf16)]
        dtype: DTypeArg,
    },
    /// Sweep adaptive loops on a held-out batch with NO early-exit, fit the
    /// Kalman noise params + exit threshold + optimal loop count, and write
    /// loop-config.json. The "small-scale tests that find optimal params" step.
    CalibrateLoops {
        #[arg(long, num_args = 1.., default_values_t = Vec::<String>::new())]
        ccsniff_from: Vec<String>,
        #[arg(long, num_args = 1.., default_values_t = Vec::<String>::new())]
        jsonl_from: Vec<String>,
        #[arg(long, default_value = DEFAULT_MODEL_REPO)]
        model_repo: String,
        /// Upper bound on loops to sweep (the video found 3-4 optimal; sweep to 8).
        #[arg(long, default_value_t = 8)]
        sweep_loops: usize,
        /// How many tokens (sampled across the batch) to calibrate on.
        #[arg(long, default_value_t = 256)]
        samples: usize,
        /// Output path for the fitted loop config.
        #[arg(long, default_value = "loop-config.json")]
        out: PathBuf,
        #[arg(long, value_enum, default_value_t = DeviceArg::Auto)]
        device: DeviceArg,
        #[arg(long, value_enum, default_value_t = DTypeArg::Bf16)]
        dtype: DTypeArg,
        #[arg(long, default_value_t = false)]
        api_pairs: bool,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Train {
            ccsniff_from,
            jsonl_from,
            steps,
            checkpoint_dir,
            checkpoint_every,
            model_repo,
            cpu,
            device,
            dtype,
            ctx_len,
            chunk_size,
            api_pairs,
            max_loops,
            conf_threshold,
            loop_config,
        } => {
            let device = if cpu { DeviceArg::Cpu } else { device };
            let lc = resolve_loop_config(loop_config, max_loops, conf_threshold)?;
            run_train(
                ccsniff_from, jsonl_from, steps, checkpoint_dir, checkpoint_every,
                model_repo, device, dtype, ctx_len, chunk_size, api_pairs, lc,
            ).await
        }
        Cmd::Serve { checkpoint, port, max_tokens, device, dtype, max_loops, conf_threshold, loop_config } => {
            let lc = resolve_loop_config(loop_config, max_loops, conf_threshold)?;
            serve::run(checkpoint, port, max_tokens, device.into_device()?, dtype.into_dtype(), lc).await
        }
        Cmd::Inspect { checkpoint } => run_inspect(checkpoint),
        Cmd::MergeStats { checkpoint, top } => run_merge_stats(checkpoint, top),
        Cmd::ValidateData { ccsniff_from, jsonl_from, api_pairs } => {
            run_validate_data(ccsniff_from, jsonl_from, api_pairs).await
        }
        Cmd::QualityAssert { checkpoint, steps, model_repo, device, dtype } => {
            run_quality_assert(checkpoint, steps, model_repo, device.into_device()?, dtype.into_dtype()).await
        }
        Cmd::CalibrateLoops {
            ccsniff_from, jsonl_from, model_repo, sweep_loops, samples, out, device, dtype, api_pairs,
        } => {
            run_calibrate_loops(
                ccsniff_from, jsonl_from, model_repo, sweep_loops, samples, out,
                device.into_device()?, dtype.into_dtype(), api_pairs,
            ).await
        }
    }
}

/// Build a LoopConfig from a fitted file (if given) or from CLI flags. When a
/// file is provided it is the base; --max-loops/--conf-threshold still override
/// so a quick experiment doesn't require editing the file.
fn resolve_loop_config(
    path: Option<PathBuf>,
    max_loops: usize,
    conf_threshold: f32,
) -> Result<LoopConfig> {
    let mut lc = match path {
        Some(p) => LoopConfig::load(&p)?.ok_or_else(|| anyhow::anyhow!("loop-config not found: {}", p.display()))?,
        None => LoopConfig::default(),
    };
    // CLI flags override the file's max_loops/threshold when explicitly set.
    if max_loops != 1 {
        lc.max_loops = max_loops;
    }
    if (conf_threshold - 0.5).abs() > f32::EPSILON {
        lc.conf_threshold = conf_threshold;
    }
    Ok(lc)
}

#[allow(clippy::too_many_arguments)]
async fn run_train(
    ccsniff_from: Vec<String>,
    jsonl_from: Vec<String>,
    steps: u64,
    checkpoint_dir: PathBuf,
    checkpoint_every: u64,
    repo: String,
    device: DeviceArg,
    dtype: DTypeArg,
    ctx_len: usize,
    chunk_size: usize,
    api_pairs: bool,
    loop_config: LoopConfig,
) -> Result<()> {
    let device = device.into_device()?;
    let dtype = dtype.into_dtype();
    eprintln!("[cli] device={:?} dtype={:?} ctx_len={} chunk_size={} max_loops={}",
        device, dtype, ctx_len, chunk_size, loop_config.max_loops);
    obs::info("cli", json!({
        "event":"train_start","repo": repo,"steps": steps,"api_pairs": api_pairs,
        "ctx_len": ctx_len, "chunk_size": chunk_size, "dtype": format!("{:?}", dtype),
        "max_loops": loop_config.max_loops,
    }));

    eprintln!("[cli] about to load model from {}", repo);
    let model = model::load(&repo, device, dtype).await?;
    eprintln!("[cli] model loaded successfully");
    let cfg = TrainConfig {
        steps,
        checkpoint_dir: checkpoint_dir.clone(),
        checkpoint_every,
        ctx_len,
        chunk_size,
        loop_config,
        ..Default::default()
    };
    eprintln!("[cli] about to create trainer");
    let mut trainer = Trainer::new(model, cfg)?;
    eprintln!("[cli] trainer created successfully");

    let mut stream = open_stream(&ccsniff_from, &jsonl_from, api_pairs).await?;
    eprintln!("[cli] stream opened successfully");

    eprintln!("[cli] entering training loop, target steps: {}", steps);
    while trainer.steps < steps {
        let trace = match stream.recv().await {
            Some(t) => t,
            None => {
                eprintln!("[cli] stream ended at step {}", trainer.steps);
                obs::warn("cli", json!({"event":"stream_ended","steps": trainer.steps}));
                break;
            }
        };
        match trainer.step_on_trace(&trace) {
            Ok(_) => {
                if trainer.steps % 10 == 0 {
                    eprintln!("[cli] step {} complete", trainer.steps);
                }
            }
            Err(e) => {
                eprintln!("[cli] step error: {}", e);
                obs::error("cli", json!({"event":"step_error","err": e.to_string()}));
            }
        }
        if trainer.steps > 0 && trainer.steps % checkpoint_every == 0 {
            eprintln!("[cli] saving checkpoint at step {}", trainer.steps);
            checkpoint::save(&checkpoint_dir, &trainer.meta(repo.clone()), &trainer.trainable)?;
        }
    }
    eprintln!("[cli] training loop complete at step {}", trainer.steps);
    checkpoint::save(&checkpoint_dir, &trainer.meta(repo), &trainer.trainable)?;
    Ok(())
}

async fn open_stream(ccsniff_from: &[String], jsonl_from: &[String], api_pairs: bool) -> Result<CcsniffStream> {
    let ccsniff_sources: Vec<&str> = ccsniff_from.iter().map(String::as_str).collect();
    let jsonl_sources: Vec<&str> = jsonl_from.iter().map(String::as_str).collect();

    if !jsonl_sources.is_empty() && !ccsniff_sources.is_empty() {
        anyhow::bail!("pass either --ccsniff-from or --jsonl-from, not both");
    }
    if !jsonl_sources.is_empty() {
        return CcsniffStream::from_jsonl_messages(&jsonl_sources, 64).await;
    }
    if ccsniff_sources == ["live"] {
        return CcsniffStream::live(64).await;
    }
    if api_pairs {
        CcsniffStream::from_files_paired(&ccsniff_sources, 64).await
    } else {
        CcsniffStream::from_files(&ccsniff_sources, 64, false).await
    }
}

fn run_inspect(checkpoint: PathBuf) -> Result<()> {
    let meta = checkpoint::load_meta(&checkpoint)?;
    println!("{}", serde_json::to_string_pretty(&meta)?);
    Ok(())
}

fn run_merge_stats(checkpoint: PathBuf, top: usize) -> Result<()> {
    let meta: CheckpointMeta = checkpoint::load_meta(&checkpoint)?;
    println!(
        "{}",
        json!({
            "checkpoint": checkpoint.display().to_string(),
            "merges_total": meta.merges_total,
            "promotions_total": meta.promotions_total,
            "top": top,
            "note": "per-pair stats are observed live; rerun training with --top-merges to dump pairs"
        })
    );
    Ok(())
}

async fn run_validate_data(ccsniff_from: Vec<String>, jsonl_from: Vec<String>, api_pairs: bool) -> Result<()> {
    let mut stream = open_stream(&ccsniff_from, &jsonl_from, api_pairs).await?;
    let mut count = 0u64;
    let mut non_empty = 0u64;
    while let Some(trace) = stream.recv().await {
        count += 1;
        if !trace.corpus_text().is_empty() {
            non_empty += 1;
        }
    }
    println!("{}", json!({
        "ccsniff_from": ccsniff_from,
        "jsonl_from": jsonl_from,
        "api_pairs": api_pairs,
        "traces_total": count,
        "traces_non_empty": non_empty,
    }));
    Ok(())
}

async fn run_quality_assert(checkpoint: PathBuf, steps: u64, model_repo: String, device: Device, dtype: DType) -> Result<()> {
    let meta = checkpoint::load_meta(&checkpoint)?;
    eprintln!("[quality] checkpoint steps={} recent_loss_mean={:.4}",
        meta.steps,
        meta.recent_loss.iter().copied().sum::<f32>() / meta.recent_loss.len().max(1) as f32
    );
    let model = model::load(&model_repo, device, dtype).await?;
    let cfg = TrainConfig { steps, checkpoint_every: steps + 1, ..Default::default() };
    let mut trainer = Trainer::new(model, cfg)?;
    checkpoint::load_into(&checkpoint, &mut trainer.trainable)?;
    let probe = "Hello, how can I help you today? I can assist with coding, writing, analysis, and more.";
    let base_ids = sttx_train::tokens::encode_text(&trainer.model.tokenizer, probe)?;
    let loss_before = trainer.forward_loss(&base_ids)?.to_dtype(DType::F32)?.to_vec0::<f32>()?;
    let mut loss_after = loss_before;
    for _ in 0..steps {
        loss_after = trainer.step_on_ids(&base_ids)?;
    }
    let mean_saved = meta.recent_loss.iter().copied().sum::<f32>() / meta.recent_loss.len().max(1) as f32;
    println!("{}", json!({
        "checkpoint": checkpoint.display().to_string(),
        "saved_mean_loss": mean_saved,
        "loss_before_probe_steps": loss_before,
        "loss_after_probe_steps": loss_after,
        "probe_steps": steps,
        "assessment": if loss_after < loss_before { "improving" } else { "not_improving" },
    }));
    Ok(())
}

/// Calibrate looping params on a held-out batch. Sweeps `sweep_loops` with NO
/// early exit, records per-loop group-confidence and CE across `samples` token
/// positions, then fits:
///   - optimal max_loops = first loop where marginal mean-CE improvement < 1%
///     (the diminishing-returns knee; the video found 3-4 on real benchmarks),
///   - conf_threshold = mean group-confidence observed at that loop,
///   - kalman_r = variance of per-loop confidence (measurement noise),
///   - kalman_q = a small fraction of r (process noise),
/// and writes loop-config.json.
#[allow(clippy::too_many_arguments)]
async fn run_calibrate_loops(
    ccsniff_from: Vec<String>,
    jsonl_from: Vec<String>,
    model_repo: String,
    sweep_loops: usize,
    samples: usize,
    out: PathBuf,
    device: Device,
    dtype: DType,
    api_pairs: bool,
) -> Result<()> {
    let sweep = sweep_loops.max(2);
    eprintln!("[calibrate] loading model {model_repo} (sweep_loops={sweep}, samples={samples})");
    let model = model::load(&model_repo, device, dtype).await?;
    // A trainer gives us the seeded state-prefix and a fresh state builder.
    let cfg = TrainConfig { steps: 0, checkpoint_every: 1, ..Default::default() };
    let trainer = Trainer::new(model, cfg)?;

    // Collect sample token sequences from the data stream.
    let mut stream = open_stream(&ccsniff_from, &jsonl_from, api_pairs).await?;
    let mut token_pool: Vec<u32> = Vec::new();
    while token_pool.len() < samples + 1 {
        match stream.recv().await {
            Some(trace) => {
                let text = trace.corpus_text();
                if text.is_empty() { continue; }
                if let Ok(ids) = sttx_train::tokens::encode_text(&trainer.model.tokenizer, &text) {
                    token_pool.extend(ids);
                }
            }
            None => break,
        }
    }
    if token_pool.len() < 2 {
        anyhow::bail!("calibrate: not enough tokens in the data stream");
    }
    token_pool.truncate(samples + 1);

    // Per-loop accumulators across all sampled positions.
    let mut conf_sum = vec![0.0f64; sweep];
    let mut conf_sq = vec![0.0f64; sweep];
    let mut ce_sum = vec![0.0f64; sweep];
    let mut n_positions = 0usize;

    // Build one state seeded with the prefix, walk the token pool, and at each
    // position run forward_looped with NO early exit to read the full sweep.
    let mut state = sttx_train::model::fresh_state(&trainer.model.config, &trainer.model.device, trainer.model.dtype)?;
    for (i, prefix) in trainer.state_prefix.iter().enumerate() {
        state.per_layer[i].att_kv = prefix.clone();
    }

    for w in token_pool.windows(2) {
        let input = w[0];
        let target = w[1];
        let snap = sttx_train::loop_infer::StateSnapshot::capture(&state);
        let logits_per_loop = {
            let t = candle_core::Tensor::new(&[input], &trainer.model.device)?.unsqueeze(0)?;
            trainer.model.model.forward_looped(&t, &mut state, &[input], sweep)?
        };
        // The committed loop-1 advance is what carries the chain forward; restore
        // then re-advance by exactly one real step so positions stay aligned.
        snap.restore(&mut state);
        let _ = trainer.model.model.forward_seq(&[input], &mut state)?;

        for (li, logits) in logits_per_loop.iter().enumerate().take(sweep) {
            let c = group_confidence(logits, 5)? as f64;
            conf_sum[li] += c;
            conf_sq[li] += c * c;
            let ce = sttx_train::loop_infer::cross_entropy_one(logits, target)? as f64;
            ce_sum[li] += ce;
        }
        n_positions += 1;
    }

    if n_positions == 0 {
        anyhow::bail!("calibrate: no positions evaluated");
    }
    let nf = n_positions as f64;
    let mean_conf: Vec<f64> = conf_sum.iter().map(|s| s / nf).collect();
    let mean_ce: Vec<f64> = ce_sum.iter().map(|s| s / nf).collect();
    // Measurement-noise variance: average per-loop variance of confidence.
    let var_conf: f64 = (0..sweep)
        .map(|i| (conf_sq[i] / nf) - mean_conf[i] * mean_conf[i])
        .map(|v| v.max(0.0))
        .sum::<f64>()
        / sweep as f64;

    // Optimal loop count = knee where marginal mean-CE improvement < 1%.
    let mut optimal = 1usize;
    for i in 1..sweep {
        let prev = mean_ce[i - 1];
        let cur = mean_ce[i];
        let rel_improve = if prev.abs() > 1e-9 { (prev - cur) / prev.abs() } else { 0.0 };
        if rel_improve >= 0.01 {
            optimal = i + 1; // loops are 1-indexed for the user
        } else {
            break;
        }
    }

    let r = var_conf.max(1e-4) as f32;
    let lc = LoopConfig {
        max_loops: optimal.max(1),
        conf_threshold: mean_conf[optimal.saturating_sub(1)] as f32,
        velocity_eps: 0.01,
        kalman_q: (r * 0.1).max(1e-4),
        kalman_r: r,
        kalman_init_var: 1.0,
        top_k: 5,
    };
    lc.save(&out)?;

    println!("{}", json!({
        "out": out.display().to_string(),
        "sweep_loops": sweep,
        "positions": n_positions,
        "mean_confidence_per_loop": mean_conf,
        "mean_ce_per_loop": mean_ce,
        "fitted": {
            "max_loops": lc.max_loops,
            "conf_threshold": lc.conf_threshold,
            "kalman_q": lc.kalman_q,
            "kalman_r": lc.kalman_r,
        },
        "note": "optimal max_loops = knee where marginal CE improvement < 1%",
    }));
    Ok(())
}
