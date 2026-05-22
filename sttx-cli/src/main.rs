
use std::path::PathBuf;

use anyhow::Result;
use candle_core::{DType, Device};
use clap::{Parser, Subcommand, ValueEnum};
use serde_json::json;
use sttx_ccsniff::CcsniffStream;
use sttx_core::obs;
use sttx_train::checkpoint::{self, CheckpointMeta};
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
        } => {
            let device = if cpu { DeviceArg::Cpu } else { device };
            run_train(
                ccsniff_from, jsonl_from, steps, checkpoint_dir, checkpoint_every,
                model_repo, device, dtype, ctx_len, chunk_size, api_pairs,
            ).await
        }
        Cmd::Serve { checkpoint, port, max_tokens, device, dtype } => {
            serve::run(checkpoint, port, max_tokens, device.into_device()?, dtype.into_dtype()).await
        }
        Cmd::Inspect { checkpoint } => run_inspect(checkpoint),
        Cmd::MergeStats { checkpoint, top } => run_merge_stats(checkpoint, top),
        Cmd::ValidateData { ccsniff_from, jsonl_from, api_pairs } => {
            run_validate_data(ccsniff_from, jsonl_from, api_pairs).await
        }
        Cmd::QualityAssert { checkpoint, steps, model_repo, device, dtype } => {
            run_quality_assert(checkpoint, steps, model_repo, device.into_device()?, dtype.into_dtype()).await
        }
    }
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
) -> Result<()> {
    let device = device.into_device()?;
    let dtype = dtype.into_dtype();
    eprintln!("[cli] device={:?} dtype={:?} ctx_len={} chunk_size={}", device, dtype, ctx_len, chunk_size);
    obs::info("cli", json!({
        "event":"train_start","repo": repo,"steps": steps,"api_pairs": api_pairs,
        "ctx_len": ctx_len, "chunk_size": chunk_size, "dtype": format!("{:?}", dtype),
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
