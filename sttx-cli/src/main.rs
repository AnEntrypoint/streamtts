
use std::path::PathBuf;

use anyhow::Result;
use candle_core::{DType, Device};
use clap::{Parser, Subcommand};
use serde_json::json;
use sttx_ccsniff::CcsniffStream;
use sttx_core::obs;
use sttx_train::checkpoint::{self, CheckpointMeta};
use sttx_train::model::{self, DEFAULT_MODEL_REPO};
use sttx_train::train::{TrainConfig, Trainer};

#[derive(Parser)]
#[command(
    name = "streamtts",
    version,
    about = "RWKV-7 streaming trainer with online dynamic tokenization and surprise replay"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    Train {
        #[arg(long)]
        ccsniff_from: String,
        #[arg(long, default_value_t = 1000)]
        steps: u64,
        #[arg(long, default_value = "ckpt")]
        checkpoint_dir: PathBuf,
        #[arg(long, default_value_t = 100)]
        checkpoint_every: u64,
        #[arg(long, default_value = DEFAULT_MODEL_REPO)]
        model_repo: String,
        #[arg(long, default_value_t = false)]
        cpu: bool,
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
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Train {
            ccsniff_from,
            steps,
            checkpoint_dir,
            checkpoint_every,
            model_repo,
            cpu,
        } => run_train(ccsniff_from, steps, checkpoint_dir, checkpoint_every, model_repo, cpu).await,
        Cmd::Inspect { checkpoint } => run_inspect(checkpoint),
        Cmd::MergeStats { checkpoint, top } => run_merge_stats(checkpoint, top),
    }
}

async fn run_train(
    ccsniff_from: String,
    steps: u64,
    checkpoint_dir: PathBuf,
    checkpoint_every: u64,
    repo: String,
    cpu: bool,
) -> Result<()> {
    let device = Device::Cpu;
    let dtype = DType::F32;
    eprintln!("[cli] device init complete");
    obs::info("cli", json!({"event":"train_start","repo": repo,"steps": steps,"device_cuda": device.is_cuda()}));

    eprintln!("[cli] about to load model from {}", repo);
    let model = model::load(&repo, device, dtype).await?;
    eprintln!("[cli] model loaded successfully");
    let cfg = TrainConfig {
        steps,
        checkpoint_dir: checkpoint_dir.clone(),
        checkpoint_every,
        ..Default::default()
    };
    eprintln!("[cli] about to create trainer");
    let mut trainer = Trainer::new(model, cfg)?;
    eprintln!("[cli] trainer created successfully");

    eprintln!("[cli] about to open ccsniff stream from {}", ccsniff_from);
    let mut stream = match ccsniff_from.as_str() {
        "live" => CcsniffStream::live(64).await?,
        path => CcsniffStream::from_file(path, 64).await?,
    };
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
