use std::path::PathBuf;

use anyhow::Result;
use candle_core::{DType, Device};
use sttx_core::trace::Trace;
use sttx_train::model;
use sttx_train::train::{TrainConfig, Trainer};

#[tokio::main]
async fn main() -> Result<()> {
    let repo = std::env::var("STTX_MODEL_REPO").unwrap_or_else(|_| model::DEFAULT_MODEL_REPO.to_string());
    eprintln!("smoke_train: model load");
    let device = Device::Cpu;
    let dtype = DType::BF16;
    let model = model::load(&repo, device.clone(), dtype).await?;

    let cfg = TrainConfig {
        steps: 5,
        max_tokens_per_step: 64,
        replay_capacity: 32,
        replay_per_step: 1,
        replay_weight: 0.1,
        lr: 1e-4,
        checkpoint_dir: PathBuf::from("ckpt-smoke"),
        checkpoint_every: 1000,
        seed: 42,
    };
    let mut trainer = Trainer::new(model, cfg)?;

    let synthetic = [
        "fn main() { println!(\"hello world\"); }",
        "let x = vec![1, 2, 3]; for v in x { println!(\"{v}\"); }",
        "the quick brown fox jumps over the lazy dog",
        "RWKV-7 streaming trainer with dynamic tokenization",
        "match cfg { Some(c) => process(c), None => default() }",
    ];

    let mut losses = Vec::new();
    for (i, txt) in synthetic.iter().enumerate() {
        let trace = Trace::UserMessage { text: (*txt).into() };
        let loss = trainer.step_on_trace(&trace)?;
        eprintln!("step {i}: loss={loss}");
        losses.push(loss);
    }

    eprintln!("losses: {losses:?}");
    sttx_train::checkpoint::save(
        &PathBuf::from("ckpt-smoke"),
        &trainer.meta(repo.clone()),
        &trainer.trainable,
    )?;
    println!(
        "{{\"steps\":{},\"losses\":{:?},\"merges\":{},\"promotions\":{}}}",
        trainer.steps, losses, trainer.dyn_tk.merges_total, trainer.dyn_tk.promotions_total
    );
    Ok(())
}
