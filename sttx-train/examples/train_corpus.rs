use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use candle_core::{DType, Device};
use serde_json::json;
use sttx_ccsniff::CcsniffStream;
use sttx_core::obs;
use sttx_train::checkpoint;
use sttx_train::model::{self, LoadedModel};
use sttx_train::train::{TrainConfig, Trainer};
use tokenizers::{models::wordlevel::WordLevel, Tokenizer};

fn build_passthrough_tokenizer(vocab_size: usize) -> Tokenizer {
    let mut vocab: ahash::AHashMap<String, u32> = ahash::AHashMap::new();
    for i in 0..vocab_size.saturating_sub(1) {
        vocab.insert(format!("__t{i}"), i as u32);
    }
    vocab.insert("[UNK]".into(), (vocab_size - 1) as u32);
    let model = WordLevel::builder()
        .vocab(vocab)
        .unk_token("[UNK]".into())
        .build()
        .expect("wordlevel");
    Tokenizer::new(model)
}

fn text_to_ids(text: &str, vocab_size: usize) -> Vec<u32> {
    text.bytes()
        .map(|b| (b as u32) % (vocab_size as u32))
        .collect()
}

#[tokio::main]
async fn main() -> Result<()> {
    let corpus = std::env::var("STTX_CORPUS")
        .unwrap_or_else(|_| "corpus/ccsniff-history.ndjson".to_string());
    let max_steps: u64 = std::env::var("STTX_STEPS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(500);
    let ckpt_dir =
        PathBuf::from(std::env::var("STTX_CKPT").unwrap_or_else(|_| "ckpt-corpus".into()));

    let device = Device::Cpu;
    let dtype = DType::F32;
    let cfg = model::build_tiny_config();
    let st_path = std::env::temp_dir().join("sttx-corpus-rwkv7.safetensors");
    model::write_random_safetensors(&cfg, &st_path, 1337)?;
    let vb = unsafe {
        candle_nn::VarBuilder::from_mmaped_safetensors(&[st_path.clone()], dtype, &device)?
    };
    let rwkv = candle_transformers::models::rwkv_v7::Model::new(&cfg, vb)?;
    let tokenizer = build_passthrough_tokenizer(cfg.vocab_size);
    let vocab_size = cfg.vocab_size;
    let loaded = LoadedModel {
        model: rwkv,
        config: cfg,
        tokenizer,
        device: device.clone(),
        dtype,
    };

    obs::info(
        "corpus",
        json!({"event":"start","corpus": corpus,"max_steps": max_steps}),
    );

    let train_cfg = TrainConfig {
        steps: max_steps,
        max_tokens_per_step: 32,
        replay_capacity: 256,
        replay_per_step: 0,
        replay_weight: 0.1,
        lr: 5e-3,
        checkpoint_dir: ckpt_dir.clone(),
        checkpoint_every: 1_000_000,
        seed: 11,
    };
    let mut trainer = Trainer::new(loaded, train_cfg)?;

    let mut stream = CcsniffStream::from_file(&corpus, 256).await?;
    let started = Instant::now();
    let mut first_window: Vec<f32> = Vec::new();
    let mut last_window: Vec<f32> = Vec::new();
    let mut nonempty_steps: u64 = 0;
    let mut consumed: u64 = 0;

    while trainer.steps < max_steps {
        let trace = match stream.recv().await {
            Some(t) => t,
            None => break,
        };
        consumed += 1;
        let text = trace.corpus_text();
        if text.is_empty() {
            continue;
        }
        let ids = text_to_ids(&text, vocab_size);
        match trainer.step_on_ids(&ids) {
            Ok(loss) => {
                if loss > 0.0 {
                    nonempty_steps += 1;
                    if first_window.len() < 32 {
                        first_window.push(loss);
                    }
                    last_window.push(loss);
                    if last_window.len() > 32 {
                        last_window.remove(0);
                    }
                }
            }
            Err(e) => obs::warn("corpus", json!({"event":"step_err","err": e.to_string()})),
        }
    }

    std::fs::create_dir_all(&ckpt_dir)?;
    checkpoint::save(
        &ckpt_dir,
        &trainer.meta("tiny-corpus".into()),
        &trainer.trainable,
    )?;

    let mean = |v: &[f32]| -> f32 {
        if v.is_empty() {
            f32::NAN
        } else {
            v.iter().sum::<f32>() / v.len() as f32
        }
    };
    let summary = json!({
        "event":"end",
        "elapsed_s": started.elapsed().as_secs_f32(),
        "consumed_traces": consumed,
        "trainer_steps": trainer.steps,
        "nonempty_steps": nonempty_steps,
        "first_window_mean": mean(&first_window),
        "last_window_mean": mean(&last_window),
        "merges_total": trainer.dyn_tk.merges_total,
        "promotions_total": trainer.dyn_tk.promotions_total,
        "replay_len": trainer.replay.len(),
        "ckpt_dir": ckpt_dir.display().to_string(),
    });
    obs::info("corpus", summary.clone());
    println!("{}", serde_json::to_string_pretty(&summary)?);
    Ok(())
}
