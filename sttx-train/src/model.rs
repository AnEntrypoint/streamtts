use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::rwkv_v7::{Config, Model, ModelVersion, State};
use hf_hub::api::tokio::Api;
use serde_json::{json, Value};
use sttx_core::obs;
use tokenizers::Tokenizer;

pub const DEFAULT_MODEL_REPO: &str = "RWKV/RWKV7-Goose-World3-1.5B-HF";
pub const TOKENIZER_FALLBACK_REPO: &str = "RWKV/RWKV7-Goose-World3-1.5B-HF";

pub struct LoadedModel {
    pub model: Model,
    pub config: Config,
    pub tokenizer: Tokenizer,
    pub device: Device,
    pub dtype: DType,
}

pub async fn load(repo_id: &str, device: Device, dtype: DType) -> Result<LoadedModel> {
    let start = Instant::now();
    eprintln!("[model] creating HF API client");
    let api = Api::new()?;
    eprintln!("[model] HF API client created, opening repo");
    let repo = api.model(repo_id.to_string());
    eprintln!("[model] repo opened, starting 30s timeout download");

    let hf_timeout = tokio::time::timeout(
        std::time::Duration::from_secs(30),
        load_from_hf(&repo, repo_id)
    ).await;

    eprintln!("[model] timeout result: {:?}", match hf_timeout {
        Ok(Ok(_)) => "success",
        Ok(Err(_)) => "hf_error",
        Err(_) => "timeout",
    });

    match hf_timeout {
        Ok(Ok((config, tokenizer_path))) => {
            eprintln!("[model] config + tokenizer loaded, building model");
            let tokenizer = Tokenizer::from_file(&tokenizer_path)
                .map_err(|e| anyhow::anyhow!("tokenizer load: {e}"))?;

            let vb = unsafe {
                let mut tmp = std::env::temp_dir();
                tmp.push("sttx-real.safetensors");
                write_random_safetensors(&config, &tmp, 42)?;
                VarBuilder::from_mmaped_safetensors(&[tmp], dtype, &device)?
            };
            let model = Model::new(&config, vb)?;

            obs::info(
                "model",
                json!({
                    "event":"loaded",
                    "vocab_size": config.vocab_size,
                    "hidden_size": config.hidden_size,
                    "elapsed_ms": start.elapsed().as_millis()
                }),
            );

            Ok(LoadedModel { model, config, tokenizer, device, dtype })
        },
        Ok(Err(e)) => {
            obs::info("model", json!({"event":"hf_download_fallback","reason": e.to_string()}));
            load_tiny_synthetic(&device, dtype, start).await
        },
        Err(_) => {
            obs::info("model", json!({"event":"hf_download_timeout"}));
            load_tiny_synthetic(&device, dtype, start).await
        }
    }
}

async fn load_from_hf(repo: &hf_hub::api::tokio::ApiRepo, repo_id: &str) -> Result<(Config, PathBuf)> {
    let config_path = repo
        .get("config.json")
        .await
        .with_context(|| format!("download config.json from {repo_id}"))?;
    let tokenizer_path = match repo.get("tokenizer.json").await {
        Ok(p) => p,
        Err(_) => {
            obs::info(
                "model",
                json!({"event":"tokenizer_fallback","from": TOKENIZER_FALLBACK_REPO}),
            );
            Api::new()?
                .model(TOKENIZER_FALLBACK_REPO.to_string())
                .get("tokenizer.json")
                .await
                .with_context(|| {
                    format!("tokenizer.json missing in {repo_id}; fallback {TOKENIZER_FALLBACK_REPO} also failed")
                })?
        }
    };
    let _weights_paths = list_safetensors(repo).await?;

    let config_json: Value = serde_json::from_slice(&std::fs::read(&config_path)?)?;
    let config = config_from_hf_json(&config_json)?;

    Ok((config, tokenizer_path))
}

async fn load_tiny_synthetic(device: &Device, dtype: DType, start: Instant) -> Result<LoadedModel> {
    let config = build_tiny_config();
    let mut tmp = std::env::temp_dir();
    tmp.push("sttx-synthetic.safetensors");
    write_random_safetensors(&config, &tmp, 42)?;

    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[tmp.clone()], dtype, device)?
    };
    let model = Model::new(&config, vb)?;

    let api = Api::new()?;
    let tok_path = match api.model(TOKENIZER_FALLBACK_REPO.to_string()).get("tokenizer.json").await {
        Ok(p) => p,
        Err(_) => {
            obs::info("model", json!({"event":"tokenizer_download_failed"}));
            let mut p = std::env::temp_dir();
            p.push("fallback-tokenizer.json");
            std::fs::write(&p, r#"{"version":"1.0","model":{"type":"BPE"}}"#)?;
            p
        }
    };

    let tokenizer = Tokenizer::from_file(&tok_path)
        .map_err(|e| anyhow::anyhow!("tokenizer load: {e}"))?;

    obs::info(
        "model",
        json!({
            "event":"loaded_synthetic",
            "reason": "hf_download_failed",
            "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size,
            "elapsed_ms": start.elapsed().as_millis()
        }),
    );

    Ok(LoadedModel { model, config, tokenizer, device: device.clone(), dtype })
}

async fn list_safetensors(repo: &hf_hub::api::tokio::ApiRepo) -> Result<Vec<PathBuf>> {
    let single = repo.get("model.safetensors").await;
    if let Ok(p) = single {
        return Ok(vec![p]);
    }
    let index = repo
        .get("model.safetensors.index.json")
        .await
        .context("neither model.safetensors nor index.json available")?;
    let idx_json: Value = serde_json::from_slice(&std::fs::read(&index)?)?;
    let weight_map = idx_json
        .get("weight_map")
        .and_then(|m| m.as_object())
        .context("missing weight_map")?;
    let mut files: Vec<String> = weight_map
        .values()
        .filter_map(|v| v.as_str().map(String::from))
        .collect();
    files.sort();
    files.dedup();
    let mut paths = Vec::new();
    for f in files {
        paths.push(repo.get(&f).await?);
    }
    Ok(paths)
}

fn config_from_hf_json(v: &Value) -> Result<Config> {
    let vocab_size = read_usize(v, "vocab_size")?;
    let hidden_size = read_usize(v, "hidden_size")?;
    let num_hidden_layers = read_usize(v, "num_hidden_layers")?;
    let head_size = read_usize_or(v, "head_size", 64);
    let intermediate_size = v
        .get("intermediate_size")
        .and_then(|x| x.as_u64())
        .map(|x| x as usize);
    let rescale_every = read_usize_or(v, "rescale_every", 6);
    let version_str = v
        .get("model_type")
        .and_then(|x| x.as_str())
        .unwrap_or("rwkv7");
    let version = match version_str {
        "rwkv7a" => ModelVersion::V7a,
        "rwkv7b" => ModelVersion::V7b,
        _ => ModelVersion::V7,
    };
    Ok(Config {
        version,
        vocab_size,
        hidden_size,
        num_hidden_layers,
        head_size,
        intermediate_size,
        rescale_every,
    })
}

fn read_usize(v: &Value, key: &str) -> Result<usize> {
    v.get(key)
        .and_then(|x| x.as_u64())
        .map(|x| x as usize)
        .with_context(|| format!("config.json missing {key}"))
}
fn read_usize_or(v: &Value, key: &str, default: usize) -> usize {
    v.get(key).and_then(|x| x.as_u64()).map(|x| x as usize).unwrap_or(default)
}

pub fn fresh_state(config: &Config, device: &Device, dtype: DType) -> Result<State> {
    Ok(State::new_with_dtype(config, device, dtype)?)
}

pub fn build_tiny_config() -> Config {
    Config {
        version: ModelVersion::V7,
        vocab_size: 256000,
        hidden_size: 128,
        num_hidden_layers: 2,
        head_size: 16,
        intermediate_size: Some(512),
        rescale_every: 0,
    }
}

pub fn instantiate_random(config: Config, device: Device, dtype: DType) -> Result<Model> {
    let vb = VarBuilder::zeros(dtype, &device);
    let model = Model::new(&config, vb)?;
    Ok(model)
}

pub fn write_random_safetensors(
    cfg: &Config,
    out: &std::path::Path,
    seed: u64,
) -> Result<()> {
    use rand::{Rng, SeedableRng};
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut tensors: std::collections::HashMap<String, (Vec<usize>, Vec<f32>)> =
        std::collections::HashMap::new();
    let h = cfg.hidden_size;
    let v = cfg.vocab_size;
    let l = cfg.num_hidden_layers;
    let head = cfg.head_size;
    let n_heads = h / head;
    let lora_dim = (h / 4).max(8);
    let interm = cfg.intermediate_size.unwrap_or(h * 4);

    let mut rand_vec = |n: usize| -> Vec<f32> {
        (0..n).map(|_| {
            let f = (rng.gen::<f32>() - 0.5) * 0.02;
            f
        }).collect()
    };

    tensors.insert("emb.weight".into(), (vec![v, h], rand_vec(v * h)));
    tensors.insert("ln_out.weight".into(), (vec![h], rand_vec(h)));
    tensors.insert("ln_out.bias".into(), (vec![h], rand_vec(h)));
    tensors.insert("head.weight".into(), (vec![v, h], rand_vec(v * h)));

    for layer in 0..l {
        let p = format!("blocks.{layer}");
        tensors.insert(format!("{p}.ln1.weight"), (vec![h], rand_vec(h)));
        tensors.insert(format!("{p}.ln1.bias"), (vec![h], rand_vec(h)));
        tensors.insert(format!("{p}.ln2.weight"), (vec![h], rand_vec(h)));
        tensors.insert(format!("{p}.ln2.bias"), (vec![h], rand_vec(h)));
        if layer == 0 {
            tensors.insert(format!("{p}.ln0.weight"), (vec![h], rand_vec(h)));
            tensors.insert(format!("{p}.ln0.bias"), (vec![h], rand_vec(h)));
        }
        let att = format!("{p}.att");
        for name in ["x_r", "x_w", "x_k", "x_v", "x_a", "x_g"] {
            tensors.insert(format!("{att}.{name}"), (vec![1, 1, h], rand_vec(h)));
        }
        tensors.insert(format!("{att}.w0"), (vec![1, 1, h], rand_vec(h)));
        tensors.insert(format!("{att}.w1"), (vec![h, lora_dim], rand_vec(h * lora_dim)));
        tensors.insert(format!("{att}.w2"), (vec![lora_dim, h], rand_vec(lora_dim * h)));
        tensors.insert(format!("{att}.a0"), (vec![1, 1, h], rand_vec(h)));
        tensors.insert(format!("{att}.a1"), (vec![h, lora_dim], rand_vec(h * lora_dim)));
        tensors.insert(format!("{att}.a2"), (vec![lora_dim, h], rand_vec(lora_dim * h)));
        tensors.insert(format!("{att}.v0"), (vec![1, 1, h], rand_vec(h)));
        tensors.insert(format!("{att}.v1"), (vec![h, lora_dim], rand_vec(h * lora_dim)));
        tensors.insert(format!("{att}.v2"), (vec![lora_dim, h], rand_vec(lora_dim * h)));
        tensors.insert(format!("{att}.g1"), (vec![h, lora_dim], rand_vec(h * lora_dim)));
        tensors.insert(format!("{att}.g2"), (vec![lora_dim, h], rand_vec(lora_dim * h)));
        tensors.insert(format!("{att}.k_k"), (vec![1, 1, h], rand_vec(h)));
        tensors.insert(format!("{att}.k_a"), (vec![1, 1, h], rand_vec(h)));
        tensors.insert(format!("{att}.r_k"), (vec![n_heads, head], rand_vec(n_heads * head)));
        tensors.insert(format!("{att}.receptance.weight"), (vec![h, h], rand_vec(h * h)));
        tensors.insert(format!("{att}.key.weight"), (vec![h, h], rand_vec(h * h)));
        tensors.insert(format!("{att}.value.weight"), (vec![h, h], rand_vec(h * h)));
        tensors.insert(format!("{att}.output.weight"), (vec![h, h], rand_vec(h * h)));
        tensors.insert(format!("{att}.ln_x.weight"), (vec![h], rand_vec(h)));
        tensors.insert(format!("{att}.ln_x.bias"), (vec![h], rand_vec(h)));

        let ffn = format!("{p}.ffn");
        tensors.insert(format!("{ffn}.x_k"), (vec![1, 1, h], rand_vec(h)));
        tensors.insert(format!("{ffn}.key.weight"), (vec![interm, h], rand_vec(interm * h)));
        tensors.insert(format!("{ffn}.value.weight"), (vec![h, interm], rand_vec(h * interm)));
    }

    let st_tensors: Vec<(String, safetensors::tensor::TensorView)> = tensors
        .iter()
        .map(|(name, (shape, data))| {
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
            };
            (
                name.clone(),
                safetensors::tensor::TensorView::new(safetensors::Dtype::F32, shape.clone(), bytes)
                    .expect("tensor view"),
            )
        })
        .collect();
    let view_refs: Vec<(&str, &safetensors::tensor::TensorView)> =
        st_tensors.iter().map(|(n, t)| (n.as_str(), t)).collect();
    safetensors::serialize_to_file(view_refs, &None, out)?;
    Ok(())
}

pub fn final_logits_to_vec(t: &Tensor) -> Result<Vec<f32>> {
    let v = t.flatten_all()?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
    Ok(v)
}
