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
        Ok(Ok((config, tokenizer_path, weights_paths))) => {
            eprintln!("[model] config + tokenizer + weights loaded, building model");
            let tokenizer = Tokenizer::from_file(&tokenizer_path)
                .map_err(|e| anyhow::anyhow!("tokenizer load: {e}"))?;

            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(&weights_paths, dtype, &device)?
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
            let msg = format!(
                "Model weights for '{repo_id}' are not available locally and the download failed: {e}\n\
                 Run this once to download the model (~3 GB):\n\
                 \n  huggingface-cli download {repo_id}\n\
                 \nOr set HF_HUB_OFFLINE=1 and ensure weights are cached."
            );
            obs::info("model", json!({"event":"hf_download_failed","reason": e.to_string()}));
            anyhow::bail!("{msg}")
        },
        Err(_) => {
            let msg = format!(
                "Model weights for '{repo_id}' could not be loaded within 30s.\n\
                 The weights are likely not fully cached locally.\n\
                 Run this once to download the model (~3 GB):\n\
                 \n  huggingface-cli download {repo_id}\n"
            );
            obs::info("model", json!({"event":"hf_download_timeout"}));
            anyhow::bail!("{msg}")
        }
    }
}

async fn load_from_hf(repo: &hf_hub::api::tokio::ApiRepo, repo_id: &str) -> Result<(Config, PathBuf, Vec<PathBuf>)> {
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
    let weights_paths = list_safetensors(repo).await?;

    let config_json: Value = serde_json::from_slice(&std::fs::read(&config_path)?)?;
    let config = config_from_hf_json(&config_json)?;

    // If the first weight file uses HF naming (model.layers.*), remap to native RWKV (blocks.*).
    let weights_paths = maybe_remap_hf_weights(&weights_paths, &config)?;

    Ok((config, tokenizer_path, weights_paths))
}

/// Detect HF-format safetensors (model.layers.* keys) and rewrite to native RWKV format
/// (blocks.* keys) expected by candle-transformers rwkv_v7. The remapped file is cached
/// alongside the original with a `.rwkv-native.safetensors` suffix.
fn maybe_remap_hf_weights(paths: &[PathBuf], config: &Config) -> Result<Vec<PathBuf>> {
    use safetensors::{tensor::TensorView, SafeTensors};
    if paths.is_empty() {
        return Ok(paths.to_vec());
    }
    let first_bytes = std::fs::read(&paths[0])?;
    let first_st = SafeTensors::deserialize(&first_bytes).context("parse first safetensors")?;
    // Detect HF format by presence of "model.embeddings.weight"
    if !first_st.names().iter().any(|n| *n == "model.embeddings.weight") {
        return Ok(paths.to_vec());
    }

    let out_path = paths[0].with_extension("rwkv-native.safetensors");
    if out_path.exists() {
        eprintln!("[model] using cached remapped weights at {}", out_path.display());
        return Ok(vec![out_path]);
    }

    eprintln!("[model] HF key format detected — remapping to native RWKV format (this runs once)");

    let mut all_bytes: Vec<Vec<u8>> = Vec::new();
    for p in paths {
        all_bytes.push(std::fs::read(p)?);
    }
    let sts: Vec<SafeTensors> = all_bytes.iter()
        .map(|b| SafeTensors::deserialize(b).context("parse shard"))
        .collect::<Result<_>>()?;

    // Each entry: (native_name, data_bytes, dtype, shape)
    let mut remapped: Vec<(String, Vec<u8>, safetensors::Dtype, Vec<usize>)> = Vec::new();

    let get = |name: &str| -> Option<(Vec<u8>, safetensors::Dtype, Vec<usize>)> {
        for st in &sts {
            if let Ok(tv) = st.tensor(name) {
                return Some((tv.data().to_vec(), tv.dtype(), tv.shape().to_vec()));
            }
        }
        None
    };

    // Transpose a row-major 2D matrix stored as bytes (element_bytes = dtype size).
    let transpose2d = |data: Vec<u8>, rows: usize, cols: usize, elem: usize| -> Vec<u8> {
        let mut out = vec![0u8; data.len()];
        for r in 0..rows {
            for c in 0..cols {
                let src = (r * cols + c) * elem;
                let dst = (c * rows + r) * elem;
                out[dst..dst + elem].copy_from_slice(&data[src..src + elem]);
            }
        }
        out
    };

    // Reshape [H] → [1, 1, H] (same bytes, different shape metadata).
    let reshape_1h = |data: Vec<u8>, h: usize| -> (Vec<u8>, Vec<usize>) {
        (data, vec![1, 1, h])
    };

    macro_rules! push {
        ($native:expr, $hf:expr) => {
            if let Some((data, dtype, shape)) = get($hf) {
                remapped.push(($native.to_string(), data, dtype, shape));
            }
        };
    }
    macro_rules! push_reshape {
        ($native:expr, $hf:expr) => {
            if let Some((data, dtype, shape)) = get($hf) {
                let h = shape[0];
                let (data2, shape2) = reshape_1h(data, h);
                remapped.push(($native.to_string(), data2, dtype, shape2));
            }
        };
    }
    // lora.0 is [lora_dim, H] in HF, needs [H, lora_dim] for candle (w1/a1/v1/g1)
    macro_rules! push_lora0 {
        ($native:expr, $hf:expr) => {
            if let Some((data, dtype, shape)) = get($hf) {
                let elem = safetensors_dtype_size(dtype);
                let (lora_dim, h) = (shape[0], shape[1]);
                let data2 = transpose2d(data, lora_dim, h, elem);
                remapped.push(($native.to_string(), data2, dtype, vec![h, lora_dim]));
            }
        };
    }
    // lora.2.weight is [H, lora_dim] in HF, needs [lora_dim, H] for candle (w2/a2/v2/g2)
    macro_rules! push_lora2w {
        ($native:expr, $hf:expr) => {
            if let Some((data, dtype, shape)) = get($hf) {
                let elem = safetensors_dtype_size(dtype);
                let (h, lora_dim) = (shape[0], shape[1]);
                let data2 = transpose2d(data, h, lora_dim, elem);
                remapped.push(($native.to_string(), data2, dtype, vec![lora_dim, h]));
            }
        };
    }

    push!("emb.weight", "model.embeddings.weight");
    push!("ln_out.weight", "model.norm.weight");
    push!("ln_out.bias", "model.norm.bias");
    push!("head.weight", "lm_head.weight");

    for i in 0..config.num_hidden_layers {
        let hpfx = format!("model.layers.{i}");
        let npfx = format!("blocks.{i}");

        if i == 0 {
            push!(&format!("{npfx}.ln0.weight"), &format!("{hpfx}.pre_norm.weight"));
            push!(&format!("{npfx}.ln0.bias"), &format!("{hpfx}.pre_norm.bias"));
        }
        push!(&format!("{npfx}.ln1.weight"), &format!("{hpfx}.attn_norm.weight"));
        push!(&format!("{npfx}.ln1.bias"), &format!("{hpfx}.attn_norm.bias"));
        push!(&format!("{npfx}.ln2.weight"), &format!("{hpfx}.ffn_norm.weight"));
        push!(&format!("{npfx}.ln2.bias"), &format!("{hpfx}.ffn_norm.bias"));

        let ha = format!("{hpfx}.attn");
        let na = format!("{npfx}.att");
        for tok in ["x_r", "x_w", "x_k", "x_v", "x_a", "x_g"] {
            push!(&format!("{na}.{tok}"), &format!("{ha}.{tok}"));
        }
        push_lora0!(&format!("{na}.w1"), &format!("{ha}.w_lora.lora.0.weight"));
        push_lora2w!(&format!("{na}.w2"), &format!("{ha}.w_lora.lora.2.weight"));
        push_reshape!(&format!("{na}.w0"), &format!("{ha}.w_lora.lora.2.bias"));
        push_lora0!(&format!("{na}.a1"), &format!("{ha}.a_lora.lora.0.weight"));
        push_lora2w!(&format!("{na}.a2"), &format!("{ha}.a_lora.lora.2.weight"));
        push_reshape!(&format!("{na}.a0"), &format!("{ha}.a_lora.lora.2.bias"));
        push_lora0!(&format!("{na}.v1"), &format!("{ha}.v_lora.lora.0.weight"));
        push_lora2w!(&format!("{na}.v2"), &format!("{ha}.v_lora.lora.2.weight"));
        push_reshape!(&format!("{na}.v0"), &format!("{ha}.v_lora.lora.2.bias"));
        push_lora0!(&format!("{na}.g1"), &format!("{ha}.g_lora.lora.0.weight"));
        push_lora2w!(&format!("{na}.g2"), &format!("{ha}.g_lora.lora.2.weight"));
        push_reshape!(&format!("{na}.k_k"), &format!("{ha}.k_k"));
        push_reshape!(&format!("{na}.k_a"), &format!("{ha}.k_a"));
        push!(&format!("{na}.r_k"), &format!("{ha}.r_k"));
        push!(&format!("{na}.receptance.weight"), &format!("{ha}.r_proj.weight"));
        push!(&format!("{na}.key.weight"), &format!("{ha}.k_proj.weight"));
        push!(&format!("{na}.value.weight"), &format!("{ha}.v_proj.weight"));
        push!(&format!("{na}.output.weight"), &format!("{ha}.o_proj.weight"));
        push!(&format!("{na}.ln_x.weight"), &format!("{ha}.g_norm.weight"));
        push!(&format!("{na}.ln_x.bias"), &format!("{ha}.g_norm.bias"));

        let hf = format!("{hpfx}.ffn");
        let nf = format!("{npfx}.ffn");
        push_reshape!(&format!("{nf}.x_k"), &format!("{hf}.x_k"));
        push!(&format!("{nf}.key.weight"), &format!("{hf}.key.weight"));
        push!(&format!("{nf}.value.weight"), &format!("{hf}.value.weight"));
    }

    let views: Vec<(String, TensorView)> = remapped.iter()
        .map(|(name, data, dtype, shape)| {
            let tv = TensorView::new(*dtype, shape.clone(), data).expect("tensor view");
            (name.clone(), tv)
        })
        .collect();
    let view_refs: Vec<(&str, &TensorView)> = views.iter()
        .map(|(n, t)| (n.as_str(), t))
        .collect();
    safetensors::serialize_to_file(view_refs, &None, &out_path)
        .context("write remapped safetensors")?;
    eprintln!("[model] remapped weights written to {}", out_path.display());

    Ok(vec![out_path])
}

fn safetensors_dtype_size(dtype: safetensors::Dtype) -> usize {
    match dtype {
        safetensors::Dtype::F32 => 4,
        safetensors::Dtype::F16 | safetensors::Dtype::BF16 => 2,
        safetensors::Dtype::F64 | safetensors::Dtype::I64 | safetensors::Dtype::U64 => 8,
        safetensors::Dtype::I32 | safetensors::Dtype::U32 => 4,
        safetensors::Dtype::I16 | safetensors::Dtype::U16 => 2,
        safetensors::Dtype::I8 | safetensors::Dtype::U8 | safetensors::Dtype::BOOL => 1,
        _ => 2,
    }
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
