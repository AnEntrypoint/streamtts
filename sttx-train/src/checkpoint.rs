use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use candle_nn::VarMap;
use serde::{Deserialize, Serialize};
use serde_json::json;
use sttx_core::obs;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointMeta {
    pub steps: u64,
    pub recent_loss: Vec<f32>,
    pub merges_total: u64,
    pub promotions_total: u64,
    pub repo_id: String,
}

pub fn save(dir: &Path, meta: &CheckpointMeta, varmap: &VarMap) -> Result<PathBuf> {
    std::fs::create_dir_all(dir).with_context(|| format!("mkdir {}", dir.display()))?;
    let weights = dir.join("trainable.safetensors");
    varmap.save(&weights).context("save varmap")?;
    let meta_path = dir.join("meta.json");
    std::fs::write(&meta_path, serde_json::to_vec_pretty(meta)?)?;
    obs::info(
        "checkpoint",
        json!({"event":"saved","dir": dir.display().to_string(),"steps": meta.steps}),
    );
    Ok(meta_path)
}

pub fn load_meta(dir: &Path) -> Result<CheckpointMeta> {
    let meta_path = dir.join("meta.json");
    let buf = std::fs::read(&meta_path)
        .with_context(|| format!("read {}", meta_path.display()))?;
    Ok(serde_json::from_slice(&buf)?)
}

pub fn load_into(dir: &Path, varmap: &mut VarMap) -> Result<()> {
    use candle_core::Tensor;
    use safetensors::SafeTensors;
    let weights_path = dir.join("trainable.safetensors");
    let mmap_bytes = std::fs::read(&weights_path)
        .with_context(|| format!("read {}", weights_path.display()))?;
    let st = SafeTensors::deserialize(&mmap_bytes).context("parse safetensors")?;
    let data = varmap.data().lock().unwrap();
    let mut loaded = 0usize;
    let mut skipped = 0usize;
    for (name, var) in data.iter() {
        match st.tensor(name) {
            Ok(view) => {
                let dtype = var.dtype();
                let shape: Vec<usize> = view.shape().to_vec();
                let t = Tensor::from_raw_buffer(
                    view.data(),
                    candle_core::DType::F32,
                    &shape,
                    var.device(),
                )
                .and_then(|t| t.to_dtype(dtype))
                .with_context(|| format!("load tensor {name}"))?;
                var.set(&t).with_context(|| format!("set var {name}"))?;
                loaded += 1;
            }
            Err(_) => {
                skipped += 1;
                obs::info(
                    "checkpoint",
                    json!({"event":"key_missing","key": name,"action":"skip_init_kept"}),
                );
            }
        }
    }
    obs::info(
        "checkpoint",
        json!({"event":"loaded","dir": dir.display().to_string(),"loaded": loaded,"skipped": skipped}),
    );
    Ok(())
}
