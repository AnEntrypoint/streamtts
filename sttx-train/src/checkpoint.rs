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
    let weights = dir.join("trainable.safetensors");
    varmap.load(&weights).context("load varmap")?;
    obs::info(
        "checkpoint",
        json!({"event":"loaded","dir": dir.display().to_string()}),
    );
    Ok(())
}
