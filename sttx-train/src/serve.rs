use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use serde_json::{json, Value};
use sttx_core::obs;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

use crate::checkpoint;
use crate::model::{self, fresh_state};
use crate::tokens;
use crate::train::build_state_prefix_pub;

pub struct InferenceEngine {
    model: crate::model::LoadedModel,
    state_prefix: Vec<Tensor>,
    max_tokens: usize,
}

impl InferenceEngine {
    pub async fn load(checkpoint_dir: PathBuf, max_tokens: usize) -> Result<Self> {
        let meta = checkpoint::load_meta(&checkpoint_dir)?;
        let device = Device::Cpu;
        let dtype = DType::F32;

        obs::info("serve", json!({"event":"loading","repo": meta.repo_id,"steps": meta.steps}));
        let loaded = model::load(&meta.repo_id, device, dtype).await?;

        let mut varmap = VarMap::new();
        checkpoint::load_into(&checkpoint_dir, &mut varmap)?;

        let vb = VarBuilder::from_varmap(&varmap, loaded.dtype, &loaded.device);
        let _ = vb;

        let state_prefix = build_state_prefix_pub(&loaded.config, &varmap, &loaded.device, loaded.dtype)?;
        obs::info("serve", json!({"event":"ready","steps": meta.steps}));

        Ok(Self { model: loaded, state_prefix, max_tokens })
    }

    pub fn generate(&self, prompt: &str) -> Result<String> {
        let ids = tokens::encode_text(&self.model.tokenizer, prompt)?;
        if ids.is_empty() {
            return Ok(String::new());
        }

        let mut state = fresh_state(&self.model.config, &self.model.device, self.model.dtype)?;
        for (i, prefix) in self.state_prefix.iter().enumerate() {
            state.per_layer[i].att_kv = prefix.clone();
        }

        let logits = self.model.model.forward_seq(&ids, &mut state)?;
        let next_id = argmax_last(&logits)?;

        let mut generated = vec![next_id];
        for _ in 1..self.max_tokens {
            let l = self.model.model.forward_seq(&[next_id], &mut state)?;
            let nid = argmax_last(&l)?;
            if nid == 0 {
                break;
            }
            generated.push(nid);
        }

        let text = self.model.tokenizer
            .decode(&generated, true)
            .map_err(|e| anyhow::anyhow!("decode: {e}"))?;
        Ok(text)
    }
}

fn argmax_last(logits: &Tensor) -> Result<u32> {
    let flat = logits.flatten_all()?;
    let v: Vec<f32> = flat.to_dtype(DType::F32)?.to_vec1()?;
    let idx = v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);
    Ok(idx as u32)
}

pub async fn run(checkpoint: PathBuf, port: u16, max_tokens: usize) -> Result<()> {
    let engine = Arc::new(InferenceEngine::load(checkpoint, max_tokens).await?);
    let listener = TcpListener::bind(format!("0.0.0.0:{port}")).await?;
    eprintln!("[serve] listening on 0.0.0.0:{port}");
    obs::info("serve", json!({"event":"listening","port": port}));

    loop {
        let (mut stream, addr) = listener.accept().await?;
        let eng = engine.clone();
        tokio::spawn(async move {
            if let Err(e) = handle_conn(&mut stream, eng).await {
                eprintln!("[serve] conn {addr} error: {e}");
            }
        });
    }
}

async fn handle_conn(
    stream: &mut tokio::net::TcpStream,
    engine: Arc<InferenceEngine>,
) -> Result<()> {
    let mut buf = vec![0u8; 65536];
    let n = stream.read(&mut buf).await?;
    let raw = std::str::from_utf8(&buf[..n]).unwrap_or("");

    let (method, path, body) = parse_http(raw);

    let response = if method == "POST" && path == "/v1/chat/completions" {
        match handle_chat(&body, &engine) {
            Ok(resp) => json_response(200, &resp),
            Err(e) => json_response(500, &json!({"error": e.to_string()}).to_string()),
        }
    } else if method == "GET" && path == "/health" {
        json_response(200, r#"{"status":"ok"}"#)
    } else {
        json_response(404, r#"{"error":"not found"}"#)
    };

    stream.write_all(response.as_bytes()).await?;
    Ok(())
}

fn handle_chat(body: &str, engine: &InferenceEngine) -> Result<String> {
    let v: Value = serde_json::from_str(body)?;
    let messages = v.get("messages").and_then(|m| m.as_array()).cloned().unwrap_or_default();

    let prompt = messages.iter().map(|m| {
        let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("user");
        let content = m.get("content").and_then(|c| c.as_str()).unwrap_or("");
        if role == "user" {
            format!("<human>{content}</human>\n")
        } else {
            format!("<bot>{content}</bot>\n")
        }
    }).collect::<String>() + "<bot>";

    let completion = engine.generate(&prompt)?;
    obs::info("serve", json!({"event":"completion","prompt_len": prompt.len(),"completion_len": completion.len()}));

    Ok(json!({
        "id": "cmpl-1",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": completion},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 0, "completion_tokens": completion.len(), "total_tokens": completion.len()}
    }).to_string())
}

fn parse_http(raw: &str) -> (&str, &str, String) {
    let mut lines = raw.splitn(2, "\r\n\r\n");
    let header = lines.next().unwrap_or("");
    let body = lines.next().unwrap_or("").to_string();
    let mut header_lines = header.lines();
    let first = header_lines.next().unwrap_or("");
    let mut parts = first.split_whitespace();
    let method = parts.next().unwrap_or("");
    let path = parts.next().unwrap_or("");
    (method, path, body)
}

fn json_response(status: u16, body: &str) -> String {
    let reason = if status == 200 { "OK" } else if status == 404 { "Not Found" } else { "Error" };
    format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
}
