
use std::process::Stdio;
use std::time::Instant;

use anyhow::{anyhow, Context, Result};
use serde_json::json;
use sttx_core::obs;
use sttx_core::trace::Trace;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::mpsc;

pub struct CcsniffStream {
    rx: mpsc::Receiver<Trace>,
    _child: Option<KillOnDrop>,
}

struct KillOnDrop(Child);

impl Drop for KillOnDrop {
    fn drop(&mut self) {
        let _ = self.0.start_kill();
    }
}

impl CcsniffStream {
    pub async fn live(channel_capacity: usize) -> Result<Self> {
        ensure_ccsniff_installed().await?;
        let cmd = if cfg!(windows) { "npx.cmd" } else { "npx" };
        let mut child = Command::new(cmd)
            .args(["--yes", "ccsniff", "-f", "--json", "--full"])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .with_context(|| format!("spawn {cmd} ccsniff failed"))?;

        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow!("ccsniff child has no stdout"))?;
        let (tx, rx) = mpsc::channel(channel_capacity);

        tokio::spawn(async move {
            let reader = BufReader::new(stdout);
            let mut lines = reader.lines();
            loop {
                match lines.next_line().await {
                    Ok(Some(line)) => {
                        let trimmed = line.trim_start();
                        if trimmed.is_empty() || trimmed.starts_with('#') {
                            continue;
                        }
                        match serde_json::from_str::<serde_json::Value>(&line) {
                            Ok(v) => {
                                let trace = Trace::from_ccsniff_event(v);
                                obs::info(
                                    "ccsniff",
                                    json!({"event": "trace", "kind_idx": match &trace {
                                        Trace::UserMessage{..}=>0,
                                        Trace::AssistantMessage{..}=>1,
                                        Trace::ToolUse{..}=>2,
                                        Trace::ToolResult{..}=>3,
                                        Trace::Other{..}=>4
                                    }}),
                                );
                                if tx.send(trace).await.is_err() {
                                    obs::warn("ccsniff", json!({"event": "consumer_dropped"}));
                                    break;
                                }
                            }
                            Err(e) => obs::warn(
                                "ccsniff",
                                json!({"event":"parse_error","err": e.to_string(),"line_len": line.len()}),
                            ),
                        }
                    }
                    Ok(None) => {
                        obs::info("ccsniff", json!({"event":"eof"}));
                        break;
                    }
                    Err(e) => {
                        obs::error("ccsniff", json!({"event":"read_error","err": e.to_string()}));
                        break;
                    }
                }
            }
        });

        Ok(Self {
            rx,
            _child: Some(KillOnDrop(child)),
        })
    }

    pub async fn from_file(path: &str, channel_capacity: usize) -> Result<Self> {
        let content = tokio::fs::read_to_string(path)
            .await
            .with_context(|| format!("read {path}"))?;
        let (tx, rx) = mpsc::channel(channel_capacity);
        tokio::spawn(async move {
            for line in content.lines() {
                if line.trim().is_empty() {
                    continue;
                }
                match serde_json::from_str::<serde_json::Value>(line) {
                    Ok(v) => {
                        let trace = Trace::from_ccsniff_event(v);
                        if tx.send(trace).await.is_err() {
                            break;
                        }
                    }
                    Err(e) => obs::warn(
                        "ccsniff",
                        json!({"event":"file_parse_error","err": e.to_string()}),
                    ),
                }
            }
        });
        Ok(Self { rx, _child: None })
    }

    pub async fn recv(&mut self) -> Option<Trace> {
        self.rx.recv().await
    }
}

async fn ensure_ccsniff_installed() -> Result<()> {
    let start = Instant::now();
    let cmd = if cfg!(windows) { "npx.cmd" } else { "npx" };
    let out = Command::new(cmd)
        .args(["--yes", "ccsniff", "--list-projects"])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .output()
        .await;
    let elapsed_ms = start.elapsed().as_millis();
    match out {
        Ok(o) if o.status.success() => {
            let lines = String::from_utf8_lossy(&o.stdout).lines().count();
            obs::info(
                "ccsniff",
                json!({"event":"health_check","projects": lines,"elapsed_ms": elapsed_ms}),
            );
            Ok(())
        }
        Ok(o) => {
            let stderr = String::from_utf8_lossy(&o.stderr).to_string();
            Err(anyhow!(
                "ccsniff --list-projects failed (exit {:?}): {}",
                o.status.code(),
                stderr
            ))
        }
        Err(e) => Err(anyhow!("npx ccsniff --list-projects exec error: {e}")),
    }
}
