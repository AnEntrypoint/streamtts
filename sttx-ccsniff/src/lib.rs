
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
                                        Trace::Pair{..}=>5,
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
        Self::from_files(&[path], channel_capacity, false).await
    }

    pub async fn from_files_paired(paths: &[&str], channel_capacity: usize) -> Result<Self> {
        Self::from_files(paths, channel_capacity, true).await
    }

    pub async fn from_files(paths: &[&str], channel_capacity: usize, pair_mode: bool) -> Result<Self> {
        let mut raw_events: Vec<serde_json::Value> = Vec::new();
        for path in paths {
            eprintln!("[ccsniff] reading file: {}", path);
            let content = std::fs::read_to_string(path)
                .with_context(|| format!("read {path}"))?;
            let line_count = content.lines().count();
            eprintln!("[ccsniff] file read, {} lines total", line_count);
            for line in content.lines() {
                if line.trim().is_empty() { continue; }
                match serde_json::from_str::<serde_json::Value>(line) {
                    Ok(v) => raw_events.push(v),
                    Err(e) => eprintln!("[ccsniff] parse error: {}", e),
                }
            }
        }

        let traces = if pair_mode {
            build_pairs(raw_events)
        } else {
            raw_events.into_iter().map(Trace::from_ccsniff_event).collect()
        };

        eprintln!("[ccsniff] parsed {} traces", traces.len());
        let (tx, rx) = mpsc::channel(channel_capacity);
        tokio::spawn(async move {
            eprintln!("[ccsniff] spawned task started, sending {} traces", traces.len());
            for (i, trace) in traces.into_iter().enumerate() {
                if (i + 1) % 10 == 0 {
                    eprintln!("[ccsniff] sent {} traces", i + 1);
                }
                if tx.send(trace).await.is_err() {
                    eprintln!("[ccsniff] consumer dropped after {} traces", i + 1);
                    break;
                }
            }
            eprintln!("[ccsniff] task complete, all traces sent");
        });
        eprintln!("[ccsniff] returning stream");
        Ok(Self { rx, _child: None })
    }

    pub async fn recv(&mut self) -> Option<Trace> {
        self.rx.recv().await
    }
}

fn build_pairs(events: Vec<serde_json::Value>) -> Vec<Trace> {
    let mut out = Vec::with_capacity(events.len() / 2);
    let mut i = 0;
    while i < events.len() {
        let ev = &events[i];
        if Trace::is_prompt_role(ev) {
            let prompt_text = {
                let t = Trace::from_ccsniff_event(ev.clone());
                let s = t.corpus_text();
                if s.is_empty() { i += 1; continue; }
                s
            };
            if i + 1 < events.len() && Trace::is_completion_for(ev, &events[i + 1]) {
                let completion_text = {
                    let t = Trace::from_ccsniff_event(events[i + 1].clone());
                    t.corpus_text()
                };
                if !completion_text.is_empty() {
                    out.push(Trace::Pair { prompt: prompt_text, completion: completion_text });
                    i += 2;
                    continue;
                }
            }
            out.push(Trace::from_ccsniff_event(ev.clone()));
        }
        i += 1;
    }
    out
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
