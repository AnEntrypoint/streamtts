
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
        let ccsniff_dev = std::path::Path::new("C:/dev/ccsniff/src/cli.js");
        let (prog, args): (&str, Vec<&str>) = if ccsniff_dev.exists() {
            ("node", vec!["C:/dev/ccsniff/src/cli.js", "-f", "--json", "--full"])
        } else {
            ensure_ccsniff_installed().await?;
            if cfg!(windows) {
                ("npx.cmd", vec!["--yes", "ccsniff", "-f", "--json", "--full"])
            } else {
                ("npx", vec!["--yes", "ccsniff", "-f", "--json", "--full"])
            }
        };
        let mut child = Command::new(prog)
            .args(&args)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .with_context(|| format!("spawn {prog} ccsniff failed"))?;

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

    /// Read JSONL files produced by ai-data-extraction (`bun run extract:all`).
    /// Each line is either `{"messages":[{"role","content"},...]}` (default),
    /// `{"conversations":[{"from","value"},...]}` (sharegpt), or
    /// `{"text":"<chat-templated string>"}` (gemma4). We support the first two
    /// natively; the third is emitted as a single AssistantMessage.
    /// Each line yields one Trace::Pair per user→assistant turn pair, plus
    /// orphan messages as plain User/AssistantMessage traces.
    pub async fn from_jsonl_messages(paths: &[&str], channel_capacity: usize) -> Result<Self> {
        let mut traces: Vec<Trace> = Vec::new();
        for path in paths {
            eprintln!("[ccsniff/jsonl] reading {}", path);
            let content = std::fs::read_to_string(path)
                .with_context(|| format!("read {path}"))?;
            for line in content.lines() {
                if line.trim().is_empty() { continue; }
                let v: serde_json::Value = match serde_json::from_str(line) {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("[ccsniff/jsonl] parse error: {}", e);
                        continue;
                    }
                };
                push_jsonl_traces(&v, &mut traces);
            }
        }
        eprintln!("[ccsniff/jsonl] parsed {} traces", traces.len());
        let (tx, rx) = mpsc::channel(channel_capacity);
        tokio::spawn(async move {
            for trace in traces.into_iter() {
                if tx.send(trace).await.is_err() {
                    break;
                }
            }
        });
        Ok(Self { rx, _child: None })
    }
}

fn push_jsonl_traces(v: &serde_json::Value, out: &mut Vec<Trace>) {
    // messages format
    if let Some(msgs) = v.get("messages").and_then(|m| m.as_array()) {
        consume_role_value_pairs(msgs, "role", "content", out);
        return;
    }
    // sharegpt format
    if let Some(convs) = v.get("conversations").and_then(|m| m.as_array()) {
        consume_role_value_pairs(convs, "from", "value", out);
        return;
    }
    // gemma4 pre-rendered text
    if let Some(t) = v.get("text").and_then(|t| t.as_str()) {
        if !t.is_empty() {
            out.push(Trace::AssistantMessage { text: t.to_string() });
        }
    }
}

fn consume_role_value_pairs(arr: &[serde_json::Value], role_key: &str, text_key: &str, out: &mut Vec<Trace>) {
    let mut pending_user: Option<String> = None;
    for m in arr {
        let role = m.get(role_key).and_then(|r| r.as_str()).unwrap_or("");
        let content = m.get(text_key).and_then(|c| c.as_str()).unwrap_or("").to_string();
        if content.is_empty() { continue; }
        let normalized_role = match role {
            "user" | "human" => "user",
            "assistant" | "gpt" | "model" => "assistant",
            "system" => "system",
            _ => "other",
        };
        match normalized_role {
            "user" => {
                if let Some(prev) = pending_user.take() {
                    out.push(Trace::UserMessage { text: prev });
                }
                pending_user = Some(content);
            }
            "assistant" => {
                if let Some(prompt) = pending_user.take() {
                    out.push(Trace::Pair { prompt, completion: content });
                } else {
                    out.push(Trace::AssistantMessage { text: content });
                }
            }
            _ => {
                // System / unknown -> emit as user-side context to feed the model.
                if let Some(prev) = pending_user.take() {
                    out.push(Trace::UserMessage { text: prev });
                }
                out.push(Trace::UserMessage { text: content });
            }
        }
    }
    if let Some(prev) = pending_user.take() {
        out.push(Trace::UserMessage { text: prev });
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
    let ccsniff_dev = std::path::Path::new("C:/dev/ccsniff/src/cli.js");
    let (cmd, args): (&str, Vec<&str>) = if ccsniff_dev.exists() {
        ("node", vec!["C:/dev/ccsniff/src/cli.js", "--list-projects"])
    } else if cfg!(windows) {
        ("npx.cmd", vec!["--yes", "ccsniff", "--list-projects"])
    } else {
        ("npx", vec!["--yes", "ccsniff", "--list-projects"])
    };
    let out = Command::new(cmd)
        .args(&args)
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
