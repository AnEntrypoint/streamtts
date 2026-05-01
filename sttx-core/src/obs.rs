use std::fs::{create_dir_all, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;

use chrono::Utc;
use serde_json::{json, Value};

static LOCK: Mutex<()> = Mutex::new(());

fn log_root() -> PathBuf {
    if let Ok(p) = std::env::var("STTX_LOG_DIR") {
        return PathBuf::from(p);
    }
    PathBuf::from(".gm").join("log")
}

pub fn emit(subsystem: &str, severity: &str, fields: Value) {
    let now = Utc::now();
    let day = now.format("%Y-%m-%d").to_string();
    let dir = log_root().join(&day);
    let path = dir.join(format!("{subsystem}.jsonl"));

    let line = json!({
        "ts": now.to_rfc3339(),
        "subsystem": subsystem,
        "severity": severity,
        "fields": fields,
    })
    .to_string();

    let _g = LOCK.lock().unwrap();
    if let Err(e) = create_dir_all(&dir) {
        eprintln!("obs: mkdir {} failed: {e}", dir.display());
        return;
    }
    match OpenOptions::new().create(true).append(true).open(&path) {
        Ok(mut f) => {
            let _ = writeln!(f, "{line}");
        }
        Err(e) => eprintln!("obs: open {} failed: {e}", path.display()),
    }
}

pub fn info(subsystem: &str, fields: Value) {
    emit(subsystem, "info", fields)
}
pub fn warn(subsystem: &str, fields: Value) {
    emit(subsystem, "warn", fields)
}
pub fn error(subsystem: &str, fields: Value) {
    emit(subsystem, "error", fields)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn concurrent_emits_yield_valid_jsonl() {
        let tmp = tempdir_path();
        std::env::set_var("STTX_LOG_DIR", &tmp);

        let handles: Vec<_> = (0..4)
            .map(|t| {
                let tag = Arc::new(format!("test-thread-{t}"));
                thread::spawn(move || {
                    for i in 0..25 {
                        info(
                            "concurrency",
                            json!({"thread": *tag, "i": i}),
                        );
                    }
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }

        let day = Utc::now().format("%Y-%m-%d").to_string();
        let path = std::path::Path::new(&tmp)
            .join(day)
            .join("concurrency.jsonl");
        let content = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        assert_eq!(lines.len(), 100, "expected 100 lines, got {}", lines.len());
        for l in &lines {
            let _: Value = serde_json::from_str(l).expect("valid json");
        }
        std::env::remove_var("STTX_LOG_DIR");
    }

    fn tempdir_path() -> String {
        let mut p = std::env::temp_dir();
        p.push(format!("sttx-obs-test-{}", std::process::id()));
        p.to_string_lossy().to_string()
    }
}
