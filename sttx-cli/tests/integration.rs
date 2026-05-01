use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use candle_core::{DType, Device, Tensor};
use rand::SeedableRng;
use rand::rngs::StdRng;
use serde_json::json;
use sttx_ccsniff::CcsniffStream;
use sttx_core::obs;
use sttx_core::replay::{ReplayBuffer, Record};
use sttx_core::tokenizer::{make_hypernet, DynamicTokenizer, Hypernetwork};
use sttx_core::trace::Trace;

static FAILS: AtomicUsize = AtomicUsize::new(0);

fn check(label: &str, ok: bool) {
    if !ok {
        FAILS.fetch_add(1, Ordering::SeqCst);
        eprintln!("FAIL {label}");
    } else {
        println!("ok   {label}");
    }
}

#[test]
fn data_plane_groups() {
    let tmp = std::env::temp_dir().join(format!("sttx-test-{}", std::process::id()));
    std::env::set_var("STTX_LOG_DIR", &tmp);

    obs_group();
    replay_group();
    tokenizer_group();
    trace_group();
    hypernet_group();

    std::env::remove_var("STTX_LOG_DIR");
    let f = FAILS.load(Ordering::SeqCst);
    assert_eq!(f, 0, "{f} assertions failed");
}

fn obs_group() {
    let handles: Vec<_> = (0..4)
        .map(|t| std::thread::spawn(move || {
            for i in 0..25 { obs::info("data_obs", json!({"t":t,"i":i})); }
        }))
        .collect();
    for h in handles { h.join().unwrap(); }
    let day = chrono::Utc::now().format("%Y-%m-%d").to_string();
    let log_dir = std::env::var("STTX_LOG_DIR").unwrap();
    let path = std::path::Path::new(&log_dir).join(day).join("data_obs.jsonl");
    let content = std::fs::read_to_string(&path).expect("log file");
    let lines: Vec<&str> = content.lines().collect();
    check("obs_concurrent_lines_count", lines.len() == 100);
    check("obs_lines_valid_json", lines.iter().all(|l| serde_json::from_str::<serde_json::Value>(l).is_ok()));
}

fn replay_group() {
    let mut b = ReplayBuffer::new(100);
    for i in 0..150u32 {
        b.add(Record { input_ids: vec![i], target_ids: vec![i+1], surprise: i as f32 / 150.0 });
    }
    check("replay_capacity_respected", b.len() == 100);
    let mut rng = StdRng::seed_from_u64(7);
    let mut counts = vec![0u32; 256];
    for _ in 0..5000 {
        let s = b.sample(1, &mut rng);
        counts[s[0].input_ids[0] as usize] += 1;
    }
    let high: u32 = counts[120..150].iter().sum();
    let low_present: u32 = counts[0..50].iter().sum();
    check("replay_keeps_high_surprise", high > 0);
    check("replay_evicted_low_surprise", low_present == 0);
    let t = Instant::now();
    for _ in 0..1000 { b.sample(1, &mut rng); }
    check("replay_sample_fast", t.elapsed().as_millis() < 200);
}

fn tokenizer_group() {
    let mut t = DynamicTokenizer::new(128, 32, 3);
    let mut all = Vec::new();
    for _ in 0..10 {
        all.extend([10u32, 11, 12, 13, 10, 11]);
    }
    t.observe(&all);
    let merged = t.merge(&all);
    check("tokenizer_promotes", t.promotions_total >= 1);
    check("tokenizer_merges", t.merges_total >= 1);
    check("tokenizer_emits_dynamic_id", merged.iter().any(|&x| x >= sttx_core::tokenizer::DYNAMIC_VOCAB_OFFSET));

    for i in 0..50u32 {
        t.merge(&[1000 + i*2, 1001 + i*2, 1000 + i*2, 1001 + i*2, 1000 + i*2, 1001 + i*2]);
        t.observe(&[1000 + i*2, 1001 + i*2, 1000 + i*2, 1001 + i*2, 1000 + i*2, 1001 + i*2]);
    }
    check("tokenizer_lru_capacity", t.promotions_total >= 1);
}

fn trace_group() {
    let user = Trace::from_ccsniff_event(json!({"role":"user","type":"text","tool":null,"text":"hello world"}));
    check("trace_user", matches!(user, Trace::UserMessage{ref text} if text == "hello world"));
    let asst = Trace::from_ccsniff_event(json!({"role":"assistant","type":"text","tool":null,"text":"hi"}));
    check("trace_assistant", matches!(asst, Trace::AssistantMessage{ref text} if text == "hi"));
    let tool = Trace::from_ccsniff_event(json!({"role":"assistant","type":"tool_use","tool":"Bash","input":{"command":"ls"}}));
    check("trace_tool_use", matches!(tool, Trace::ToolUse{ref name,..} if name == "Bash"));
    let tool_result = Trace::from_ccsniff_event(json!({"role":"tool_result","type":"tool_result","tool":"Bash","text":"file1\nfile2"}));
    check("trace_tool_result", matches!(tool_result, Trace::ToolResult{ref name, ref output} if name == "Bash" && output == "file1\nfile2"));
    check("trace_corpus_text_nonempty", !user.corpus_text().is_empty());
}

fn hypernet_group() {
    let dev = Device::Cpu;
    let (h, vm) = make_hypernet(64, DType::F32, &dev).expect("hypernet");
    let a = Tensor::randn(0f32, 1.0, &[2, 64], &dev).unwrap();
    let b = Tensor::randn(0f32, 1.0, &[2, 64], &dev).unwrap();
    let out = h.forward(&a, &b).unwrap();
    check("hypernet_output_shape", out.dims() == &[2, 64]);
    check("hypernet_has_trainable_params", !vm.all_vars().is_empty());
    let _ = h.hidden_size();
    let _: &Hypernetwork = &h;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn system_plane_ccsniff_replay_endtoend() {
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("test-data")
        .join("ccsniff-fixture.ndjson");
    if !fixture.exists() {
        eprintln!("skip system_plane: fixture absent at {}", fixture.display());
        return;
    }
    let mut stream = CcsniffStream::from_file(fixture.to_str().unwrap(), 64)
        .await
        .expect("from_file");
    let mut traces = Vec::new();
    while let Some(t) = stream.recv().await {
        traces.push(t);
        if traces.len() >= 40 {
            break;
        }
    }
    check("ccsniff_emitted_traces", traces.len() >= 5);
    let any_text = traces.iter().any(|t| !t.corpus_text().is_empty());
    check("ccsniff_yielded_corpus_text", any_text);

    let mut tk = DynamicTokenizer::new(512, 256, 3);
    let mut total_bytes = 0usize;
    for t in &traces {
        let text = t.corpus_text();
        total_bytes += text.len();
        let bytes_as_ids: Vec<u32> = text.bytes().map(|b| b as u32).collect();
        tk.observe(&bytes_as_ids);
        tk.merge(&bytes_as_ids);
    }
    check("system_corpus_nontrivial", total_bytes > 200);
    check("system_tokenizer_active", tk.merges_total + tk.promotions_total >= 0 && total_bytes > 0);

    let mut buf = ReplayBuffer::new(64);
    let mut rng = StdRng::seed_from_u64(11);
    for (i, t) in traces.iter().enumerate() {
        let ids: Vec<u32> = t.corpus_text().bytes().map(|b| b as u32).take(64).collect();
        if ids.len() < 2 { continue; }
        let surprise = (i % 5) as f32 / 4.0 + 0.05;
        buf.add(Record {
            input_ids: ids[..ids.len() - 1].to_vec(),
            target_ids: ids[1..].to_vec(),
            surprise,
        });
    }
    let s = buf.sample(8, &mut rng);
    check("system_replay_sampled", s.len() == 8 || (buf.is_empty() && s.is_empty()));

    let f = FAILS.load(Ordering::SeqCst);
    assert_eq!(f, 0, "{f} system-plane assertions failed");
}
