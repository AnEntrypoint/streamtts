# AGENTS.md — streamtts architecture reference

## Purpose

streamtts is a RWKV-7 streaming trainer that ingests Claude Code traces (via [ccsniff](https://github.com/AnEntrypoint/ccsniff)) and adapts a 1.5B-parameter model in real time using state-tuning + a hypernetwork over an online dynamic tokenizer. Single Rust binary, distributable via `cargo build --release`.

## Crate layout

| Crate | Responsibility |
|---|---|
| `sttx-core` | Structured JSONL observability, online bigram-merge dynamic tokenizer with hypernetwork, surprise-prioritized replay buffer, shared trace types |
| `sttx-ccsniff` | Tokio subprocess adapter spawning `npx ccsniff`, line-buffered JSONL parser, typed `Trace` event enum, bounded mpsc backpressure |
| `sttx-train` | HF model load via `hf-hub`, candle `rwkv_v7::Model` instantiation, state-tuning training loop, checkpoint serialization |
| `sttx-cli` | `streamtts` binary entry point, clap-derive subcommand dispatch |

## Key dependency facts

- `candle-transformers >= 0.9` exposes `models::rwkv_v7::{Config, Model, State, ModelVersion}`. `Model::forward_seq(token_ids, &mut state) -> Result<Tensor>` is the streaming primitive. `State` is **not Clone** — must be carried by mutable reference across forward calls.
- HF repo `RWKV/RWKV7-Goose-World3-1.5B-HF` is the canonical 1.5B world model with safetensors + tokenizer.json.
- ccsniff is `npm i -g ccsniff`; CLI emits JSONL on stdout.

## Memory budget (target: 6 GB)

- RWKV-7 1.5B BF16 weights: ~3 GB
- Activations (state-tuning, batch 1, ctx ≤ 1024): ~0.5 GB
- Per-layer recurrent state (trainable): ~10 MB
- Hypernetwork weights (trainable): ~10 MB
- Replay buffer (1000 × 1024 tokens × 4 bytes): ~4 MB
- AdamW moments on trainable subset: ~40 MB
- Headroom: rest

## Observability

`.gm/log/<YYYY-MM-DD>/<subsystem>.jsonl` mirrors plugkit shape. One JSON line per state transition, external IO with timing, or non-trivial decision. No `println!` scatter outside `obs::emit`.

## Single integration test

`tests/integration.rs` (workspace root) — under 200 lines, real subprocess, real model download, real training step, real checkpoint round-trip. Run via `cargo test --release`.

## candle-transformers Model Testing & RWKV-7 Tensor Layout

### Test Witness Pattern for Large Models

When production code requires loading multi-GB pretrained safetensors but the test budget cannot afford the download, create a synthetic tiny safetensors file (e.g., RWKV-7 with vocab=64, hidden=32, layers=2, heads=16) to exercise the entire forward + autograd graph against random-fill F32 tensors. The witness covers every code path the production loop executes — only byte values differ.

**Critical constraint**: candle's `VarBuilder::zeros(...)` and `from_varmap(...)` do **not** support `get_unchecked` (used internally by some models including `rwkv_v7`'s `infer_lora_dims`). Only `from_mmaped_safetensors`, `from_pth`, or `from_buffered_safetensors` provide unchecked access. Therefore, the witness **must be a real on-disk safetensors file**, not an in-memory VarMap.

**Reference implementation**: `sttx-train/src/model.rs::write_random_safetensors` and `sttx-cli/tests/train_witness.rs`.

### RWKV-7 Safetensors Tensor Layout (candle-transformers 0.10)

Expected shapes for exact model reconstruction:

**Top level**:
- `emb.weight` [V, H]
- `ln_out.weight`, `ln_out.bias` [H]
- `head.weight` [V, H]

**Per-layer `blocks.{i}` (attention + FFN block)**:
- `ln1.weight`, `ln1.bias` [H]
- `ln2.weight`, `ln2.bias` [H]
- Layer 0 **only**: `ln0.weight`, `ln0.bias` [H]

**Attention sublayer `blocks.{i}.att`**:
- Token-shift: `x_r`, `x_w`, `x_k`, `x_v`, `x_a`, `x_g` all [1, 1, H]
- Decay LoRA: `w0` [1,1,H], `w1` [H, lora_dim], `w2` [lora_dim, H]
- ICL rate LoRA: `a0` [1,1,H], `a1` [H, lora_dim], `a2` [lora_dim, H]
- Value residual LoRA: `v0` [1,1,H], `v1` [H, lora_dim], `v2` [lora_dim, H] — **present on layer 0 as well**, despite some documentation claiming layer > 0 only
- Gate LoRA: `g1` [H, lora_dim], `g2` [lora_dim, H]
- Key processing: `k_k`, `k_a` [1,1,H], `r_k` [n_heads, head_size]
- Linear: `receptance.weight`, `key.weight`, `value.weight`, `output.weight` all [H, H]
- Groupnorm: `ln_x.weight`, `ln_x.bias` [H]

**FFN sublayer `blocks.{i}.ffn`**:
- `x_k` [1,1,H]
- `key.weight` [intermediate, H]
- `value.weight` [H, intermediate]

**LoRA dimension inference**: The inner LoRA dimension is auto-inferred from `w1`'s second dimension. For tiny test configs, use `(H/4).max(8)` to balance shape coverage against memory.

## Learning audit

- **Audit cycle 2026-05-01**:
  - Queried: 5 stable items (candle-transformers API, sttx crate layout, memory budget, observability, integration test)
  - Recall status: rs-learn store returning no results (service initialization pending)
  - Action: Both test witness + tensor layout facts newly ingested to rs-learn (reference/candle-transformers-test-witness, reference/rwkv-v7-tensor-layout) and appended to AGENTS.md as non-obvious caveats. Next audit cycle will test migration readiness once recall service is operational.
