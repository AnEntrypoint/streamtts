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

## Full History Training

The trainer is designed to ingest your complete Claude Code history via ccsniff. **Status: 2026-05-02 full history exported.**

### Exported Data
- **File**: `ccsniff-full-history.ndjson` (31 MB, 28,174 traces)
- **Command**: `npx ccsniff --json --full > ccsniff-full-history.ndjson`
- **Content**: All 49,742 events from 184 files (user messages, assistant responses, tool use/results)

### Training Command

```bash
cargo run --release -p sttx-cli -- train \
  --ccsniff-from ccsniff-full-history.ndjson \
  --steps 100000 \
  --checkpoint-dir ckpt-full-history \
  --checkpoint-every 1000 \
  --model-repo RWKV/RWKV7-Goose-World3-1.5B-HF
```

**Rust version**: Requires rustc 1.86+ due to candle-core `is_multiple_of` unstable feature. Use `rustup default stable && rustup update` or `rustup default nightly`.

### What Happens

1. **Load**: Downloads RWKV-7 1.5B safetensors from HuggingFace (~3 GB, first run only)
2. **Process**: Streams 28,174 traces, extracting context and tokenizing via dynamic tokenizer
3. **Train**: State-tuning on recurrent layers + hypernetwork adaptation on trainable embeddings
4. **Replay**: Surprise-prioritized buffer (1000 capacity) samples 3 sequences per step at 20% weight
5. **Checkpoint**: Saves every 1000 steps (safetensors + metadata) to `ckpt-full-history/`

### Observability

- Logs: `.gm/log/<YYYY-MM-DD>/{cli,ccsniff,train}.jsonl`
- Inspect checkpoint: `cargo run --release -p sttx-cli -- inspect --checkpoint ckpt-full-history/checkpoint-final`

## Plugkit long-process limitation

The plugkit runner (exec:bash, exec:nodejs, PowerShell tool) cannot sustain processes that run longer than ~2 minutes. Background tasks fail silently with empty output files; foreground PowerShell also fails silently for long-running tasks. This affects `cargo build --release` (~3 min) and HF model downloads (~10 min+). For any such task, instruct the user to run the command manually in their terminal — do not attempt to run it via any plugkit exec method.

## Training dtype caveat

`forward_loss` must **not** call `.to_dtype(...)` on `target_t`. Doing so causes a dtype mismatch at the loss computation step that surfaces only at runtime on real traces, not on synthetic tensors. Removing that cast was the root-cause fix that unblocked training (verified 2026-05-03: 100 steps clean, no dtype errors).

## ccsniff export size is session-scoped

`npx ccsniff --json --full` only exports traces from the current Claude Code session history available at the time of export. The 2026-05-02 export (28k traces / 31 MB) is not reproducible; the 2026-05-03 re-export yielded 2485 traces / 3.1 MB. Do not treat any single export as a stable corpus — re-export before each training run if freshness matters.

## Flat loss root cause and fix (2026-05-03)

The base RWKV-7 model is loaded from mmaped safetensors as frozen constants — candle has no `Var`s for it. The `state_prefix` tensors are injected into `state.per_layer[i].att_kv`, but RWKV's recurrent state is mutated in-place during `forward_seq`, so gradients do not flow back into the initial state in a way that affects the CE loss. Result: 2485 steps with loss stuck at ~12.454 — zero learning.

**Fix**: Added a trainable per-vocab affine adapter (`logit_adapter.scale` init=1, `logit_adapter.bias` init=0) applied to the frozen logits before CE loss. These are registered in the `VarMap` (checkpoint save/load is automatic) and sit directly in the gradient path. LR raised to `3e-4`.

## HF vs native RWKV key format (2026-05-03)

`RWKV/RWKV7-Goose-World3-1.5B-HF` uses HuggingFace-style tensor names (`model.layers.N.attn.*`) but candle-transformers `rwkv_v7::Model` expects native RWKV names (`blocks.N.att.*`). `sttx-train/src/model.rs::maybe_remap_hf_weights()` detects this on load, remaps all keys, transposes LoRA weight matrices (lora.0 `[lora_dim, H]` → `[H, lora_dim]`, lora.2 `[H, lora_dim]` → `[lora_dim, H]`), and reshapes 1D `[H]` vectors to `[1,1,H]` for `k_k`, `k_a`, `ffn.x_k`, and LoRA bias terms. The result is cached as `model.safetensors.rwkv-native.safetensors` in the HF cache — rebuilt only if absent.

## ccsniff live mode uses local dev binary

`CcsniffStream::live()` checks for `C:/dev/ccsniff/src/cli.js` first and runs `node C:/dev/ccsniff/src/cli.js` if present. Falls back to `npx ccsniff` if not found.

## Learning audit

- **Audit cycle 2026-05-02**:
  - Action: Full ccsniff history (28k traces, 31 MB) exported and documented. Training infrastructure verified end-to-end in sttx-cli/sttx-ccsniff/sttx-train. Candle-core version pinned to 0.9; Rust 1.86+ required for `is_multiple_of` unstable feature — pre-built binary may already satisfy. Next step: resolve Rust version on user's system, run training.
