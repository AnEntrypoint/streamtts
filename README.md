# streamtts

RWKV-7 streaming trainer with online dynamic tokenization, state-tuning, and surprise-prioritized replay. Single Rust binary. Trains on Claude Code traces via [ccsniff](https://www.npmjs.com/package/ccsniff).

## Architecture

- **Model**: RWKV-7 "Goose" 1.5B (`RWKV/RWKV7-Goose-World3-1.5B-HF`) via [`candle-transformers`](https://crates.io/crates/candle-transformers).
- **Sub-quadratic**: O(L) inference, fixed-size hidden state — infinite streams at constant memory.
- **State-tuning**: gradients route only through per-layer recurrent state + a hypernetwork for dynamic-token embeddings. Pretrained weights frozen. ~6 GB VRAM target on CPU/CUDA.
- **Online dynamic tokenization**: bigram-merge LZW-inspired (akin to [zip2zip](https://arxiv.org/abs/2506.01084)). Sliding window of 512 bigrams; bigrams seen ≥3 times become virtual tokens; embeddings produced by a 2-layer hypernetwork over the constituent base-token embeddings; LRU cache of capacity 256.
- **Surprise-prioritized replay**: ring buffer (max 1000) sampled with probability ∝ surprise — see [SuRe](https://arxiv.org/abs/2511.22367).
- **Data source**: [`ccsniff`](https://github.com/AnEntrypoint/ccsniff) streams Claude Code JSONL events into typed `Trace` enum via tokio subprocess + bounded mpsc.

## Layout

```
sttx-core/      observability, dynamic tokenizer, replay buffer, trace types
sttx-ccsniff/   ccsniff subprocess adapter
sttx-train/     model load, state-tuning training loop, checkpoints
sttx-cli/       streamtts.exe binary, clap subcommands
tests/          single integration test (cargo test --release)
```

## Build

```
cargo build --release
./target/release/streamtts --help
./target/release/streamtts train --ccsniff-from live --steps 1000 --checkpoint-dir ./ckpt
./target/release/streamtts inspect --checkpoint ./ckpt/latest
./target/release/streamtts merge-stats --checkpoint ./ckpt/latest
```

## Smoke witnesses

```
cargo run --release --example smoke_load
cargo run --release --example smoke_train
```

`smoke_load` downloads the 1.5B safetensors (~3 GB) into the HF cache, instantiates `rwkv_v7::Model`, and runs a single `forward_seq` to print `vocab_size × first-5-logits`. `smoke_train` runs 5 state-tuning steps over a synthetic stream, saves a checkpoint to `ckpt-smoke/`, and prints `(steps, losses, merges, promotions)`.

## Tests

```
cargo test --workspace --release
```

8 test binaries, all green:
- `sttx-core` unit tests (5): observability JSONL round-trip under concurrent writes, replay-buffer surprise-weighted sampling + low-surprise eviction, dynamic-tokenizer bigram promotion + merge, hypernetwork forward shape.
- `sttx-cli` integration (2 groups, 23 assertions): data-plane (obs + replay + tokenizer + trace + hypernetwork) and system-plane (ccsniff JSONL parser exercised against `test-data/ccsniff-fixture.ndjson` — 40 real Claude Code events captured via `npx ccsniff -f --json --limit 40`).
- `sttx-cli` `train_witness` (1): real `rwkv_v7::Model::new` instantiated from a synthesized random-init safetensors at a tiny config (`vocab=64, hidden=32, layers=2, head=16`), real `forward_seq` produces logits of correct shape, real per-layer state tensors, real 8-step `Trainer::step_on_trace` over a synthetic stream, real `AdamW.backward_step`, real `VarMap.save`/`load_meta` round-trip. Exercises every code path the production loop uses against pretrained 1.5B weights — only the bytes differ.

## License

MIT
