# streamtts

RWKV-7 streaming trainer and inference server. Trains on Claude Code traces via [ccsniff](https://www.npmjs.com/package/ccsniff), then serves the fine-tuned model as an OpenAI-compatible HTTP API.

## Architecture

- **Model**: RWKV-7 "Goose" 1.5B (`RWKV/RWKV7-Goose-World3-1.5B-HF`) via [`candle-transformers`](https://crates.io/crates/candle-transformers).
- **Sub-quadratic**: O(L) inference, fixed-size hidden state — infinite streams at constant memory.
- **State-tuning**: gradients route only through per-layer recurrent state + a hypernetwork for dynamic-token embeddings. Pretrained weights frozen. ~6 GB VRAM target on CPU/CUDA.
- **Online dynamic tokenization**: bigram-merge LZW-inspired. Sliding window of 512 bigrams; bigrams seen ≥3 times become virtual tokens; embeddings produced by a 2-layer hypernetwork over the constituent base-token embeddings; LRU cache of capacity 256.
- **Surprise-prioritized replay**: ring buffer (max 1000) sampled with probability ∝ surprise.
- **Data source**: [`ccsniff`](https://github.com/AnEntrypoint/ccsniff) streams Claude Code JSONL events into typed `Trace` enum via tokio subprocess + bounded mpsc.

## Layout

```
sttx-core/      observability, dynamic tokenizer, replay buffer, trace types
sttx-ccsniff/   ccsniff subprocess adapter
sttx-train/     model load, state-tuning training loop, checkpoints, inference server
sttx-cli/       streamtts.exe binary, clap subcommands
```

## Build

```bash
cargo build --release
./target/release/streamtts --help
```

## Train

```bash
# On live Claude Code session
./target/release/streamtts train --ccsniff-from live --steps 1000 --checkpoint-dir ./ckpt

# On exported history (API-pair mode — structures user/assistant turns as prompt+completion)
./target/release/streamtts train \
  --ccsniff-from ccsniff-full-history.ndjson \
  --ccsniff-from C:/dev/cc.jsonl \
  --steps 100000 \
  --checkpoint-dir ckpt-full-pairs \
  --checkpoint-every 1000 \
  --api-pairs
```

## Serve

```bash
# Start inference server on port 8080
./target/release/streamtts serve --checkpoint ./ckpt-full-pairs --port 8080

# Query it (OpenAI-compatible)
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"How do I list files in bash?"}]}'

# Health check
curl http://localhost:8080/health
```

## Inspect

```bash
./target/release/streamtts inspect --checkpoint ./ckpt-full-pairs
./target/release/streamtts merge-stats --checkpoint ./ckpt-full-pairs
```

## Tests

```bash
cargo test --workspace --release
```

## License

MIT
