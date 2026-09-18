# HivLLM — All your models. One sticky hive. 🐝

Auto-discovers OpenAI-compatible endpoints running on this machine
and gathers them behind a **single endpoint**.

## How it works

1. **Discovery** (`src/discovery.rs`)
   - Probes well-known localhost ports (`11434` Ollama, `1234` LM Studio,
     `8000` vLLM, `8080` llama.cpp, …) via `GET /v1/models`.
   - Scans `ss -tln` listening ports + `docker ps` published ports
     and probes each for `/v1/models`.
   - Re-scans every `--scan-interval` seconds (default 30).

2. **Unified hive** (`src/hive.rs`)
   - `GET /v1/models` → aggregated model list from all members.
   - `POST /v1/chat/completions`, `/v1/completions`, `/v1/embeddings` →
     routed by `model`
     (JSON body field; `?model=` query param wins as override).
   - Same model name on several endpoints → **round-robin load-balancing**.
   - Streaming (`"stream": true`) SSE is passed through.

3. **Ops**
   - `GET /health`
   - `GET /api/hive/endpoints` → discovered members + their models.

## Run

```bash
cargo run --  # serves on :8335 (BEES 🐝) by default
# with extra ports + faster rescan:
cargo run -- --extra-ports 9000,8081 --scan-interval 10
```

Then:

```bash
curl localhost:8335/v1/models
curl localhost:8335/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"llama3.1","messages":[{"role":"user","content":"hi"}]}'
# query-param override also works:
curl 'localhost:8335/v1/chat/completions?model=llama3.1' \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hi"}]}'
```

## Layout

- `src/main.rs` — CLI + Axum server
- `src/discovery.rs` — port / process / docker scan + `/v1/models` probe
- `src/hive.rs` — aggregation, routing, load-balancing, proxy
