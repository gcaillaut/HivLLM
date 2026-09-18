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

2. **Unified hive** (`src/hive.rs`, `src/load.rs`)
   - `GET /v1/models` → aggregated model list from all members.
   - `POST /v1/chat/completions`, `/v1/completions`, `/v1/embeddings` →
     routed by `model`
     (JSON body field; `?model=` query param wins as override).
   - **Load-aware routing**: backends are polled (`--load-interval`, default
     5s) via provider probes — vLLM `GET /load` today (needs
     `--enable-server-load-tracking` server-side; without it the backend
     simply reports unknown) — and requests go to the lowest `server_load`
     (tracked requests, not GPU util). Ties and unknown-load backends
     round-robin; no load info at all degrades to plain round-robin.
   - Unreachable backends fail over to the next candidate instead of
     failing the request.
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
- `src/logging.rs` — query log entries + `LogSink` trait

## Query logging

Every proxied query (chat / completions / embeddings — successes,
streaming requests, and routing failures) is appended to
`hivllm-queries.jsonl` (JSON lines) unless `--log-file ""` is passed:

```bash
cargo run -- --log-file queries.jsonl --log-format jsonl
```

Inspect with `jq`:

```bash
jq -c '{ts, route, model, upstream, status, latency_ms}' hivllm-queries.jsonl
jq -s 'group_by(.model) | map({model: .[0].model, n: length})' hivllm-queries.jsonl
# read back full responses (content + reasoning kept separate):
jq -c '{model, response: {content, reasoning, tool_calls}}' hivllm-queries.jsonl
```

Each entry carries the full client `request` and the full upstream
`response` by default: `response.content` and `response.reasoning` in
separate fields, merged `response.tool_calls`, plus the raw payload
(`response.raw`, absent for streams — those store the assembled text and
a `chunks` count instead). Tool *results* (role `"tool"`) travel in later
client requests, already logged in full under `request`.

Cap log size with a truncation strategy:

```bash
cargo run -- --log-truncate chars --log-max-chars 500
```

Adding a sink (YAML file, Langfuse, Logfire, …) means implementing the
`LogSink` trait in `src/logging.rs` and wiring it in `main.rs` —
the proxy code doesn't change.
