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
   - `ss` / `docker ps` run asynchronously with a 5s cap: a missing or
     wedged Docker daemon never stalls discovery.

2. **Unified hive** (`src/hive.rs`, `src/load.rs`)
   - `GET /v1/models` → aggregated model list from all members.
   - `POST /v1/chat/completions`, `/v1/completions`, `/v1/embeddings` →
     routed by `model`
     (JSON body field; `?model=` query param wins as override).
    - **Load-aware routing**: backends are polled (`--load-interval`, default
      5s) via provider probes — vLLM `GET /metrics`
      (`num_requests_running` + `num_requests_waiting`, always exported)
      first, `GET /load` as fallback (only truthful with
      `--enable-server-load-tracking` server-side) — and requests go to
      the lowest *effective* load: `max(server_load, hive-observed
      in-flight requests)` (requests count as in flight from the moment
      they are sent). Ties round-robin. Unreachable backends, and backends
      answering `429`/`500`/`502`/`503` (overloaded, broken, still loading,
      a downstream hive's `loop detected`), fail over to the next candidate
      — streams included, since the status arrives before any byte is
      sent. Unreachable, `429` and `503` backends are also ranked last for
      15s, so they stop looking idle. The last candidate's error is passed
      through as-is. Other `4xx` are never retried (every backend would
      repeat them), nor timeouts / `504` (the generation may still be
      running).
    - Stdout shows the same numbers the balancer uses, per model:
      positive server reports as `load=N`, anything unverified as
      `load=~N` (unknown backends, or a `0` that could equally mean idle
      or untracked). Backends ordered by decreasing load.
   - `GET /load[?model=]` reports the hive's own aggregate pressure
     (median of member effective loads, exact reports only) with marker
     `"hivllm": {"exact": bool}` so upstream hives route on real numbers —
     per model when `?model=` is given, global otherwise. Unverified
     aggregates are marked inexact and never trusted downstream, so a
     stale number can't circulate hive-to-hive as fact (approximations
     are always recomputed locally, never forwarded as exact). Every hive stamps its
     responses with a unique instance id (`x-hivllm-id`, random per
     process, pin it with `--hive-id`), which discovery records for members
     that are hives. Forwarded requests carry the ids of the hives they
     went through (`x-hivllm-via`) and are never sent back to a visited
     hive, whatever address it is known by — mutual preferences can't
     ping-pong forever, across hosts too (answer is 502 `loop detected`
     instead). A hive that discovers itself under another address (static
     URL, LAN IP, container name) skips that endpoint.
   - **Hive-of-hives model routes (path vector):** in `/v1/models`, each
     model carries `"hivllm": {"paths": [[hive ids…]]}` — the chains of
     hives through which it reaches a real backend (`[]` = served
     directly). A hive probing a peer sends its id as `x-hivllm-via`, and
     the peer leaves out routes through it; routes through the hive itself
     are always dropped, and a model with no route left disappears. So
     hives that discover each other (in pairs or rings) can't keep a dead
     backend's model alive by re-advertising it: the hive that lost the
     backend drops it on its next scan, and stale routes elsewhere expire
     one hop per scan. Requests are never forwarded to a member whose only
     routes loop back. Paths are capped at 8 hops, 8 routes per model.
   - Streaming (`"stream": true`) SSE is passed through.
   - Upstream response headers (`content-type`, request ids, rate-limit
     headers, …) reach the client, minus hop-by-hop headers and the
     upstream's CORS headers (the hive's own policy applies).
   - Request bodies up to `--max-body-mb` (default 64 MiB, 0 = unlimited)
     are accepted: long contexts and base64 images fit.

3. **Ops**
   - `GET /health`
   - `GET /api/hive/endpoints` → discovered members + their models.
   - `GET /api/hive/backends` → per-model backends with the loads the
     balancer routes on (`{models: [{id, backends: [{ip, port, load, exact}]}]}`).
   - `GET /api/hive/queries?limit=100` → last query-log entries, newest first.

## Security

- `--api-key` (or `HIVLLM_API_KEY`, preferred: flags show in `ps`) requires
  `Authorization: Bearer <key>` on every route except `/health`. Set it
  whenever the hive is reachable beyond localhost — the query log holds
  every prompt and response.
- The client's `Authorization` header is **not** forwarded to backends
  (it would reach every candidate). `--forward-auth` restores passthrough;
  with `--api-key`, that forwards the hive key itself.
- CORS (`--cors-origin`) defaults to `local`: only pages served from
  `localhost` / `127.0.0.1` / `[::1]` (any port) may call the hive from a
  browser. Pass a comma-separated origin list for a UI hosted elsewhere,
  `*` for any site (then any web page you visit can read the query log),
  or `""` to disable.

## Timeouts

- `--connect-timeout` (default 5s): a dead host fails over quickly.
- `--read-timeout` (default 600s, 0 = none): longest silence tolerated
  from a backend. There is no total timeout, so a stream is never cut
  while tokens keep flowing — but non-streaming backends stay silent
  until the whole generation is done, so this caps those.

## Run

```bash
cargo run --  # serves on :8335 (BEES 🐝) by default
# with extra ports + faster rescan:
cargo run -- --extra-ports 9000,8081 --scan-interval 10
# pin backends discovery can't see (docker service names, remote hosts):
cargo run -- --static-backends http://llamacpp:8080,http://gpu-box:8000
# expose beyond localhost (containers, LAN):
cargo run -- --bind 0.0.0.0
```

## Docker (`docker/`)

```bash
docker build -f docker/Dockerfile -t hivllm .
docker run --rm --network host hivllm
# persist the query log + custom flags:
docker run --rm --network host -v hivllm-logs:/data hivllm \
  --log-file /data/hivllm-queries.jsonl --port 8335
# or via compose:
docker compose -f docker/docker-compose.yml up --build
```

Notes:
- `--network host` (Linux) is the zero-config path: discovery probes localhost,
  which sees host backends in the host's network namespace.
- For bridge networks (siblings like `http://llamacpp:8080`), combine
  `--bind 0.0.0.0` (the hive binds `127.0.0.1` by default, unreachable through
  published ports otherwise) with `--static-backends http://llamacpp:8080` —
  static entries are probed for `/v1/models` on every rescan, flow through
  routing, load probes and logging like discovered ones, and rejoin
  automatically after flapping. Unreachable entries are skipped with a warning.
- macOS/Windows have no host net — run the binary directly there instead.
- Containerized discovery covers well-known ports + `ss` listeners (`docker ps`
  scanning stays host-side). Process names aren't visible across the container
  boundary, so members show as `port-XXXX` instead.
- Runs as unprivileged user: stick to ports > 1024. The image carries a
  `HEALTHCHECK` on `:8335/health` (override it when serving another `--port`).
- Image is ~110MB (multi-stage `rust:1-bookworm` → `debian:bookworm-slim`).

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

The log rolls by size: once `hivllm-queries.jsonl` would exceed
`--log-max-mb` (default 100 MiB, 0 = never), it is renamed to
`hivllm-queries.<UTC timestamp>.jsonl`, gzipped in the background
(`--log-compress false` to keep plain files) and only the newest
`--log-keep` archives stay (default 20, 0 = all). Entries are never cut by
rolling, and `/api/hive/queries` reads through archives when the live file
is short. Read archives back with `zcat`:

```bash
zcat hivllm-queries.*.jsonl.gz | jq -c '{ts, model, status}'
cargo run -- --log-max-mb 500 --log-keep 50
```

Cap entry size with a truncation strategy:

```bash
cargo run -- --log-truncate chars --log-max-chars 500
```

Adding a sink (YAML file, Langfuse, Logfire, …) means implementing the
`LogSink` trait in `src/logging.rs` and wiring it in `main.rs` —
the proxy code doesn't change.
