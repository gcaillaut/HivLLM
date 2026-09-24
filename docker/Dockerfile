# HivLLM — All your models. One sticky hive. 🐝
#
# Build from the repo root:
#   docker build -f docker/Dockerfile -t hivllm .
# Run (host networking so the hive can probe localhost backends):
#   docker run --rm --network host hivllm
#   docker run --rm --network host -v hivllm-logs:/data hivllm --log-file /data/hivllm-queries.jsonl

# ---------- builder ----------
FROM rust:1-bookworm AS builder
WORKDIR /build
COPY Cargo.toml Cargo.lock ./
COPY src ./src
RUN cargo build --locked --release

# ---------- runtime ----------
FROM debian:bookworm-slim
# libssl3: reqwest links OpenSSL dynamically — the binary won't start without it.
# ca-certificates: upstream TLS. curl: HEALTHCHECK.
# iproute2: `ss -tln` powers discovery of host listening ports.
# (No docker client: `docker ps` scanning stays host-side; well-known ports
# + ss listeners cover containerized discovery.)
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl iproute2 libssl3 \
    && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --shell /usr/sbin/nologin hivllm
WORKDIR /app
COPY --from=builder /build/target/release/hivllm /app/hivllm
# The unprivileged user must be able to write the default query log
# (./hivllm-queries.jsonl) into WORKDIR.
RUN chown -R hivllm:hivllm /app
USER hivllm
EXPOSE 8335
# Serve on another port with `-e HIVLLM_PORT=…` (same as --port): the
# healthcheck follows it. /health needs no API key.
ENV HIVLLM_PORT=8335
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${HIVLLM_PORT}/health" > /dev/null
ENTRYPOINT ["/app/hivllm"]
# Sane defaults (same as the binary): :8335, 30s rescan, 5s load poll,
# jsonl query log in WORKDIR. Override by appending flags, e.g.:
#   docker run --rm --network host -e HIVLLM_PORT=8344 hivllm --scan-interval 10
CMD []
