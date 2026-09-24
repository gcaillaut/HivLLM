//! HivLLM — All your models. One sticky hive. 🐝
//!
//! Auto-discovers OpenAI-compatible endpoints on this machine
//! (localhost port scan + process/docker scan) and serves them
//! behind a single endpoint:
//!
//! - `GET /v1/models` — aggregated model list
//! - `POST /v1/chat/completions`, `/v1/completions`, `/v1/embeddings` —
//!   routed by `model` to the least-loaded backend
//!   (JSON body field, `?model=` query param wins as override).
//!   Ties and backends without load info round-robin; unreachable
//!   backends fail over to the next candidate.

mod discovery;
mod docker;
mod hive;
mod load;
mod logging;

use axum::{
    routing::{get, post},
    Router,
};
use clap::{Parser, ValueEnum};
use std::{
    net::{IpAddr, SocketAddr},
    sync::Arc,
    time::Duration,
};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use hive::Hive;
use logging::{JsonLinesSink, RequestLogger};

/// Query-log output format. Only `jsonl` is implemented today —
/// `yaml` (and Langfuse/Logfire API sinks) plug into the same
/// `LogSink` trait (see `src/logging.rs`).
#[derive(Debug, Clone, Copy, ValueEnum)]
enum LogFormat {
    Jsonl,
}

#[derive(Parser, Debug)]
#[command(name = "hivllm", about = "All your models. One sticky hive. 🐝")]
struct Args {
    /// Port for the unified hive endpoint (8335 = BEES 🐝)
    #[arg(long, default_value_t = 8335)]
    port: u16,

    /// Interface to bind (127.0.0.1 keeps the hive local; 0.0.0.0 exposes
    /// it, e.g. from a container with published ports)
    #[arg(long, default_value = "127.0.0.1")]
    bind: IpAddr,

    /// Extra localhost ports to probe (in addition to well-known ones)
    #[arg(long, value_delimiter = ',')]
    extra_ports: Vec<u16>,

    /// Static backends discovery can't see: base URLs probed for
    /// `/v1/models` on every rescan (docker service names like
    /// `http://llamacpp:8080`, remote hosts, …). A base may already
    /// include the OpenAI `/v1` prefix when the backend lives under a
    /// path-routing gateway
    /// (`http://host:8180/general-stage1/v1` probes
    /// `.../general-stage1/v1/models` and routes
    /// `.../general-stage1/v1/chat/completions`).
    /// Comma-separated and/or repeatable. Unreachable entries are
    /// skipped until they answer.
    #[arg(long, value_delimiter = ',')]
    static_backends: Vec<String>,

    /// Docker Engine socket for container auto-discovery
    /// (`-v /var/run/docker.sock:/var/run/docker.sock` + opt-in labels
    /// `hivllm.enable=true`, `hivllm.port=8080`). Empty = disabled.
    #[arg(long, default_value = "")]
    docker_socket: String,

    /// Re-scan interval in seconds (0 = scan once at startup)
    #[arg(long, default_value_t = 30)]
    scan_interval: u64,

    /// Backend load poll interval in seconds for least-load routing
    /// (0 = disable polling, fall back to plain round-robin)
    #[arg(long, default_value_t = 5)]
    load_interval: u64,

    /// Query log file (one entry per line). Empty = no query logging.
    #[arg(long, default_value = "hivllm-queries.jsonl")]
    log_file: String,

    /// Query log format
    #[arg(long, value_enum, default_value_t = LogFormat::Jsonl)]
    log_format: LogFormat,

    /// Truncation strategy for long text in query logs (requests + responses)
    #[arg(long, value_enum, default_value_t = logging::Truncate::None)]
    log_truncate: logging::Truncate,

    /// Max chars per text field when --log-truncate chars
    #[arg(long, default_value_t = 2000)]
    log_max_chars: usize,

    /// CORS allowed origin for browser UIs ("*" = any; empty = disabled)
    #[arg(long, default_value = "*")]
    cors_origin: String,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "hivllm=info,tower_http=info".into()),
        ))
        .with(tracing_subscriber::fmt::layer())
        .init();

    let args = Args::parse();

    let logger = if args.log_file.is_empty() {
        RequestLogger::new()
    } else {
        let sink: Arc<dyn logging::LogSink> = match args.log_format {
            LogFormat::Jsonl => Arc::new(JsonLinesSink::open(&args.log_file).await?),
        };
        tracing::info!(file = %args.log_file, sink = sink.name(), "🐝 query log enabled");
        RequestLogger::new().with_sink(sink)
    };
    let hive = Hive::new(args.port)
        .with_logger(logger)
        .with_log_options(args.log_truncate, args.log_max_chars);

    // Docker socket discovery (opt-in): containers labeled
    // `hivllm.enable=true` join the hive; lifecycle events refresh it.
    let docker = if args.docker_socket.is_empty() {
        None
    } else {
        match docker::DockerDiscovery::connect(&args.docker_socket).await {
            Ok(d) => {
                tracing::info!(socket = %args.docker_socket, "🐝 docker discovery enabled");
                Some(d)
            }
            Err(e) => {
                tracing::warn!(error = %e, "docker socket unreachable (mount it with -v and match its group, e.g. --group-add $(stat -c %g /var/run/docker.sock)), continuing without it");
                None
            }
        }
    };

    // Initial discovery before serving.
    hive
        .refresh(&args.extra_ports, &args.static_backends, docker.as_ref())
        .await;

    // Prime load state so the first requests are already load-informed.
    // The load refresh announces the hive view; without polling, the
    // discovery refresh announces it instead — either way, exactly once.
    if args.load_interval > 0 {
        hive.refresh_load().await;
        let hive_bg = hive.clone();
        let interval = args.load_interval;
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(interval)).await;
                hive_bg.refresh_load().await;
            }
        });
    } else {
        hive.log_view().await;
    }

    // Background re-scan loop.
    if args.scan_interval > 0 {
        let hive_bg = hive.clone();
        let extra = args.extra_ports.clone();
        let pinned = args.static_backends.clone();
        let dock = docker.clone();
        let interval = args.scan_interval;
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(interval)).await;
                hive_bg.refresh(&extra, &pinned, dock.as_ref()).await;
                hive_bg.log_view().await;
            }
        });
    }

    // Docker lifecycle watcher: instant refresh on container changes.
    if let Some(d) = docker {
        let hive_bg = hive.clone();
        let extra = args.extra_ports.clone();
        let pinned = args.static_backends.clone();
        tokio::spawn(async move {
            d.watch_events(hive_bg, extra, pinned).await;
        });
    }

    let mut app = Router::new()
        .route("/health", get(hive::health))
        .route("/load", get(hive::server_load))
        .route("/v1/models", get(hive::list_models))
        .route("/v1/chat/completions", post(hive::chat_completions))
        .route("/v1/completions", post(hive::completions))
        .route("/v1/embeddings", post(hive::embeddings))
        .route("/api/hive/endpoints", get(hive::list_endpoints))
        .route("/api/hive/backends", get(hive::list_backends))
        .route("/api/hive/queries", get(hive::list_queries))
        .with_state(hive);

    // Browser UIs (HiveChat) need CORS; CLI/curl don't care.
    if !args.cors_origin.is_empty() {
        use tower_http::cors::{Any, CorsLayer};
        let layer = if args.cors_origin == "*" {
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any)
        } else {
            match args.cors_origin.parse::<axum::http::HeaderValue>() {
                Ok(origin) => CorsLayer::new()
                    .allow_origin(origin)
                    .allow_methods(Any)
                    .allow_headers(Any),
                Err(e) => {
                    tracing::warn!(%e, "invalid --cors-origin, CORS disabled");
                    CorsLayer::new()
                }
            }
        };
        app = app.layer(layer);
    }

    let addr = SocketAddr::from((args.bind, args.port));
    tracing::info!("🐝 HivLLM hive listening on http://{addr}");
    tracing::info!("   GET  /v1/models");
    tracing::info!("   POST /v1/chat/completions  (route by `model`)");
    tracing::info!("   POST /v1/completions       (route by `model`)");
    tracing::info!("   POST /v1/embeddings        (route by `model`)");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

// `anyhow` is used only for main's error type; add it as a tiny dep-free alias.
mod anyhow {
    pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
}
