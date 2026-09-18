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
mod hive;
mod load;
mod logging;

use axum::{
    routing::{get, post},
    Router,
};
use clap::{Parser, ValueEnum};
use std::{net::SocketAddr, sync::Arc, time::Duration};
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

    /// Extra localhost ports to probe (in addition to well-known ones)
    #[arg(long, value_delimiter = ',')]
    extra_ports: Vec<u16>,

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

    // Initial discovery before serving.
    hive.refresh(&args.extra_ports).await;

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
        let interval = args.scan_interval;
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(interval)).await;
                hive_bg.refresh(&extra).await;
                hive_bg.log_view().await;
            }
        });
    }

    let app = Router::new()
        .route("/health", get(hive::health))
        .route("/load", get(hive::server_load))
        .route("/v1/models", get(hive::list_models))
        .route("/v1/chat/completions", post(hive::chat_completions))
        .route("/v1/completions", post(hive::completions))
        .route("/v1/embeddings", post(hive::embeddings))
        .route("/api/hive/endpoints", get(hive::list_endpoints))
        .with_state(hive);

    let addr = SocketAddr::from(([127, 0, 0, 1], args.port));
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
