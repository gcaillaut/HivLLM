//! HivLLM — All your models. One sticky hive. 🐝
//!
//! Auto-discovers OpenAI-compatible endpoints on this machine
//! (localhost port scan + process/docker scan) and serves them
//! behind a single endpoint:
//!
//! - `GET /v1/models` — aggregated model list
//! - `POST /v1/chat/completions`, `/v1/completions`, `/v1/embeddings` —
//!   routed by `model`
//!   (JSON body field, `?model=` query param wins as override).
//!   Same model name on several endpoints => round-robin load-balancing.

mod discovery;
mod hive;

use axum::{
    routing::{get, post},
    Router,
};
use clap::Parser;
use std::{net::SocketAddr, time::Duration};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use hive::Hive;

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
    let hive = Hive::new(args.port);

    // Initial discovery before serving.
    hive.refresh(&args.extra_ports).await;

    // Background re-scan loop.
    if args.scan_interval > 0 {
        let hive_bg = hive.clone();
        let extra = args.extra_ports.clone();
        let interval = args.scan_interval;
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(interval)).await;
                hive_bg.refresh(&extra).await;
            }
        });
    }

    let app = Router::new()
        .route("/health", get(hive::health))
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
