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
    extract::Request,
    http::{header, HeaderValue, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
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
use logging::{JsonLinesSink, RequestLogger, Rotation};

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

    /// Roll the query log once it would exceed this many MiB
    /// (0 = never roll). Rolled files are timestamped next to the log.
    #[arg(long, default_value_t = 100)]
    log_max_mb: u64,

    /// Rolled query-log archives to keep (0 = keep all)
    #[arg(long, default_value_t = 20)]
    log_keep: usize,

    /// Gzip rolled query-log archives
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    log_compress: bool,

    /// Upstream connect timeout in seconds (a dead backend fails over
    /// to the next candidate after this)
    #[arg(long, default_value_t = 5)]
    connect_timeout: u64,

    /// Upstream read timeout in seconds: longest silence tolerated from a
    /// backend (0 = none). Non-streaming backends answer only once the
    /// generation is done, so this also caps non-streaming generations;
    /// streams are never cut while tokens keep flowing.
    #[arg(long, default_value_t = 600)]
    read_timeout: u64,

    /// Largest accepted request body in MiB (0 = unlimited). Long
    /// contexts and base64 images easily exceed a few MB.
    #[arg(long, default_value_t = 64)]
    max_body_mb: usize,

    /// Require `Authorization: Bearer <key>` on every route but /health.
    /// Prefer the env var: flags are visible in `ps`.
    #[arg(long, env = "HIVLLM_API_KEY", hide_env_values = true, default_value = "")]
    api_key: String,

    /// Forward the client's `Authorization` header to backends. Off by
    /// default: the token would reach every candidate backend. With
    /// --api-key, that forwards the hive key itself (fine when the whole
    /// fleet shares it).
    #[arg(long)]
    forward_auth: bool,

    /// Instance id stamped on responses and Via paths, so peer hives
    /// recognise this one whatever address they use. Random by default;
    /// pin it for stable ids in logs. No commas.
    #[arg(long)]
    hive_id: Option<String>,

    /// CORS allowed origins for browser UIs: "local" = pages served from
    /// localhost / 127.0.0.1 / [::1] on any port; "*" = any site (lets any
    /// web page you visit read the query log); a comma-separated origin
    /// list; empty = disabled.
    #[arg(long, default_value = "local")]
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
            LogFormat::Jsonl => Arc::new(
                JsonLinesSink::open_with(
                    &args.log_file,
                    Rotation {
                        max_bytes: args.log_max_mb.saturating_mul(1024 * 1024),
                        keep: args.log_keep,
                        compress: args.log_compress,
                    },
                )
                .await?,
            ),
        };
        tracing::info!(
            file = %args.log_file,
            sink = sink.name(),
            max_mb = args.log_max_mb,
            keep = args.log_keep,
            compress = args.log_compress,
            "🐝 query log enabled"
        );
        RequestLogger::new().with_sink(sink)
    };
    let read_timeout = (args.read_timeout > 0).then(|| Duration::from_secs(args.read_timeout));
    let hive = Hive::new(args.port)
        .with_logger(logger)
        .with_log_options(args.log_truncate, args.log_max_chars)
        .with_timeouts(Duration::from_secs(args.connect_timeout), read_timeout)
        .with_forward_auth(args.forward_auth)
        .with_max_body((args.max_body_mb > 0).then(|| args.max_body_mb.saturating_mul(1024 * 1024)));
    let hive = match args.hive_id {
        Some(id) if !hive::valid_hive_id(&id) => {
            return Err(format!("invalid --hive-id {id:?}: printable ASCII, no spaces or commas").into());
        }
        Some(id) => hive.with_hive_id(id),
        None => hive,
    };

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

    if args.api_key.is_empty() && !args.bind.is_loopback() {
        tracing::warn!(bind = %args.bind, "hive exposed without --api-key: anyone who can reach it can use every model and read the query log");
    }
    let hive_id = hive.own_id();
    let app = build_router(hive, &args.cors_origin, &args.api_key);

    let addr = SocketAddr::from((args.bind, args.port));
    tracing::info!("🐝 HivLLM hive {} listening on http://{addr}", hive_id);
    tracing::info!("   GET  /v1/models");
    tracing::info!("   POST /v1/chat/completions  (route by `model`)");
    tracing::info!("   POST /v1/completions       (route by `model`)");
    tracing::info!("   POST /v1/embeddings        (route by `model`)");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}

/// All routes, behind API-key auth (when a key is set) and CORS (outermost,
/// so preflights are answered before auth).
fn build_router(hive: Hive, cors_origin: &str, api_key: &str) -> Router {
    let mut app = hive::routes(hive);

    if !api_key.is_empty() {
        let expected: Arc<[u8]> = format!("Bearer {api_key}").into_bytes().into();
        app = app.layer(axum::middleware::from_fn(move |req: Request, next: Next| {
            let expected = expected.clone();
            async move { require_api_key(&expected, req, next).await }
        }));
    }
    if let Some(cors) = cors_layer(cors_origin) {
        app = app.layer(cors);
    }
    app
}

async fn require_api_key(expected: &[u8], req: Request, next: Next) -> Response {
    // Container healthchecks carry no credentials.
    if req.uri().path() == "/health" {
        return next.run(req).await;
    }
    let presented = req
        .headers()
        .get(header::AUTHORIZATION)
        .map(|v| v.as_bytes())
        .unwrap_or_default();
    if constant_time_eq(presented, expected) {
        return next.run(req).await;
    }
    let body = serde_json::json!({
        "error": { "message": "missing or invalid API key", "type": "hive_error" }
    });
    (StatusCode::UNAUTHORIZED, axum::Json(body)).into_response()
}

/// Compare without leaking the position of the first mismatch.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    a.len() == b.len() && a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// Browser pages served from this machine, any port.
fn is_local_origin(origin: &HeaderValue) -> bool {
    let Ok(origin) = origin.to_str() else {
        return false;
    };
    let Some(rest) = origin
        .strip_prefix("http://")
        .or_else(|| origin.strip_prefix("https://"))
    else {
        return false;
    };
    let host = match rest.strip_prefix("[::1]") {
        Some(port) => return port.is_empty() || port.starts_with(':'),
        None => rest.split(':').next().unwrap_or(rest),
    };
    host == "localhost" || host == "127.0.0.1"
}

fn cors_layer(spec: &str) -> Option<tower_http::cors::CorsLayer> {
    use tower_http::cors::{AllowHeaders, AllowOrigin, Any, CorsLayer};
    let origin = match spec.trim() {
        "" => return None,
        "*" => AllowOrigin::any(),
        "local" => AllowOrigin::predicate(|o, _| is_local_origin(o)),
        list => {
            let origins: Vec<HeaderValue> = list
                .split(',')
                .map(str::trim)
                .filter(|o| !o.is_empty())
                .filter_map(|o| match o.parse() {
                    Ok(v) => Some(v),
                    Err(e) => {
                        tracing::warn!(origin = o, %e, "invalid --cors-origin entry, skipped");
                        None
                    }
                })
                .collect();
            if origins.is_empty() {
                return None;
            }
            AllowOrigin::list(origins)
        }
    };
    // Mirrored, not `*`: browsers never let a wildcard cover `Authorization`.
    Some(
        CorsLayer::new()
            .allow_origin(origin)
            .allow_methods(Any)
            .allow_headers(AllowHeaders::mirror_request()),
    )
}

// `anyhow` is used only for main's error type; add it as a tiny dep-free alias.
mod anyhow {
    pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_origins_only() {
        let ok = |o: &str| is_local_origin(&HeaderValue::from_str(o).unwrap());
        assert!(ok("http://localhost:5173"));
        assert!(ok("http://localhost"));
        assert!(ok("https://127.0.0.1:8443"));
        assert!(ok("http://[::1]:3000"));
        assert!(!ok("https://evil.example"));
        assert!(!ok("http://localhost.evil.example"));
        assert!(!ok("http://127.0.0.1.evil.example:80"));
        assert!(!ok("http://[::1].evil"));
        assert!(!ok("null"));
    }

    async fn serve(app: Router) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        url
    }

    #[tokio::test]
    async fn api_key_guards_everything_but_health() {
        let url = serve(build_router(Hive::new(0), "local", "k3y")).await;
        let c = reqwest::Client::new();
        let get = |path: &str, key: Option<&str>| {
            let mut r = c.get(format!("{url}{path}"));
            if let Some(k) = key {
                r = r.bearer_auth(k);
            }
            r.send()
        };
        assert_eq!(get("/health", None).await.unwrap().status(), 200);
        assert_eq!(get("/api/hive/queries", None).await.unwrap().status(), 401);
        assert_eq!(get("/v1/models", Some("wrong")).await.unwrap().status(), 401);
        assert_eq!(get("/v1/models", Some("k3y")).await.unwrap().status(), 200);
    }

    #[tokio::test]
    async fn cors_answers_local_pages_only_and_allows_authorization() {
        let url = serve(build_router(Hive::new(0), "local", "k3y")).await;
        let preflight = |origin: &'static str| {
            reqwest::Client::new()
                .request(reqwest::Method::OPTIONS, format!("{url}/api/hive/queries"))
                .header("Origin", origin)
                .header("Access-Control-Request-Method", "GET")
                .header("Access-Control-Request-Headers", "authorization")
                .send()
        };
        let local = preflight("http://localhost:5173").await.unwrap();
        let h = local.headers();
        assert_eq!(h["access-control-allow-origin"], "http://localhost:5173");
        assert!(h["access-control-allow-headers"]
            .to_str()
            .unwrap()
            .contains("authorization"));
        let evil = preflight("https://evil.example").await.unwrap();
        assert!(evil.headers().get("access-control-allow-origin").is_none());
        assert!(cors_layer("").is_none());
    }
}
