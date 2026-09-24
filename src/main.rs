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

mod config;
mod discovery;
mod docker;
mod hive;
mod load;
mod logging;
mod multipart;

use axum::{
    extract::Request,
    http::{header, HeaderValue, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Router,
};
use clap::Parser;
use std::{
    net::{IpAddr, SocketAddr},
    sync::Arc,
    time::Duration,
};
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

use config::{Config, LogFormat, StaticBackend};
use hive::Hive;
use logging::{JsonLinesSink, RequestLogger, Rotation};

/// Command line. Every flag overrides the config file key noted in its
/// help; unset flags leave the file (or the built-in default) alone.
#[derive(Parser, Debug)]
#[command(name = "hivllm", about = "All your models. One sticky hive. 🐝")]
struct Args {
    /// YAML config file (see config/hivllm.example.yaml). Flags and env
    /// variables override its keys.
    #[arg(long, env = "HIVLLM_CONFIG")]
    config: Option<std::path::PathBuf>,

    /// Print the effective configuration (inline secrets masked) and exit
    #[arg(long)]
    print_config: bool,

    /// [server.port] Port for the unified hive endpoint (default 8335 =
    /// BEES 🐝). In containers prefer HIVLLM_PORT: the image's HEALTHCHECK
    /// follows it.
    #[arg(long, env = "HIVLLM_PORT")]
    port: Option<u16>,

    /// [server.bind] Interface to bind (default 127.0.0.1 keeps the hive
    /// local; 0.0.0.0 exposes it, e.g. from a container)
    #[arg(long)]
    bind: Option<IpAddr>,

    /// [discovery.extra_ports] Extra localhost ports to probe (in addition
    /// to well-known ones). Comma-separated and/or repeatable.
    #[arg(long, value_delimiter = ',')]
    extra_ports: Option<Vec<u16>>,

    /// [discovery.static_backends] Static backends discovery can't see:
    /// base URLs probed for `/v1/models` on every rescan (docker service
    /// names, remote hosts, …); may include a gateway's `/v1` prefix
    /// (`http://host:8180/general-stage1/v1`). Replaces the file's list;
    /// keys for them stay in the file. Comma-separated and/or repeatable.
    #[arg(long, value_delimiter = ',')]
    static_backends: Option<Vec<String>>,

    /// [discovery.docker_socket] Docker Engine socket for container
    /// discovery (opt-in labels `hivllm.enable=true`, `hivllm.port=8080`).
    /// Empty (default) = disabled.
    #[arg(long)]
    docker_socket: Option<String>,

    /// [discovery.scan_interval] Re-scan interval in seconds (default 30;
    /// 0 = scan once at startup)
    #[arg(long)]
    scan_interval: Option<u64>,

    /// [discovery.drop_after] Consecutive failed scans before a backend
    /// leaves the hive (default 3; 1 = at the first failure). Docker
    /// containers always leave at once.
    #[arg(long)]
    drop_after: Option<u32>,

    /// [routing.load_interval] Backend load poll interval in seconds
    /// (default 5; 0 = no polling, plain round-robin)
    #[arg(long)]
    load_interval: Option<u64>,

    /// [log.file] Query log file (default hivllm-queries.jsonl). Empty =
    /// no query logging.
    #[arg(long)]
    log_file: Option<String>,

    /// [log.format] Query log format (default jsonl)
    #[arg(long, value_enum)]
    log_format: Option<LogFormat>,

    /// [log.truncate] Truncation of long text in query logs (default none)
    #[arg(long, value_enum)]
    log_truncate: Option<logging::Truncate>,

    /// [log.max_chars] Max chars per text field with `--log-truncate
    /// chars` (default 2000)
    #[arg(long)]
    log_max_chars: Option<usize>,

    /// [log.max_mb] Roll the query log once it would exceed this many MiB
    /// (default 100; 0 = never roll)
    #[arg(long)]
    log_max_mb: Option<u64>,

    /// [log.keep] Rolled query-log archives to keep (default 20; 0 = all)
    #[arg(long)]
    log_keep: Option<usize>,

    /// [log.compress] Gzip rolled query-log archives (default true)
    #[arg(long, num_args = 0..=1, default_missing_value = "true")]
    log_compress: Option<bool>,

    /// [routing.connect_timeout] Upstream connect timeout in seconds
    /// (default 5): a dead backend fails over after this
    #[arg(long)]
    connect_timeout: Option<u64>,

    /// [routing.read_timeout] Longest silence tolerated from a backend, in
    /// seconds (default 600; 0 = none). Also caps non-streaming
    /// generations; streams are never cut while tokens flow.
    #[arg(long)]
    read_timeout: Option<u64>,

    /// [server.max_body_mb] Largest accepted request body in MiB
    /// (default 64; 0 = unlimited)
    #[arg(long)]
    max_body_mb: Option<usize>,

    /// [server.api_key] Require `Authorization: Bearer <key>` on every
    /// route but /health. Prefer the env var (flags show in `ps`) or
    /// `server.api_key_env` in the file.
    #[arg(long, env = "HIVLLM_API_KEY", hide_env_values = true)]
    api_key: Option<String>,

    /// [routing.forward_auth] Forward the client's `Authorization` header
    /// to backends without a configured key (default false: the token
    /// would reach every candidate)
    #[arg(long, num_args = 0..=1, default_missing_value = "true")]
    forward_auth: Option<bool>,

    /// [server.hive_id] Instance id stamped on responses and Via paths.
    /// Random by default; pin it for stable ids in logs. No commas.
    #[arg(long)]
    hive_id: Option<String>,

    /// [server.cors_origin] CORS allowed origins for browser UIs (default
    /// "local": pages from localhost / 127.0.0.1 / [::1]); "*" = any site
    /// (lets any page you visit read the query log); a comma-separated
    /// origin list; empty = disabled.
    #[arg(long)]
    cors_origin: Option<String>,
}

/// Command-line flag → config key, for every flag that has one.
#[cfg(test)]
const FLAG_KEYS: &[(&str, &str)] = &[
    ("port", "server.port"),
    ("bind", "server.bind"),
    ("hive_id", "server.hive_id"),
    ("api_key", "server.api_key"),
    ("cors_origin", "server.cors_origin"),
    ("max_body_mb", "server.max_body_mb"),
    ("extra_ports", "discovery.extra_ports"),
    ("static_backends", "discovery.static_backends"),
    ("docker_socket", "discovery.docker_socket"),
    ("scan_interval", "discovery.scan_interval"),
    ("drop_after", "discovery.drop_after"),
    ("load_interval", "routing.load_interval"),
    ("connect_timeout", "routing.connect_timeout"),
    ("read_timeout", "routing.read_timeout"),
    ("forward_auth", "routing.forward_auth"),
    ("log_file", "log.file"),
    ("log_format", "log.format"),
    ("log_truncate", "log.truncate"),
    ("log_max_chars", "log.max_chars"),
    ("log_max_mb", "log.max_mb"),
    ("log_keep", "log.keep"),
    ("log_compress", "log.compress"),
];

/// Flags without a config key: they pick or print the config itself.
#[cfg(test)]
const META_FLAGS: &[&str] = &["config", "print_config", "help", "version"];

impl Args {
    /// Overlay every flag that was given onto `cfg`.
    fn apply(self, cfg: &mut Config) {
        fn set<T>(slot: &mut T, v: Option<T>) {
            if let Some(v) = v {
                *slot = v;
            }
        }
        let (srv, disc, rt, log) = (&mut cfg.server, &mut cfg.discovery, &mut cfg.routing, &mut cfg.log);
        set(&mut srv.port, self.port);
        set(&mut srv.bind, self.bind);
        if self.hive_id.is_some() {
            srv.hive_id = self.hive_id;
        }
        if let Some(key) = self.api_key {
            // A flag / HIVLLM_API_KEY replaces whatever the file said.
            srv.api_key = Some(key);
            srv.api_key_env = None;
        }
        set(&mut srv.cors_origin, self.cors_origin);
        set(&mut srv.max_body_mb, self.max_body_mb);
        set(&mut disc.extra_ports, self.extra_ports);
        if let Some(urls) = self.static_backends {
            disc.static_backends = urls.into_iter().map(StaticBackend::Url).collect();
        }
        set(&mut disc.docker_socket, self.docker_socket);
        set(&mut disc.scan_interval, self.scan_interval);
        set(&mut disc.drop_after, self.drop_after);
        set(&mut rt.load_interval, self.load_interval);
        set(&mut rt.connect_timeout, self.connect_timeout);
        set(&mut rt.read_timeout, self.read_timeout);
        set(&mut rt.forward_auth, self.forward_auth);
        set(&mut log.file, self.log_file);
        set(&mut log.format, self.log_format);
        set(&mut log.truncate, self.log_truncate);
        set(&mut log.max_chars, self.log_max_chars);
        set(&mut log.max_mb, self.log_max_mb);
        set(&mut log.keep, self.log_keep);
        set(&mut log.compress, self.log_compress);
    }
}

#[tokio::main]
async fn main() -> std::process::ExitCode {
    match run().await {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("hivllm: {e}");
            std::process::ExitCode::FAILURE
        }
    }
}

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::new(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "hivllm=info,tower_http=info".into()),
        ))
        .with(tracing_subscriber::fmt::layer())
        .init();

    let args = Args::parse();
    let mut cfg = match &args.config {
        Some(path) => Config::load(path)?,
        None => Config::default(),
    };
    let (config_path, print_config) = (args.config.clone(), args.print_config);
    args.apply(&mut cfg);
    if print_config {
        print!("{}", cfg.to_yaml_redacted());
        return Ok(());
    }
    if let Some(path) = &config_path {
        tracing::info!(config = %path.display(), "🐝 config loaded");
    }
    let api_key = cfg.server_api_key()?.unwrap_or_default();
    let credentials = cfg.credentials()?;
    let static_backends = cfg.static_urls();
    let (srv, disc, rt, log) = (&cfg.server, &cfg.discovery, &cfg.routing, &cfg.log);

    let logger = if log.file.is_empty() {
        RequestLogger::new()
    } else {
        let sink: Arc<dyn logging::LogSink> = match log.format {
            LogFormat::Jsonl => Arc::new(
                JsonLinesSink::open_with(
                    &log.file,
                    Rotation {
                        max_bytes: log.max_mb.saturating_mul(1024 * 1024),
                        keep: log.keep,
                        compress: log.compress,
                    },
                )
                .await?,
            ),
        };
        tracing::info!(
            file = %log.file,
            sink = sink.name(),
            max_mb = log.max_mb,
            keep = log.keep,
            compress = log.compress,
            "🐝 query log enabled"
        );
        RequestLogger::new().with_sink(sink).spawn_writer()
    };
    let log_queue = logger.clone();
    let read_timeout = (rt.read_timeout > 0).then(|| Duration::from_secs(rt.read_timeout));
    let hive = Hive::new(srv.port)
        .with_logger(logger)
        .with_log_options(log.truncate, log.max_chars)
        .with_timeouts(Duration::from_secs(rt.connect_timeout), read_timeout)
        .with_forward_auth(rt.forward_auth)
        .with_drop_after(disc.drop_after)
        .with_max_body((srv.max_body_mb > 0).then(|| srv.max_body_mb.saturating_mul(1024 * 1024)));
    let hive = match srv.hive_id.clone() {
        Some(id) if !hive::valid_hive_id(&id) => {
            return Err(format!("invalid hive id {id:?}: printable ASCII, no spaces or commas").into());
        }
        Some(id) => hive.with_hive_id(id),
        None => hive,
    };
    if !credentials.is_empty() {
        tracing::info!(backends = credentials.len(), "🐝 per-backend API keys loaded");
    }
    let hive = hive.with_credentials(credentials);

    // Docker socket discovery (opt-in): containers labeled
    // `hivllm.enable=true` join the hive; lifecycle events refresh it.
    let docker = if disc.docker_socket.is_empty() {
        None
    } else {
        match docker::DockerDiscovery::connect(&disc.docker_socket).await {
            Ok(d) => {
                tracing::info!(socket = %disc.docker_socket, "🐝 docker discovery enabled");
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
        .refresh(&disc.extra_ports, &static_backends, docker.as_ref())
        .await;

    // Prime load state so the first requests are already load-informed.
    // The load refresh announces the hive view; without polling, the
    // discovery refresh announces it instead — either way, exactly once.
    if rt.load_interval > 0 {
        hive.refresh_load().await;
        let hive_bg = hive.clone();
        let interval = rt.load_interval;
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
    if disc.scan_interval > 0 {
        let hive_bg = hive.clone();
        let extra = disc.extra_ports.clone();
        let pinned = static_backends.clone();
        let dock = docker.clone();
        let interval = disc.scan_interval;
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
        let extra = disc.extra_ports.clone();
        let pinned = static_backends.clone();
        tokio::spawn(async move {
            d.watch_events(hive_bg, extra, pinned).await;
        });
    }

    if api_key.is_empty() && !srv.bind.is_loopback() {
        tracing::warn!(bind = %srv.bind, "hive exposed without an API key: anyone who can reach it can use every model and read the query log");
    }
    let hive_id = hive.own_id();
    let app = build_router(hive, &srv.cors_origin, &api_key);

    let addr = SocketAddr::from((srv.bind, srv.port));
    tracing::info!("🐝 HivLLM hive {} listening on http://{addr}", hive_id);
    tracing::info!("   GET  /v1/models, /v1/models/{{id}}");
    tracing::info!("   POST /v1/chat/completions, /v1/completions, /v1/embeddings, /v1/responses");
    tracing::info!("   POST rerank / score / pooling / classify / tokenize / detokenize");
    tracing::info!("   POST /v1/audio/speech, /v1/audio/transcriptions, /v1/audio/translations");
    tracing::info!("        (all routed by `model`)");

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;
    // In-flight requests are done; their log entries (streams log from a
    // task woken as the body ends) get a moment to queue, then land on disk.
    tokio::time::sleep(Duration::from_millis(200)).await;
    log_queue.flush().await;
    tracing::info!("🐝 hive stopped, query log flushed");
    Ok(())
}

/// Ctrl-C, or SIGTERM (`docker stop`) on Unix.
async fn shutdown_signal() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let term = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut s) => {
                s.recv().await;
            }
            Err(_) => std::future::pending::<()>().await,
        }
    };
    #[cfg(not(unix))]
    let term = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => {},
        _ = term => {},
    }
    tracing::info!("🐝 shutting down: finishing in-flight requests");
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


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_flag_has_a_config_key() {
        use clap::CommandFactory;
        let defaults = serde_yaml_ng::to_value(Config::default()).unwrap();
        for arg in Args::command().get_arguments() {
            let id = arg.get_id().as_str();
            if META_FLAGS.contains(&id) {
                continue;
            }
            let (_, key) = FLAG_KEYS
                .iter()
                .find(|(flag, _)| *flag == id)
                .unwrap_or_else(|| panic!("flag --{id} has no config key"));
            let mut node = &defaults;
            for part in key.split('.') {
                node = node
                    .get(part)
                    .unwrap_or_else(|| panic!("--{id} maps to {key}, which the config lacks"));
            }
            // The flag's help names its key, so `--help` documents it.
            let help = arg.get_help().map(|h| h.to_string()).unwrap_or_default();
            assert!(help.contains(&format!("[{key}]")), "--{id} help should mention [{key}]");
        }
    }

    #[test]
    fn flags_override_the_file_and_the_file_overrides_defaults() {
        let mut cfg = Config::from_yaml(
            "server:\n  port: 8000\n  api_key_env: SOME_VAR\ndiscovery:\n  scan_interval: 10\nlog:\n  compress: true\n",
        )
        .unwrap();
        let args = Args::try_parse_from([
            "hivllm",
            "--port",
            "9000",
            "--log-compress",
            "false",
            "--forward-auth",
            "--api-key",
            "flag-key",
            "--static-backends",
            "http://a:1,http://b:2",
        ])
        .unwrap();
        args.apply(&mut cfg);
        assert_eq!(cfg.server.port, 9000);
        assert!(!cfg.log.compress);
        assert!(cfg.routing.forward_auth);
        assert_eq!(cfg.discovery.scan_interval, 10, "untouched by flags: the file's value");
        assert_eq!(cfg.routing.load_interval, 5, "in neither: the default");
        assert_eq!(cfg.server_api_key().unwrap().as_deref(), Some("flag-key"));
        assert_eq!(cfg.static_urls(), vec!["http://a:1", "http://b:2"]);
    }

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
