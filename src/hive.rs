//! Unified hive: aggregated `/v1/models` + routed `/v1/chat/completions`,
//! `/v1/completions`, `/v1/embeddings` with load-aware routing: the
//! least-loaded backend wins, ties and unknown loads round-robin.

use axum::{
    body::Body,
    extract::{Query, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use bytes::Bytes;
use chrono::Utc;
use futures::{stream, StreamExt};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::{oneshot, Mutex, RwLock};

use crate::discovery::{discover, DiscoveredEndpoint};
use crate::load::{effective_load, probe_backend, HivLoadProbe, Load, LoadProbe, VllmLoadProbe, VllmMetricsProbe};
use crate::logging::{
    extract_response, truncate_value, LogEntry, LoggedResponse, RequestLogger, StreamAcc,
    StreamSummary, TeeStream, Truncate,
};

#[derive(Clone)]
pub struct Hive {
    pub client: reqwest::Client,
    /// Port the hive itself listens on — always excluded from discovery.
    own_port: u16,
    /// This hive's id on the Via path (its served base_url).
    own_id: String,
    logger: RequestLogger,
    log_truncate: Truncate,
    log_max_chars: usize,
    /// Provider load probes: per-model hive aggregates first, then vLLM
    /// `/metrics`, then vLLM `/load`.
    probes: Vec<Arc<dyn LoadProbe>>,
    /// Probes usable when a backend serves several models: no bare
    /// `/load` — a marker-less server-level number can't be attributed to
    /// one model of many (a stale hive global would smear across models).
    scoped_probes: Vec<Arc<dyn LoadProbe>>,
    /// (endpoint id, model) → last polled load. Rebuilt by every
    /// [`Hive::refresh_load`].
    loads: Arc<RwLock<HashMap<(String, String), Load>>>,
    /// base_url → requests currently being served by the hive. Used as a
    /// load approximation for backends without load tracking.
    inflight: Arc<Mutex<HashMap<String, u64>>>,
    /// Last rendered hive view (change detection for stdout logging).
    last_view: Arc<Mutex<String>>,
    endpoints: Arc<RwLock<Vec<DiscoveredEndpoint>>>,
    /// (model, effective load) -> next index. Equal effective loads rotate
    /// round-robin. All-idle degrades to plain round-robin.
    rr: Arc<Mutex<HashMap<(String, u64), usize>>>,
}

impl Hive {
    pub fn new(own_port: u16) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(300))
            .build()
            .expect("reqwest client");
        Self {
            client,
            own_port,
            own_id: format!("http://127.0.0.1:{own_port}"),
            logger: RequestLogger::new(),
            log_truncate: Truncate::None,
            log_max_chars: 2000,
            probes: vec![
                Arc::new(HivLoadProbe),
                Arc::new(VllmMetricsProbe),
                Arc::new(VllmLoadProbe),
            ],
            scoped_probes: vec![Arc::new(HivLoadProbe), Arc::new(VllmMetricsProbe)],
            loads: Arc::new(RwLock::new(HashMap::new())),
            inflight: Arc::new(Mutex::new(HashMap::new())),
            last_view: Arc::new(Mutex::new(String::new())),
            endpoints: Arc::new(RwLock::new(Vec::new())),
            rr: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn with_logger(mut self, logger: RequestLogger) -> Self {
        self.logger = logger;
        self
    }

    /// Override the Via id (tests serve on ephemeral ports).
    pub fn with_own_id(mut self, id: String) -> Self {
        self.own_id = id;
        self
    }

    pub fn with_log_options(mut self, truncate: Truncate, max_chars: usize) -> Self {
        self.log_truncate = truncate;
        self.log_max_chars = max_chars;
        self
    }

    /// Record one query, applying the truncation policy first.
    async fn log(&self, mut entry: LogEntry) {
        entry.request = truncate_value(entry.request, self.log_truncate, self.log_max_chars);
        if let Some(r) = entry.response.take() {
            entry.response = Some(r.truncated(self.log_truncate, self.log_max_chars));
        }
        self.logger.log(entry).await;
    }

    pub async fn refresh(&self, extra_ports: &[u16]) {
        let found = discover(&self.client, extra_ports, &[self.own_port]).await;
        *self.endpoints.write().await = found;
    }

    pub async fn snapshot(&self) -> Vec<DiscoveredEndpoint> {
        self.endpoints.read().await.clone()
    }

    /// This hive's own id on the Via path (same shape as member base_urls).
    fn own_id(&self) -> String {
        self.own_id.clone()
    }

    /// Aggregate load of the whole hive (`model = None`) or of one model,
    /// as `(number, exact)`. Served as `GET /load[?model=]` with marker
    /// `"hivllm": {"exact": bool}` so upstream hives route on real numbers:
    /// only positively-sighted reports vote exact (a bare `0` never does),
    /// approximations never launder into exact across hops.
    ///
    /// Median of exact votes; with none exact, median of approximations
    /// (usually `~0`); no members → `(0, false)`. Even counts take the
    /// floored average of the middle pair.
    pub async fn aggregate_load(&self, model: Option<&str>) -> (u64, bool) {
        let endpoints = self.endpoints.read().await;
        let loads = self.loads.read().await;
        let inflight = self.inflight.lock().await;
        let mut exact = Vec::new();
        let mut approx = Vec::new();
        for ep in endpoints.iter() {
            for m in &ep.models {
                if model.is_some_and(|wanted| wanted != m) {
                    continue;
                }
                let server = loads
                    .get(&(ep.id.clone(), m.clone()))
                    .copied()
                    .unwrap_or(Load::Unknown);
                let flying = inflight.get(&ep.base_url).copied().unwrap_or(0);
                let (num, is_exact) = effective_load(server, flying);
                if is_exact {
                    exact.push(num);
                } else {
                    approx.push(num);
                }
            }
        }
        if exact.is_empty() {
            (Self::median(approx), false)
        } else {
            (Self::median(exact), true)
        }
    }

    fn median(mut vals: Vec<u64>) -> u64 {
        vals.sort_unstable();
        match vals.len() {
            0 => 0,
            n if n % 2 == 1 => vals[n / 2],
            n => vals[n / 2 - 1].saturating_add(vals[n / 2]) / 2,
        }
    }

    /// Poll load probes for every (endpoint, model) pair (concurrently)
    /// and replace the load map. Backends that fail probing stay `Unknown`
    /// — usable, routed round-robin. Run on a short interval; never
    /// per-request.
    pub async fn refresh_load(&self) {
        let endpoints = self.endpoints.read().await.clone();
        let mut jobs = Vec::new();
        for ep in &endpoints {
            for m in &ep.models {
                jobs.push((
                    ep.id.clone(),
                    ep.base_url.clone(),
                    m.clone(),
                    ep.models.len(),
                ));
            }
        }
        let probed: HashMap<(String, String), Load> = stream::iter(jobs)
            .map(|(id, base_url, model, n_models)| {
                // Multi-model backends skip the bare-`/load` fallback: a
                // marker-less server-level number is that model's load only
                // for single-model servers. Box-level `/metrics` still
                // applies (a shared box is busy for every model on it).
                let chain = if n_models > 1 {
                    self.scoped_probes.clone()
                } else {
                    self.probes.clone()
                };
                let client = self.client.clone();
                async move {
                    let load = probe_backend(&chain, &client, &base_url, &model).await;
                    ((id, model), load)
                }
            })
            .buffer_unordered(32)
            .collect()
            .await;
        let known = probed.values().filter(|l| matches!(l, Load::Known(_))).count();
        let probe_names: Vec<&str> = self.probes.iter().map(|p| p.name()).collect();
        tracing::debug!(?probe_names, known, total = probed.len(), "hive load refresh");
        *self.loads.write().await = probed;
        self.log_view().await;
    }

    async fn inflight_inc(&self, base_url: &str) {
        *self
            .inflight
            .lock()
            .await
            .entry(base_url.to_string())
            .or_insert(0) += 1;
    }

    async fn inflight_dec(&self, base_url: &str) {
        let mut inflight = self.inflight.lock().await;
        if let Some(n) = inflight.get_mut(base_url) {
            *n = n.saturating_sub(1);
        }
    }

    /// Split `http://127.0.0.1:9037` into `("127.0.0.1", Some(9037))`.
    fn split_host_port(base_url: &str) -> (String, Option<u16>) {
        let no_scheme = base_url.split("://").last().unwrap_or(base_url);
        match no_scheme.rfind(':') {
            Some(i) => (
                no_scheme[..i].to_string(),
                no_scheme[i + 1..].parse().ok(),
            ),
            None => (no_scheme.to_string(), None),
        }
    }

    /// Per-model backends with the SAME effective loads the balancer routes
    /// on. Backs both the stdout view and `GET /api/hive/backends`.
    pub async fn backends_view(&self) -> BackendsView {
        let endpoints = self.endpoints.read().await;
        let loads = self.loads.read().await;
        let inflight = self.inflight.lock().await;
        // model -> rows
        let mut models: HashMap<String, Vec<BackendInfo>> = HashMap::new();
        for ep in endpoints.iter() {
            for m in &ep.models {
                let server = loads
                    .get(&(ep.id.clone(), m.clone()))
                    .copied()
                    .unwrap_or(Load::Unknown);
                let flying = inflight.get(&ep.base_url).copied().unwrap_or(0);
                let (num, exact) = effective_load(server, flying);
                let (ip, port) = Self::split_host_port(&ep.base_url);
                models.entry(m.clone()).or_default().push(BackendInfo {
                    ip,
                    port,
                    load: num,
                    exact,
                });
            }
        }
        drop(inflight);
        drop(loads);
        drop(endpoints);
        let mut names: Vec<String> = models.keys().cloned().collect();
        names.sort_unstable();
        let view_models = names
            .into_iter()
            .map(|name| {
                // Decreasing load; exact reports before approximations on ties.
                let mut backends = models.remove(&name).unwrap_or_default();
                backends.sort_by(|a, b| {
                    b.load
                        .cmp(&a.load)
                        .then(b.exact.cmp(&a.exact))
                });
                ModelBackends {
                    id: name,
                    backends,
                }
            })
            .collect();
        BackendsView {
            models: view_models,
        }
    }

    /// Last `limit` query-log entries, newest first. Empty when no
    /// file-backed sink is configured.
    pub async fn recent_queries(&self, limit: usize) -> Vec<Value> {
        let Some(path) = self.logger.file_sink_path() else {
            return Vec::new();
        };
        let content = tokio::fs::read_to_string(&path).await.unwrap_or_default();
        content
            .lines()
            .rev()
            .take(limit)
            .filter_map(|line| serde_json::from_str(line).ok())
            .collect()
    }

    /// Render the per-model hive view: for each model, every backend with
    /// ip, port and the SAME effective load the balancer routes on —
    /// tracked `server_load`, or `~N` hive-observed in-flight requests
    /// when the server report doesn't cover them (no tracking, stuck at
    /// 0, …). Backends ordered by decreasing load.
    async fn render_view(&self) -> String {
        let view = self.backends_view().await;
        let n_backends: usize = view.models.iter().map(|m| m.backends.len()).sum();
        let mut out = format!(
            "🐝 hive: {}, {}",
            plural(view.models.len(), "model", "models"),
            plural(n_backends, "backend", "backends"),
        );
        for m in &view.models {
            out.push_str(&format!(
                "\n  {} ({}):",
                m.id,
                plural(m.backends.len(), "backend", "backends")
            ));
            for b in &m.backends {
                let port = b.port.map(|p| p.to_string()).unwrap_or("?".to_string());
                if b.exact {
                    out.push_str(&format!("\n    {}:{port} load={}", b.ip, b.load));
                } else {
                    out.push_str(&format!("\n    {}:{port} load=~{}", b.ip, b.load));
                }
            }
        }
        out
    }

    /// Log the per-model view, but only when it changed since last time
    /// (membership or loads), so the 5s load poll stays quiet when idle.
    pub async fn log_view(&self) {
        let view = self.render_view().await;
        let mut last = self.last_view.lock().await;
        if *last != view {
            *last = view.clone();
            tracing::info!("{view}");
        }
    }

    /// Order backends serving `model` by effective load (lowest first)
    /// `max(server_load, hive-observed in-flight)`, so a `/load` stuck at
    /// 0 without server-side tracking degrades to the hive's own signal
    /// instead of pretending the backend is idle. Equal effective loads
    /// rotate round-robin. The proxy tries candidates in order and fails
    /// over to the next one when a backend is unreachable.
    pub async fn candidates(&self, model: &str) -> Vec<String> {
        let endpoints = self.endpoints.read().await;
        let loads = self.loads.read().await;
        let inflight = self.inflight.lock().await;
        let mut ranked: Vec<(String, u64)> = Vec::new();
        for ep in endpoints
            .iter()
            .filter(|e| e.models.iter().any(|m| m == model))
        {
            let server = loads
                .get(&(ep.id.clone(), model.to_string()))
                .copied()
                .unwrap_or(Load::Unknown);
            let flying = inflight.get(&ep.base_url).copied().unwrap_or(0);
            let (eff, _) = effective_load(server, flying);
            ranked.push((ep.base_url.clone(), eff));
        }
        drop(inflight);
        drop(loads);
        drop(endpoints);
        ranked.sort_by_key(|(_, n)| *n); // stable: ties keep discovery order
        let mut rr = self.rr.lock().await;
        let mut out = Vec::with_capacity(ranked.len());
        // Rotate each run of equal effective load independently.
        let mut i = 0;
        while i < ranked.len() {
            let load = ranked[i].1;
            let mut j = i + 1;
            while j < ranked.len() && ranked[j].1 == load {
                j += 1;
            }
            let group = &mut ranked[i..j];
            if group.len() > 1 {
                let counter = rr.entry((model.to_string(), load)).or_insert(0);
                let rot = *counter % group.len();
                *counter = counter.wrapping_add(1);
                group.rotate_left(rot);
            }
            out.extend(group.iter().map(|(u, _)| u.clone()));
            i = j;
        }
        out
    }

    /// [`Hive::candidates`] minus already-visited backends. A request that
    /// arrives with every backend on its Via path is a loop — the proxy
    /// answers 502 instead of forwarding forever.
    pub async fn candidates_excluding(
        &self,
        model: &str,
        visited: &HashSet<String>,
    ) -> Vec<String> {
        self.candidates(model)
            .await
            .into_iter()
            .filter(|u| !visited.contains(u))
            .collect()
    }
}

// ---------- handlers ----------

fn plural(n: usize, one: &str, many: &str) -> String {
    if n == 1 {
        format!("1 {one}")
    } else {
        format!("{n} {many}")
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct BackendInfo {
    pub ip: String,
    pub port: Option<u16>,
    pub load: u64,
    pub exact: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct ModelBackends {
    pub id: String,
    pub backends: Vec<BackendInfo>,
}

#[derive(Debug, Clone, Serialize)]
pub struct BackendsView {
    pub models: Vec<ModelBackends>,
}

#[derive(Serialize)]
struct ModelObject {
    id: String,
    object: &'static str,
    created: u64,
    owned_by: String,
}

#[derive(Serialize)]
struct ModelsResponse {
    object: &'static str,
    data: Vec<ModelObject>,
}

pub async fn list_models(State(hive): State<Hive>) -> impl IntoResponse {
    let endpoints = hive.snapshot().await;
    let mut seen = HashSet::new();
    let mut data = Vec::new();
    for ep in &endpoints {
        let owner = ep.name.clone();
        for m in &ep.models {
            if seen.insert(m.clone()) {
                data.push(ModelObject {
                    id: m.clone(),
                    object: "model",
                    created: 0,
                    owned_by: owner.clone(),
                });
            }
        }
    }
    data.sort_by(|a, b| a.id.cmp(&b.id));
    Json(ModelsResponse {
        object: "list",
        data,
    })
}

#[derive(Debug, Deserialize, Default)]
pub struct ModelQuery {
    /// `?model=` override. Body `model` field takes precedence if both set?
    /// We let the query param win when present (explicit routing request).
    pub model: Option<String>,
}

fn resolve_model(query: &ModelQuery, body: &Value) -> Option<String> {
    if let Some(m) = &query.model {
        if !m.is_empty() {
            return Some(m.clone());
        }
    }
    body.get("model")?.as_str().map(|s| s.to_string())
}

fn json_error(status: StatusCode, message: String) -> Response {
    let payload = serde_json::json!({
        "error": { "message": message, "type": "hive_error" }
    });
    (status, Json(payload)).into_response()
}

/// Request header carrying the comma-separated ids (`base_url`s) of hives
/// that already forwarded this request. A hive never forwards to a backend
/// already on the path — this is what stops hive-of-hives ping-pong
/// (A strictly prefers B while B strictly prefers A) from looping forever.
pub const VIA_HEADER: &str = "x-hivllm-via";

/// Ordered, deduplicated Via path from incoming headers.
fn parse_via(headers: &HeaderMap) -> Vec<String> {
    let mut path = Vec::new();
    if let Some(v) = headers.get(VIA_HEADER).and_then(|v| v.to_str().ok()) {
        for part in v.split(',').map(str::trim).filter(|p| !p.is_empty()) {
            if !path.contains(&part.to_string()) {
                path.push(part.to_string());
            }
        }
    }
    path
}

pub async fn chat_completions(
    State(hive): State<Hive>,
    Query(query): Query<ModelQuery>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    proxy_by_model(hive, query, headers, body, "chat", "/v1/chat/completions").await
}

pub async fn completions(
    State(hive): State<Hive>,
    Query(query): Query<ModelQuery>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    proxy_by_model(hive, query, headers, body, "completions", "/v1/completions").await
}

pub async fn embeddings(
    State(hive): State<Hive>,
    Query(query): Query<ModelQuery>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    proxy_by_model(hive, query, headers, body, "embeddings", "/v1/embeddings").await
}

/// One proxied-query outcome, assembled at each exit of [`proxy_by_model`].
struct Outcome {
    model: String,
    upstream: Option<String>,
    stream: bool,
    status: StatusCode,
    usage: Option<Value>,
    error: Option<String>,
    request: Value,
    response: Option<LoggedResponse>,
}

/// Record one query outcome. No-op unless a sink is configured.
async fn log_query(hive: &Hive, route: &'static str, o: Outcome, start: Instant) {
    hive
        .log(LogEntry {
            ts: Utc::now(),
            route,
            model: o.model,
            upstream: o.upstream,
            stream: o.stream,
            status: o.status.as_u16(),
            latency_ms: start.elapsed().as_millis() as u64,
            usage: o.usage,
            error: o.error,
            request: o.request,
            response: o.response,
        })
        .await;
}

async fn proxy_by_model(
    hive: Hive,
    query: ModelQuery,
    headers: HeaderMap,
    body: Bytes,
    route: &'static str,
    upstream_path: &str,
) -> Response {
    let start = Instant::now();
    // Parse body as JSON value (keep raw for forwarding).
    let mut value: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(_) => {
            let msg = "invalid JSON body".to_string();
            log_query(
                &hive,
                route,
                Outcome {
                    model: String::new(),
                    upstream: None,
                    stream: false,
                    status: StatusCode::BAD_REQUEST,
                    usage: None,
                    error: Some(msg.clone()),
                    request: Value::Null,
                    response: None,
                },
                start,
            )
            .await;
            return json_error(StatusCode::BAD_REQUEST, msg);
        }
    };

    let Some(model) = resolve_model(&query, &value) else {
        let msg = "missing `model`: set JSON body {\"model\": \"...\"} or ?model=...".to_string();
        log_query(
            &hive,
            route,
            Outcome {
                model: String::new(),
                upstream: None,
                stream: false,
                status: StatusCode::BAD_REQUEST,
                usage: None,
                error: Some(msg.clone()),
                request: value,
                response: None,
            },
            start,
        )
        .await;
        return json_error(StatusCode::BAD_REQUEST, msg);
    };

    // If ?model= was used, normalize the forwarded body so upstreams see it too.
    if let Some(obj) = value.as_object_mut() {
        obj.insert("model".to_string(), Value::String(model.clone()));
    }

    let via_path = parse_via(&headers);
    let visited: HashSet<String> = via_path.iter().cloned().collect();
    let candidates = hive.candidates_excluding(&model, &visited).await;
    if candidates.is_empty() {
        // Nothing left to try: either the model is unknown (404), or every
        // backend is already on the Via path — a hive-of-hives loop (502).
        if !hive.candidates(&model).await.is_empty() {
            let msg = format!(
                "loop detected: every backend for model `{model}` already visited ({})",
                via_path.join(", ")
            );
            log_query(
                &hive,
                route,
                Outcome {
                    model,
                    upstream: None,
                    stream: false,
                    status: StatusCode::BAD_GATEWAY,
                    usage: None,
                    error: Some(msg.clone()),
                    request: value,
                    response: None,
                },
                start,
            )
            .await;
            return json_error(StatusCode::BAD_GATEWAY, msg);
        }
        let available: Vec<String> = {
            let eps = hive.snapshot().await;
            let mut set = HashSet::new();
            for ep in &eps {
                for m in &ep.models {
                    set.insert(m.clone());
                }
            }
            let mut v: Vec<String> = set.into_iter().collect();
            v.sort();
            v
        };
        let msg = format!("model `{model}` not found in hive. Available: {available:?}");
        log_query(
            &hive,
            route,
            Outcome {
                model,
                upstream: None,
                stream: false,
                status: StatusCode::NOT_FOUND,
                usage: None,
                error: Some(msg.clone()),
                request: value,
                response: None,
            },
            start,
        )
        .await;
        return json_error(StatusCode::NOT_FOUND, msg);
    }

    let stream_mode = value.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);
    let fwd_body = serde_json::to_vec(&value).unwrap_or_default();

    // Passthrough Authorization if caller provided one.
    let auth: Option<String> = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    // Outgoing Via path: what we received, plus ourselves.
    let mut fwd_path = via_path;
    let me = hive.own_id();
    if !fwd_path.contains(&me) {
        fwd_path.push(me);
    }
    let fwd_via = fwd_path.join(", ");

    // Try candidates in load order; an unreachable backend fails over to
    // the next one instead of failing the request.
    let mut last_error = String::new();
    for upstream in &candidates {
        let url = format!("{}{}", upstream.trim_end_matches('/'), upstream_path);
        let mut req = hive
            .client
            .post(&url)
            .header("Content-Type", "application/json")
            .header(VIA_HEADER, fwd_via.clone())
            .body(fwd_body.clone());
        if let Some(a) = &auth {
            req = req.header(reqwest::header::AUTHORIZATION, a.clone());
        }

        let resp = match req.send().await {
            Ok(r) => r,
            Err(e) => {
                tracing::warn!(%url, error = %e, "upstream failed, trying next hive member");
                last_error = e.to_string();
                continue;
            }
        };
        hive.inflight_inc(upstream).await;

        let status = StatusCode::from_u16(resp.status().as_u16())
            .unwrap_or(StatusCode::BAD_GATEWAY);

        if stream_mode {
            // Tee the SSE bytes: client streams untouched while we rebuild the
            // response for the log. The entry is emitted when the stream ends,
            // so logged latency covers the full generation.
            let acc = Arc::new(std::sync::Mutex::new(StreamAcc::default()));
            let (tx, rx) = oneshot::channel::<StreamSummary>();
            let body = Body::from_stream(TeeStream::new(resp.bytes_stream(), acc.clone(), tx));
            let hive2 = hive.clone();
            let (model, value, upstream) = (model.clone(), value.clone(), upstream.clone());
            tokio::spawn(async move {
                let (summary, error) = match rx.await {
                    Ok(s) => (
                        s,
                        (!status.is_success()).then(|| format!("upstream status {status}")),
                    ),
                    Err(_) => (
                        acc.lock().map(|a| a.summary()).unwrap_or_default(),
                        Some("stream ended before completion".to_string()),
                    ),
                };
                log_query(
                    &hive2,
                    route,
                    Outcome {
                        model,
                        upstream: Some(upstream.clone()),
                        stream: true,
                        status,
                        usage: None,
                        error,
                        request: value,
                        response: Some(summary.into_response()),
                    },
                    start,
                )
                .await;
                hive2.inflight_dec(&upstream).await;
            });
            let mut resp_headers = HeaderMap::new();
            resp_headers.insert("Content-Type", "text/event-stream".parse().unwrap());
            resp_headers.insert("Cache-Control", "no-cache".parse().unwrap());
            return (status, resp_headers, body).into_response();
        }

        let bytes = resp.bytes().await.unwrap_or_default();
        let parsed = serde_json::from_slice::<Value>(&bytes).ok();
        let usage = parsed
            .as_ref()
            .and_then(|v| v.get("usage").cloned());
        let response = parsed.as_ref().map(extract_response);
        let err = (!status.is_success()).then(|| format!("upstream status {status}"));
        log_query(
            &hive,
            route,
            Outcome {
                model: model.clone(),
                upstream: Some(upstream.clone()),
                stream: false,
                status,
                usage,
                error: err,
                request: value.clone(),
                response,
            },
            start,
        )
        .await;
        hive.inflight_dec(upstream).await;
        let mut resp_headers = HeaderMap::new();
        resp_headers.insert("Content-Type", "application/json".parse().unwrap());
        return (status, resp_headers, body_from_bytes(bytes)).into_response();
    }

    // Every candidate failed to accept the request.
    let msg = format!(
        "all {} hive member(s) unreachable for model `{model}`: {last_error}",
        candidates.len()
    );
    log_query(
        &hive,
        route,
        Outcome {
            model,
            upstream: None,
            stream: stream_mode,
            status: StatusCode::BAD_GATEWAY,
            usage: None,
            error: Some(msg.clone()),
            request: value,
            response: None,
        },
        start,
    )
    .await;
    json_error(StatusCode::BAD_GATEWAY, msg)
}

fn body_from_bytes(bytes: bytes::Bytes) -> Body {
    Body::from(bytes)
}

pub async fn list_endpoints(State(hive): State<Hive>) -> impl IntoResponse {
    Json(hive.snapshot().await)
}

/// Per-model backends with the loads the balancer routes on (JSON form
/// of the stdout hive view). Built for the HiveChat dashboard.
pub async fn list_backends(State(hive): State<Hive>) -> impl IntoResponse {
    Json(hive.backends_view().await)
}

#[derive(Debug, Deserialize, Default)]
pub struct QueriesQuery {
    pub limit: Option<usize>,
}

/// Last query-log entries, newest first (empty without a file sink).
/// Built for the HiveChat queries view.
pub async fn list_queries(
    State(hive): State<Hive>,
    Query(query): Query<QueriesQuery>,
) -> impl IntoResponse {
    let limit = query.limit.unwrap_or(100).min(1000);
    Json(hive.recent_queries(limit).await)
}

pub async fn health() -> impl IntoResponse {
    Json(serde_json::json!({ "status": "ok", "hive": "sticky 🐝" }))
}

/// vLLM-compatible load endpoint, optionally per model: aggregate pressure
/// of this hive (or of its backends serving `?model=`) so upstream hives
/// route on real numbers. The `"hivllm": true` marker tells upstream hives
/// this is a per-model aggregate, not a box-level total.
#[derive(Debug, Deserialize, Default)]
pub struct LoadQuery {
    pub model: Option<String>,
}

pub async fn server_load(
    State(hive): State<Hive>,
    Query(query): Query<LoadQuery>,
) -> impl IntoResponse {
    let (n, exact) = hive.aggregate_load(query.model.as_deref()).await;
    Json(serde_json::json!({
        "server_load": n,
        "hivllm": { "exact": exact },
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        routing::{get, post},
        Router,
    };

    fn ep(id: &str, port: u16, models: &[&str]) -> DiscoveredEndpoint {
        DiscoveredEndpoint {
            id: id.to_string(),
            name: "test".into(),
            base_url: format!("http://127.0.0.1:{port}"),
            models: models.iter().map(|s| s.to_string()).collect(),
            source: "test".into(),
        }
    }

    async fn hive_with(
        endpoints: Vec<DiscoveredEndpoint>,
        loads: Vec<((&str, &str), Load)>,
    ) -> Hive {
        let hive = Hive::new(0);
        *hive.endpoints.write().await = endpoints;
        *hive.loads.write().await = loads
            .into_iter()
            .map(|((id, model), l)| ((id.to_string(), model.to_string()), l))
            .collect();
        hive
    }

    fn urls(hive_ports: &[u16]) -> Vec<String> {
        hive_ports
            .iter()
            .map(|p| format!("http://127.0.0.1:{p}"))
            .collect()
    }

    #[tokio::test]
    async fn routes_to_least_loaded_backend() {
        let hive = hive_with(
            vec![
                ep("a", 9001, &["m"]),
                ep("b", 9002, &["m"]),
                ep("c", 9003, &["m"]),
            ],
            vec![
                (("a", "m"), Load::Known(5)),
                (("b", "m"), Load::Known(1)),
                (("c", "m"), Load::Known(3)),
            ],
        )
        .await;
        assert_eq!(hive.candidates("m").await, urls(&[9002, 9003, 9001]));
        // Stable across calls when loads don't change.
        assert_eq!(hive.candidates("m").await, urls(&[9002, 9003, 9001]));
    }

    #[tokio::test]
    async fn equal_loads_break_ties_round_robin() {
        let hive = hive_with(
            vec![ep("a", 9001, &["m"]), ep("b", 9002, &["m"])],
            vec![(("a", "m"), Load::Known(2)), (("b", "m"), Load::Known(2))],
        )
        .await;
        assert_eq!(hive.candidates("m").await, urls(&[9001, 9002]));
        assert_eq!(hive.candidates("m").await, urls(&[9002, 9001]));
        assert_eq!(hive.candidates("m").await, urls(&[9001, 9002]));
    }

    #[tokio::test]
    async fn unknown_loads_fall_back_to_round_robin() {
        let hive = hive_with(
            vec![
                ep("a", 9001, &["m"]),
                ep("b", 9002, &["m"]),
                ep("c", 9003, &["m"]),
            ],
            vec![],
        )
        .await;
        assert_eq!(hive.candidates("m").await, urls(&[9001, 9002, 9003]));
        assert_eq!(hive.candidates("m").await, urls(&[9002, 9003, 9001]));
        assert_eq!(hive.candidates("m").await, urls(&[9003, 9001, 9002]));
    }

    #[tokio::test]
    async fn unknown_idle_joins_zero_load_tie() {
        let hive = hive_with(
            vec![
                ep("busy", 9001, &["m"]),
                ep("mystery", 9002, &["m"]),
                ep("idle", 9003, &["m"]),
            ],
            vec![(("busy", "m"), Load::Known(50)), (("idle", "m"), Load::Known(0))],
        )
        .await;
        // Same effective load (0) round-robins in discovery order; busy last.
        assert_eq!(hive.candidates("m").await, urls(&[9002, 9003, 9001]));
    }

    #[tokio::test]
    async fn hive_observed_load_counts_when_server_reports_zero() {
        let hive = hive_with(
            vec![ep("stuck", 9001, &["m"]), ep("idle", 9002, &["m"])],
            vec![(("stuck", "m"), Load::Known(0)), (("idle", "m"), Load::Known(0))],
        )
        .await;
        // Simulate 4 requests the hive is serving on "stuck" right now:
        // a `/load` frozen at 0 must not outrank a truly idle backend.
        for _ in 0..4 {
            hive.inflight_inc("http://127.0.0.1:9001").await;
        }
        assert_eq!(hive.candidates("m").await, urls(&[9002, 9001]));
    }

    #[tokio::test]
    async fn unknown_model_has_no_candidates() {
        let hive = hive_with(vec![ep("a", 9001, &["m"])], vec![(("a", "m"), Load::Known(0))]).await;
        assert!(hive.candidates("nope").await.is_empty());
    }

    #[tokio::test]
    async fn visited_backends_are_excluded() {
        let hive = hive_with(
            vec![ep("a", 9001, &["m"]), ep("b", 9002, &["m"])],
            vec![(("a", "m"), Load::Known(0)), (("b", "m"), Load::Known(0))],
        )
        .await;
        let visited: HashSet<String> =
            ["http://127.0.0.1:9001".to_string()].into_iter().collect();
        assert_eq!(
            hive.candidates_excluding("m", &visited).await,
            urls(&[9002])
        );
        let all: HashSet<String> = ["http://127.0.0.1:9001".to_string(), "http://127.0.0.1:9002".to_string()]
            .into_iter()
            .collect();
        assert!(hive.candidates_excluding("m", &all).await.is_empty());
    }

    #[test]
    fn via_header_parses_ordered_and_deduped() {
        let mut headers = HeaderMap::new();
        headers.insert(VIA_HEADER, "http://127.0.0.1:8335, http://127.0.0.1:8080 ,http://127.0.0.1:8335".parse().unwrap());
        assert_eq!(
            parse_via(&headers),
            vec![
                "http://127.0.0.1:8335".to_string(),
                "http://127.0.0.1:8080".to_string()
            ]
        );
        assert!(parse_via(&HeaderMap::new()).is_empty());
    }

    #[tokio::test]
    async fn view_orders_backends_by_decreasing_load() {
        let hive = hive_with(
            vec![
                ep("idle", 9001, &["m"]),
                ep("busy", 9002, &["m"]),
                ep("mystery", 9003, &["m"]),
            ],
            vec![(("idle", "m"), Load::Known(0)), (("busy", "m"), Load::Known(7))],
        )
        .await;
        // Simulate 2 hive-observed in-flight requests on the unknown backend.
        hive.inflight_inc("http://127.0.0.1:9003").await;
        hive.inflight_inc("http://127.0.0.1:9003").await;
        let view = hive.render_view().await;
        let busy = view.find("127.0.0.1:9002 load=7").expect("busy row");
        let approx = view.find("127.0.0.1:9003 load=~2").expect("approx row");
        let idle = view.find("127.0.0.1:9001 load=~0").expect("idle row");
        assert!(busy < approx && approx < idle, "{view}");
    }

    #[tokio::test]
    async fn view_handles_empty_hive() {
        let hive = Hive::new(0);
        assert_eq!(
            hive.render_view().await,
            "🐝 hive: 0 models, 0 backends"
        );
    }

    #[tokio::test]
    async fn backends_view_mirrors_stdout_rows() {
        let hive = hive_with(
            vec![
                ep("busy", 9001, &["m"]),
                ep("idle", 9002, &["m"]),
            ],
            vec![(("busy", "m"), Load::Known(7))],
        )
        .await;
        hive.inflight_inc("http://127.0.0.1:9002").await;
        hive.inflight_inc("http://127.0.0.1:9002").await;
        let view = hive.backends_view().await;
        assert_eq!(view.models.len(), 1);
        assert_eq!(view.models[0].id, "m");
        let rows = &view.models[0].backends;
        assert_eq!(rows.len(), 2);
        // Decreasing load: exact 7 first, approx ~2 second.
        assert_eq!(rows[0].ip, "127.0.0.1");
        assert_eq!(rows[0].port, Some(9001));
        assert_eq!((rows[0].load, rows[0].exact), (7, true));
        assert_eq!((rows[1].load, rows[1].exact), (2, false));
        // Same data the stdout view renders.
        let text = hive.render_view().await;
        assert!(text.contains("127.0.0.1:9001 load=7"));
        assert!(text.contains("127.0.0.1:9002 load=~2"));
    }

    #[tokio::test]
    async fn recent_queries_reads_newest_first() {
        use crate::logging::{JsonLinesSink, LogEntry, RequestLogger};
        use chrono::Utc;
        let path = std::env::temp_dir().join("hivllm-test-recent.jsonl");
        let _ = std::fs::remove_file(&path);
        let sink = JsonLinesSink::open(path.to_str().unwrap()).await.unwrap();
        let logger = RequestLogger::new().with_sink(std::sync::Arc::new(sink));
        for i in 0..3 {
            logger
                .log(LogEntry {
                    ts: Utc::now(),
                    route: "chat",
                    model: format!("m{i}"),
                    upstream: None,
                    stream: false,
                    status: 200,
                    latency_ms: i,
                    usage: None,
                    error: None,
                    request: serde_json::json!({}),
                    response: None,
                })
                .await;
        }
        let hive = Hive::new(0).with_logger(logger);
        let entries = hive.recent_queries(2).await;
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0]["model"], "m2");
        assert_eq!(entries[1]["model"], "m1");
        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn recent_queries_empty_without_file_sink() {
        assert!(Hive::new(0).recent_queries(10).await.is_empty());
    }

    #[tokio::test]
    async fn aggregate_load_is_the_median() {
        let hive = hive_with(
            vec![
                ep("a", 9001, &["m"]),
                ep("b", 9002, &["m"]),
                ep("c", 9003, &["m"]),
            ],
            vec![
                (("a", "m"), Load::Known(10)),
                (("b", "m"), Load::Known(30)),
                (("c", "m"), Load::Known(32)),
            ],
        )
        .await;
        assert_eq!(hive.aggregate_load(None).await, (30, true));
    }

    #[tokio::test]
    async fn aggregate_load_empty_hive_is_zero() {
        assert_eq!(Hive::new(0).aggregate_load(None).await, (0, false));
    }

    #[tokio::test]
    async fn aggregate_load_even_counts_average() {
        let hive = hive_with(
            vec![ep("a", 9001, &["m"]), ep("b", 9002, &["m"])],
            vec![(("a", "m"), Load::Known(30)), (("b", "m"), Load::Known(32))],
        )
        .await;
        assert_eq!(hive.aggregate_load(None).await, (31, true));
    }

    #[tokio::test]
    async fn aggregate_load_exact_outvotes_approximations() {
        // The user's case: one saturated backend (32) next to idle and
        // untracked ones — the pressure must stay visible, not median
        // itself away into `median(0, 0, 32) == 0`.
        let hive = hive_with(
            vec![
                ep("idle", 9001, &["m"]),
                ep("hot", 9002, &["m"]),
                ep("mystery", 9003, &["m"]),
            ],
            vec![(("idle", "m"), Load::Known(0)), (("hot", "m"), Load::Known(32))],
        )
        .await;
        assert_eq!(hive.aggregate_load(None).await, (32, true));
    }

    #[tokio::test]
    async fn aggregate_load_falls_back_to_approximations() {
        let hive = hive_with(
            vec![ep("a", 9001, &["m"]), ep("b", 9002, &["m"])],
            vec![],
        )
        .await;
        hive.inflight_inc("http://127.0.0.1:9002").await;
        hive.inflight_inc("http://127.0.0.1:9002").await;
        // No exact loads: median of approximations (0, 2) → 1.
        assert_eq!(hive.aggregate_load(None).await, (1, false));
    }

    #[tokio::test]
    async fn aggregate_load_scopes_to_one_model() {
        // "hot" is pressured on m2 only; "cold" serves m1 while idle.
        // ?model=m2 must report hot's 12, not an average smeared with m1.
        let hive = hive_with(
            vec![
                ep("hot", 9001, &["m1", "m2"]),
                ep("cold", 9002, &["m1"]),
            ],
            vec![
                ((("hot", "m1")), Load::Known(4)),
                ((("hot", "m2")), Load::Known(12)),
                ((("cold", "m1")), Load::Known(0)),
            ],
        )
        .await;
        assert_eq!(hive.aggregate_load(Some("m2")).await, (12, true));
        // m1: exact {4} outvotes cold's unverified zero.
        assert_eq!(hive.aggregate_load(Some("m1")).await, (4, true));
        assert_eq!(hive.aggregate_load(Some("nope")).await, (0, false));
    }

    #[tokio::test]
    async fn load_endpoint_accepts_model_param() {
        let hive = hive_with(
            vec![ep("hot", 9001, &["m1", "m2"])],
            vec![((("hot", "m2")), Load::Known(12))],
        )
        .await;
        let resp = server_load(
            State(hive),
            Query(LoadQuery {
                model: Some("m2".to_string()),
            }),
        )
        .await
        .into_response();
        let body =
            axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        let v: Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(v["server_load"], 12);
        assert_eq!(v["hivllm"]["exact"], true);
    }

    #[tokio::test]
    async fn bare_load_is_skipped_for_multi_model_backends() {
        // Mock old hive: global /load with no marker and no /metrics.
        // Its number must not smear across models.
        let mock = Router::new().route(
            "/load",
            get(|| async { Json(serde_json::json!({ "server_load": 99 })) }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move { axum::serve(listener, mock).await.unwrap() });
        fn member(id: &str, base_url: &str, models: &[&str]) -> DiscoveredEndpoint {
            DiscoveredEndpoint {
                id: id.to_string(),
                name: "test".into(),
                base_url: base_url.to_string(),
                models: models.iter().map(|s| s.to_string()).collect(),
                source: "test".into(),
            }
        }

        let hive = Hive::new(0);
        *hive.endpoints.write().await = vec![
            member("single", &url, &["m1"]),
            member("multi", &url, &["m1", "m2"]),
        ];
        hive.refresh_load().await;
        let loads = hive.loads.read().await;
        // Single-model backend: bare /load is that model's load.
        assert_eq!(
            loads.get(&("single".to_string(), "m1".to_string())),
            Some(&Load::Known(99))
        );
        // Multi-model backend: global refused, stays usable-but-unknown.
        assert_eq!(
            loads.get(&("multi".to_string(), "m1".to_string())),
            Some(&Load::Unknown)
        );
        assert_eq!(
            loads.get(&("multi".to_string(), "m2".to_string())),
            Some(&Load::Unknown)
        );
    }

    #[tokio::test]
    async fn unhealthy_backend_does_not_block_healthy_ones() {
        // Live mock upstream serving a canned completion.
        let mock = Router::new().route(
            "/v1/chat/completions",
            post(|| async {
                Json(serde_json::json!({
                    "id": "mock-1",
                    "object": "chat.completion",
                    "choices": [{"message": {"role": "assistant", "content": "mock"}}],
                    "usage": {"total_tokens": 1}
                }))
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .unwrap();
        let live_port = listener.local_addr().unwrap().port();
        tokio::spawn(async move { axum::serve(listener, mock).await.unwrap() });

        // Dead backend: reserve a port, then close it.
        let tmp = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .unwrap();
        let dead_port = tmp.local_addr().unwrap().port();
        drop(tmp);

        let hive = hive_with(
            vec![ep("dead", dead_port, &["m"]), ep("live", live_port, &["m"])],
            // Dead backend *looks* idle, so it is tried first and must be skipped.
            vec![(("dead", "m"), Load::Known(0)), (("live", "m"), Load::Known(9))],
        )
        .await;

        let app = Router::new()
            .route("/v1/chat/completions", post(chat_completions))
            .with_state(hive);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .unwrap();
        let hive_url = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let resp = reqwest::Client::new()
            .post(format!("{hive_url}/v1/chat/completions"))
            .json(&serde_json::json!({"model": "m", "messages": []}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let body: Value = resp.json().await.unwrap();
        assert_eq!(body["id"], "mock-1");
    }

    #[tokio::test]
    async fn hive_ping_pong_terminates_with_loop_detected() {
        // Two hives that strictly prefer each other: without Via filtering
        // this would bounce forever. A forwards to B with Via:[A], B finds
        // only visited candidates and answers 502 — the client gets a fast
        // error, not a hang.
        async fn serve_hive(
            listener: tokio::net::TcpListener,
            hive: Hive,
        ) -> String {
            let url = format!("http://{}", listener.local_addr().unwrap());
            let app = Router::new()
                .route("/v1/chat/completions", post(chat_completions))
                .with_state(hive);
            tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
            url
        }
        fn member(id: &str, base_url: &str) -> DiscoveredEndpoint {
            DiscoveredEndpoint {
                id: id.to_string(),
                name: "hive".into(),
                base_url: base_url.to_string(),
                models: vec!["m".to_string()],
                source: "test".into(),
            }
        }

        let la = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url_a = format!("http://{}", la.local_addr().unwrap());
        let lb = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url_b = format!("http://{}", lb.local_addr().unwrap());

        let ha = Hive::new(0).with_own_id(url_a.clone());
        *ha.endpoints.write().await = vec![member("b", &url_b)];
        *ha.loads.write().await =
            HashMap::from([(("b".to_string(), "m".to_string()), Load::Known(0))]);
        serve_hive(la, ha).await;

        let hb = Hive::new(0).with_own_id(url_b.clone());
        *hb.endpoints.write().await = vec![member("a", &url_a)];
        *hb.loads.write().await =
            HashMap::from([(("a".to_string(), "m".to_string()), Load::Known(0))]);
        serve_hive(lb, hb).await;

        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(10))
            .build()
            .unwrap();
        let resp = client
            .post(format!("{url_a}/v1/chat/completions"))
            .json(&serde_json::json!({"model": "m", "messages": []}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 502);
        let body: Value = resp.json().await.unwrap();
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap_or("")
                .contains("loop detected"),
            "{body}"
        );
    }
}
