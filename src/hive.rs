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
use crate::load::{probe_backend, Load, LoadProbe, VllmLoadProbe};
use crate::logging::{
    extract_response, truncate_value, LogEntry, LoggedResponse, RequestLogger, StreamAcc,
    StreamSummary, TeeStream, Truncate,
};

#[derive(Clone)]
pub struct Hive {
    pub client: reqwest::Client,
    /// Port the hive itself listens on — always excluded from discovery.
    own_port: u16,
    logger: RequestLogger,
    log_truncate: Truncate,
    log_max_chars: usize,
    /// Provider load probes (vLLM `/load` today, more later).
    probes: Vec<Arc<dyn LoadProbe>>,
    /// Endpoint id → last polled load. Rebuilt by every [`Hive::refresh_load`].
    loads: Arc<RwLock<HashMap<String, Load>>>,
    endpoints: Arc<RwLock<Vec<DiscoveredEndpoint>>>,
    /// (model, load-bucket) -> next index. Buckets rotate round-robin:
    /// each tie-group of equal load, plus the unknown-load group, gets
    /// its own counter. All-unknown degrades to plain round-robin.
    rr: Arc<Mutex<HashMap<(String, Option<u64>), usize>>>,
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
            logger: RequestLogger::new(),
            log_truncate: Truncate::None,
            log_max_chars: 2000,
            probes: vec![Arc::new(VllmLoadProbe)],
            loads: Arc::new(RwLock::new(HashMap::new())),
            endpoints: Arc::new(RwLock::new(Vec::new())),
            rr: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn with_logger(mut self, logger: RequestLogger) -> Self {
        self.logger = logger;
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
        tracing::info!(count = found.len(), "hive discovery refresh");
        for ep in &found {
            tracing::info!(
                base_url = %ep.base_url,
                models = ?ep.models,
                source = %ep.source,
                "hive member"
            );
        }
        *self.endpoints.write().await = found;
    }

    pub async fn snapshot(&self) -> Vec<DiscoveredEndpoint> {
        self.endpoints.read().await.clone()
    }

    /// Poll load probes for every known endpoint (concurrently) and
    /// replace the load map. Backends that fail probing stay `Unknown` —
    /// usable, routed round-robin. Run on a short interval; never per-request.
    pub async fn refresh_load(&self) {
        let endpoints = self.endpoints.read().await.clone();
        let probed: HashMap<String, Load> = stream::iter(endpoints)
            .map(|ep| {
                let probes = self.probes.clone();
                let client = self.client.clone();
                async move {
                    let load = probe_backend(&probes, &client, &ep.base_url).await;
                    (ep.id.clone(), load)
                }
            })
            .buffer_unordered(16)
            .collect()
            .await;
        let known = probed.values().filter(|l| matches!(l, Load::Known(_))).count();
        tracing::debug!(known, total = probed.len(), "hive load refresh");
        *self.loads.write().await = probed;
    }

    /// Order backends serving `model`: lowest known load first, then
    /// unknown-load backends. Equal loads (and the all-unknown case) rotate
    /// round-robin. The proxy tries candidates in order and fails over to
    /// the next one when a backend is unreachable.
    pub async fn candidates(&self, model: &str) -> Vec<String> {
        let endpoints = self.endpoints.read().await;
        let loads = self.loads.read().await;
        let mut known: Vec<(String, u64)> = Vec::new();
        let mut unknown: Vec<String> = Vec::new();
        for ep in endpoints
            .iter()
            .filter(|e| e.models.iter().any(|m| m == model))
        {
            match loads.get(&ep.id).copied().unwrap_or(Load::Unknown) {
                Load::Known(n) => known.push((ep.base_url.clone(), n)),
                Load::Unknown => unknown.push(ep.base_url.clone()),
            }
        }
        drop(loads);
        drop(endpoints);
        known.sort_by_key(|(_, n)| *n); // stable: ties keep discovery order
        let mut rr = self.rr.lock().await;
        let mut out = Vec::with_capacity(known.len() + unknown.len());
        // Rotate each run of equal load independently.
        let mut i = 0;
        while i < known.len() {
            let load = known[i].1;
            let mut j = i + 1;
            while j < known.len() && known[j].1 == load {
                j += 1;
            }
            let group = &mut known[i..j];
            if group.len() > 1 {
                let counter = rr
                    .entry((model.to_string(), Some(load)))
                    .or_insert(0);
                let rot = *counter % group.len();
                *counter = counter.wrapping_add(1);
                group.rotate_left(rot);
            }
            out.extend(group.iter().map(|(u, _)| u.clone()));
            i = j;
        }
        if !unknown.is_empty() {
            let counter = rr.entry((model.to_string(), None)).or_insert(0);
            let rot = *counter % unknown.len();
            *counter = counter.wrapping_add(1);
            unknown.rotate_left(rot);
            out.extend(unknown);
        }
        out
    }
}

// ---------- handlers ----------

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

    let candidates = hive.candidates(&model).await;
    if candidates.is_empty() {
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

    // Try candidates in load order; an unreachable backend fails over to
    // the next one instead of failing the request.
    let mut last_error = String::new();
    for upstream in &candidates {
        let url = format!("{}{}", upstream.trim_end_matches('/'), upstream_path);
        let mut req = hive
            .client
            .post(&url)
            .header("Content-Type", "application/json")
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
                        upstream: Some(upstream),
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

pub async fn health() -> impl IntoResponse {
    Json(serde_json::json!({ "status": "ok", "hive": "sticky 🐝" }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{routing::post, Router};

    fn ep(id: &str, port: u16, models: &[&str]) -> DiscoveredEndpoint {
        DiscoveredEndpoint {
            id: id.to_string(),
            name: "test".into(),
            base_url: format!("http://127.0.0.1:{port}"),
            models: models.iter().map(|s| s.to_string()).collect(),
            source: "test".into(),
        }
    }

    async fn hive_with(endpoints: Vec<DiscoveredEndpoint>, loads: Vec<(&str, Load)>) -> Hive {
        let hive = Hive::new(0);
        *hive.endpoints.write().await = endpoints;
        *hive.loads.write().await = loads
            .into_iter()
            .map(|(id, l)| (id.to_string(), l))
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
                ("a", Load::Known(5)),
                ("b", Load::Known(1)),
                ("c", Load::Known(3)),
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
            vec![("a", Load::Known(2)), ("b", Load::Known(2))],
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
    async fn known_loads_come_before_unknown_ones() {
        let hive = hive_with(
            vec![
                ep("busy", 9001, &["m"]),
                ep("mystery", 9002, &["m"]),
                ep("idle", 9003, &["m"]),
            ],
            vec![("busy", Load::Known(50)), ("idle", Load::Known(0))],
        )
        .await;
        // Known loads sorted first; unknown stays usable at the end.
        assert_eq!(hive.candidates("m").await, urls(&[9003, 9001, 9002]));
    }

    #[tokio::test]
    async fn unknown_model_has_no_candidates() {
        let hive = hive_with(vec![ep("a", 9001, &["m"])], vec![("a", Load::Known(0))]).await;
        assert!(hive.candidates("nope").await.is_empty());
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
            vec![("dead", Load::Known(0)), ("live", Load::Known(9))],
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
}
