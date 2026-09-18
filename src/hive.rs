//! Unified hive: aggregated `/v1/models` + routed `/v1/chat/completions`
//! with round-robin load-balancing when several endpoints share a model name.

use axum::{
    body::Body,
    extract::{Query, State},
    http::{HeaderMap, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use bytes::Bytes;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
    time::Duration,
};
use tokio::sync::{Mutex, RwLock};

use crate::discovery::{discover, DiscoveredEndpoint};

#[derive(Clone)]
pub struct Hive {
    pub client: reqwest::Client,
    /// Port the hive itself listens on — always excluded from discovery.
    own_port: u16,
    endpoints: Arc<RwLock<Vec<DiscoveredEndpoint>>>,
    /// model name -> next index (round-robin)
    rr: Arc<Mutex<HashMap<String, usize>>>,
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
            endpoints: Arc::new(RwLock::new(Vec::new())),
            rr: Arc::new(Mutex::new(HashMap::new())),
        }
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

    /// Pick one upstream base_url offering `model`, round-robin across matches.
    pub async fn pick(&self, model: &str) -> Option<String> {
        let endpoints = self.endpoints.read().await;
        let matches: Vec<String> = endpoints
            .iter()
            .filter(|e| e.models.iter().any(|m| m == model))
            .map(|e| e.base_url.clone())
            .collect();
        if matches.is_empty() {
            return None;
        }
        let mut rr = self.rr.lock().await;
        let idx = rr.entry(model.to_string()).or_insert(0);
        let chosen = matches[*idx % matches.len()].clone();
        *idx = idx.wrapping_add(1);
        Some(chosen)
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
    proxy_by_model(hive, query, headers, body, "/v1/chat/completions").await
}

pub async fn completions(
    State(hive): State<Hive>,
    Query(query): Query<ModelQuery>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    proxy_by_model(hive, query, headers, body, "/v1/completions").await
}

pub async fn embeddings(
    State(hive): State<Hive>,
    Query(query): Query<ModelQuery>,
    headers: HeaderMap,
    body: Bytes,
) -> Response {
    proxy_by_model(hive, query, headers, body, "/v1/embeddings").await
}

async fn proxy_by_model(
    hive: Hive,
    query: ModelQuery,
    headers: HeaderMap,
    body: Bytes,
    upstream_path: &str,
) -> Response {
    // Parse body as JSON value (keep raw for forwarding).
    let mut value: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(_) => return json_error(StatusCode::BAD_REQUEST, "invalid JSON body".into()),
    };

    let Some(model) = resolve_model(&query, &value) else {
        return json_error(
            StatusCode::BAD_REQUEST,
            "missing `model`: set JSON body {\"model\": \"...\"} or ?model=...".into(),
        );
    };

    // If ?model= was used, normalize the forwarded body so upstreams see it too.
    if let Some(obj) = value.as_object_mut() {
        obj.insert("model".to_string(), Value::String(model.clone()));
    }

    let Some(upstream) = hive.pick(&model).await else {
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
        return json_error(
            StatusCode::NOT_FOUND,
            format!("model `{model}` not found in hive. Available: {available:?}"),
        );
    };

    let url = format!("{}{}", upstream.trim_end_matches('/'), upstream_path);
    let stream_mode = value.get("stream").and_then(|v| v.as_bool()).unwrap_or(false);

    let mut req = hive
        .client
        .post(&url)
        .header("Content-Type", "application/json")
        .body(serde_json::to_vec(&value).unwrap_or_default());

    // Passthrough Authorization if caller provided one.
    if let Some(auth) = headers.get(axum::http::header::AUTHORIZATION) {
        if let Ok(s) = auth.to_str() {
            req = req.header(reqwest::header::AUTHORIZATION, s.to_string());
        }
    }

    let resp = match req.send().await {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!(%url, error = %e, "upstream unreachable");
            return json_error(
                StatusCode::BAD_GATEWAY,
                format!("upstream {upstream} unreachable: {e}"),
            );
        }
    };

    let status = StatusCode::from_u16(resp.status().as_u16())
        .unwrap_or(StatusCode::BAD_GATEWAY);

    if stream_mode {
        // Passthrough SSE bytes to preserve OpenAI streaming.
        let stream = resp.bytes_stream();
        let body = Body::from_stream(stream);
        let mut resp_headers = HeaderMap::new();
        resp_headers.insert("Content-Type", "text/event-stream".parse().unwrap());
        resp_headers.insert("Cache-Control", "no-cache".parse().unwrap());
        (status, resp_headers, body).into_response()
    } else {
        let bytes = resp.bytes().await.unwrap_or_default();
        let mut resp_headers = HeaderMap::new();
        resp_headers.insert("Content-Type", "application/json".parse().unwrap());
        (status, resp_headers, body_from_bytes(bytes)).into_response()
    }
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
