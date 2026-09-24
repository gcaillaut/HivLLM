//! HivLLM discovery: find OpenAI-compatible endpoints on this machine.
//!
//! v1 scope (per project decisions):
//! - Local port scan: probe well-known ports + `ss` listening ports on 127.0.0.1
//!   for `GET /v1/models`.
//! - Process + Docker scan: map listening sockets to processes (`ss -tlnp`),
//!   list `docker ps` published ports, probe each for `/v1/models`.

use futures::{stream, StreamExt};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::process::Command;
use std::time::Duration;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveredEndpoint {
    /// Stable id, e.g. "http-127.0.0.1-11434"
    pub id: String,
    /// Human hint, e.g. "ollama" / process name / container port
    pub name: String,
    /// Base URL without trailing slash, e.g. "http://127.0.0.1:11434"
    pub base_url: String,
    /// Model ids reported by upstream `GET /v1/models`
    pub models: Vec<String>,
    /// Where we found it: "well-known-port" | "ss-listener" | "docker" | "manual"
    pub source: String,
    /// Set when the backend is itself a HivLLM hive: its instance id
    /// (`x-hivllm-id` response header), as it appears on Via paths.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub hive_id: Option<String>,
}

/// Response header every hive stamps with its instance id, so peers can
/// recognise it whatever address they reach it by.
pub const HIVE_ID_HEADER: &str = "x-hivllm-id";

/// A successful `/v1/models` probe.
#[derive(Debug, Clone)]
pub struct Probed {
    pub models: Vec<String>,
    /// Instance id when the backend is a hive.
    pub hive_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ModelList {
    #[serde(default)]
    data: Vec<ModelEntry>,
}

#[derive(Debug, Deserialize)]
struct ModelEntry {
    id: String,
}

/// Well-known OpenAI-compatible ports on localhost.
pub fn well_known_ports(extra: &[u16]) -> Vec<u16> {
    let mut ports = vec![
        11434, // Ollama
        1234,  // LM Studio
        8000,  // vLLM default
        8080,  // llama.cpp server default
        5000,  // text-generation-webui / misc
        5001, 8001, 11435, 11436, 6333, 6334,
    ];
    for p in extra {
        if !ports.contains(p) {
            ports.push(*p);
        }
    }
    ports
}

/// Models endpoint for a base URL. Bare server roots probe
/// `{base}/v1/models`; bases that already carry the OpenAI `/v1` prefix
/// (e.g. `http://host:8180/general-stage1/v1` behind a path-routing
/// gateway) probe `{base}/models` instead of doubling to `/v1/v1/models`.
pub fn models_url(base_url: &str) -> String {
    let b = base_url.trim_end_matches('/');
    if b.ends_with("/v1") {
        format!("{b}/models")
    } else {
        format!("{b}/v1/models")
    }
}

/// Server root for server-level routes (`/metrics`, `/load`): the base URL
/// with a trailing OpenAI `/v1` prefix stripped, since those routes live
/// next to `/v1`, not under it.
pub fn server_root(base_url: &str) -> String {
    let b = base_url.trim_end_matches('/');
    if let Some(stripped) = b.strip_suffix("/v1") {
        if stripped.is_empty() {
            b.to_string()
        } else {
            stripped.trim_end_matches('/').to_string()
        }
    } else {
        b.to_string()
    }
}

/// Join a base URL with an OpenAI API path (`/v1/chat/completions`, …)
/// without doubling a `/v1` prefix the base already carries.
pub fn join_upstream_path(base_url: &str, api_path: &str) -> String {
    let b = base_url.trim_end_matches('/');
    if b.ends_with("/v1") && api_path.starts_with("/v1") {
        format!("{b}{}", &api_path[3..])
    } else {
        format!("{b}{api_path}")
    }
}

/// Probe a single base URL for OpenAI compat. Returns model ids (and the
/// hive id, for hives) on success.
pub async fn probe_openai_endpoint(client: &reqwest::Client, base_url: &str) -> Option<Probed> {
    let url = models_url(base_url);
    let resp = client
        .get(&url)
        .timeout(Duration::from_secs(2))
        .send()
        .await
        .ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let hive_id = resp
        .headers()
        .get(HIVE_ID_HEADER)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let body: ModelList = resp.json().await.ok()?;
    let mut models: Vec<String> = body.data.into_iter().map(|m| m.id).collect();
    models.sort();
    models.dedup();
    Some(Probed { models, hive_id })
}

/// `ss -tln` listening TCP ports (Linux). Falls back to empty on error.
pub fn ss_listening_ports() -> Vec<u16> {
    let out = Command::new("ss")
        .args(["-tln"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
        .unwrap_or_default();
    parse_ss_ports(&out)
}

fn parse_ss_ports(ss_output: &str) -> Vec<u16> {
    let mut ports = HashSet::new();
    for token in ss_output.split_whitespace() {
        // tokens like 127.0.0.1:11434, [::]:8000, *:1234
        if let Some(idx) = token.rfind(':') {
            let maybe_port = &token[idx + 1..];
            // strip trailing chars like `]` — ports are numeric
            let digits: String = maybe_port.chars().take_while(|c| c.is_ascii_digit()).collect();
            if let Ok(p) = digits.parse::<u16>() {
                // only keep plausible user ports; skip 22/80/443 noise? keep all, probing filters.
                if p != 22 {
                    ports.insert(p);
                }
            }
        }
    }
    let mut v: Vec<u16> = ports.into_iter().collect();
    v.sort_unstable();
    v
}

/// `ss -tlnp` -> map port -> process name (best effort).
pub fn ss_port_processes() -> HashMap<u16, String> {
    let out = Command::new("ss")
        .args(["-tlnp"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
        .unwrap_or_default();
    let mut map = HashMap::new();
    for line in out.lines() {
        // crude: find :PORT then users:(("name",pid=...))
        let port: Option<u16> = line.split_whitespace().find_map(|tok| {
            tok.rfind(':').and_then(|i| {
                tok[i + 1..]
                    .chars()
                    .take_while(|c| c.is_ascii_digit())
                    .collect::<String>()
                    .parse()
                    .ok()
            })
        });
        if let Some(p) = port {
            if let Some(start) = line.find("users:((\"") {
                let rest = &line[start + 9..];
                if let Some(end) = rest.find('"') {
                    map.insert(p, rest[..end].to_string());
                }
            }
        }
    }
    map
}

/// `docker ps` published host ports (best effort, empty if docker missing).
pub fn docker_host_ports() -> Vec<u16> {
    let out = Command::new("docker")
        .args(["ps", "--format", "{{.Ports}}"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
        .unwrap_or_default();
    let mut ports = HashSet::new();
    for line in out.lines() {
        // e.g. "0.0.0.0:8000->8000/tcp, :::8000->8000/tcp" or "127.0.0.1:11434->11434/tcp"
        for part in line.split(',') {
            for tok in part.split_whitespace() {
                // take host side before "->"
                let left = tok.split("->").next().unwrap_or(tok);
                if let Some(idx) = left.rfind(':') {
                    if let Ok(p) = left[idx + 1..].parse::<u16>() {
                        ports.insert(p);
                    }
                }
            }
        }
    }
    let mut v: Vec<u16> = ports.into_iter().collect();
    v.sort_unstable();
    v
}

fn proc_hint(port: u16, proc_map: &HashMap<u16, String>) -> String {
    if let Some(name) = proc_map.get(&port) {
        return name.clone();
    }
    match port {
        11434 => "ollama".into(),
        1234 => "lmstudio".into(),
        8000 | 8001 => "vllm".into(),
        8080 => "llama.cpp".into(),
        _ => format!("port-{port}"),
    }
}

/// Full discovery pass. Probes well-known + ss + docker ports concurrently.
/// `exclude_ports` (e.g. the hive's own listen port) are never probed,
/// so the hive can't discover itself and loop requests back into itself.
pub async fn discover(
    client: &reqwest::Client,
    extra_ports: &[u16],
    exclude_ports: &[u16],
) -> Vec<DiscoveredEndpoint> {
    let mut candidates: Vec<u16> = well_known_ports(extra_ports);
    candidates.extend(ss_listening_ports());
    candidates.extend(docker_host_ports());
    candidates.sort_unstable();
    candidates.dedup();
    candidates.retain(|p| !exclude_ports.contains(p));

    let proc_map = ss_port_processes();
    let docker_ports: HashSet<u16> = docker_host_ports().into_iter().collect();

    let results: Vec<Option<DiscoveredEndpoint>> = stream::iter(candidates)
        .map(|port| {
            let proc_map = &proc_map;
            let docker_ports = &docker_ports;
            async move {
                let base_url = format!("http://127.0.0.1:{port}");
                let probed = probe_openai_endpoint(client, &base_url).await?;
                let source = if docker_ports.contains(&port) {
                    "docker"
                } else if well_known_ports(&[]).contains(&port) {
                    "well-known-port"
                } else {
                    "ss-listener"
                };
                Some(DiscoveredEndpoint {
                    id: format!("http-127.0.0.1-{port}"),
                    name: proc_hint(port, proc_map),
                    base_url,
                    models: probed.models,
                    source: source.to_string(),
                    hive_id: probed.hive_id,
                })
            }
        })
        .buffer_unordered(32)
        .collect()
        .await;

    let mut endpoints: Vec<DiscoveredEndpoint> = results.into_iter().flatten().collect();
    endpoints.sort_by(|a, b| a.base_url.cmp(&b.base_url));
    endpoints
}

/// Normalize an operator-provided backend URL: default to `http://` when no
/// scheme is given, strip trailing slashes.
pub fn normalize_base_url(raw: &str) -> String {
    let with_scheme = if raw.contains("://") {
        raw.to_string()
    } else {
        format!("http://{raw}")
    };
    with_scheme.trim_end_matches('/').to_string()
}

/// Host part of a base URL, for ids and display names.
fn url_host(base_url: &str) -> String {
    let no_scheme = base_url.split("://").last().unwrap_or(base_url);
    no_scheme
        .split('/')
        .next()
        .unwrap_or(no_scheme)
        .to_string()
}

/// Probe operator-provided backends (docker service names, remote hosts,
/// anything discovery can't see). Unreachable entries are skipped with a
/// warning — never added half-dead. Re-run every refresh so flapping
/// backends rejoin automatically.
pub async fn probe_static(
    client: &reqwest::Client,
    urls: &[String],
) -> Vec<DiscoveredEndpoint> {
    let probed: Vec<Option<DiscoveredEndpoint>> = stream::iter(urls.iter().cloned())
        .map(|raw| {
            let client = client.clone();
            async move {
                let base_url = normalize_base_url(&raw);
                let probed = match probe_openai_endpoint(&client, &base_url).await {
                    Some(p) => p,
                    None => {
                        tracing::warn!(%base_url, "static backend unreachable, skipping");
                        return None;
                    }
                };
                let host = url_host(&base_url);
                let id: String = format!(
                    "static-{}",
                    base_url
                        .chars()
                        .map(|c| if c.is_alphanumeric() { c } else { '-' })
                        .collect::<String>()
                );
                Some(DiscoveredEndpoint {
                    id,
                    name: host,
                    base_url,
                    models: probed.models,
                    source: "static".to_string(),
                    hive_id: probed.hive_id,
                })
            }
        })
        .buffer_unordered(16)
        .collect()
        .await;
    let mut endpoints: Vec<DiscoveredEndpoint> = probed.into_iter().flatten().collect();
    endpoints.sort_by(|a, b| a.base_url.cmp(&b.base_url));
    endpoints
}

/// Merge discovered + static endpoints, deduplicated by `base_url` with
/// unioned model lists. A static entry marks the merged source `static`
/// (operator intent wins the label).
pub fn merge_endpoints(
    discovered: Vec<DiscoveredEndpoint>,
    statik: Vec<DiscoveredEndpoint>,
) -> Vec<DiscoveredEndpoint> {
    let mut by_url: HashMap<String, DiscoveredEndpoint> = HashMap::new();
    for ep in discovered.into_iter().chain(statik) {
        match by_url.get_mut(&ep.base_url) {
            Some(existing) => {
                for m in &ep.models {
                    if !existing.models.contains(m) {
                        existing.models.push(m.clone());
                    }
                }
                existing.models.sort();
                if ep.source == "static" {
                    existing.source = "static".to_string();
                }
                if existing.hive_id.is_none() {
                    existing.hive_id = ep.hive_id;
                }
            }
            None => {
                by_url.insert(ep.base_url.clone(), ep);
            }
        }
    }
    let mut out: Vec<DiscoveredEndpoint> = by_url.into_values().collect();
    out.sort_by(|a, b| a.base_url.cmp(&b.base_url));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_ss_output() {
        let sample = "State Recv-Q Send-Q Local Address:Port Peer Address:Port\nLISTEN 0 128 127.0.0.1:11434 0.0.0.0:*\nLISTEN 0 128 [::]:8000 [::]:*";
        let mut ports = parse_ss_ports(sample);
        ports.sort_unstable();
        assert!(ports.contains(&11434));
        assert!(ports.contains(&8000));
    }

    #[test]
    fn well_known_contains_defaults() {
        let ports = well_known_ports(&[9999]);
        assert!(ports.contains(&11434));
        assert!(ports.contains(&1234));
        assert!(ports.contains(&9999));
    }

    #[test]
    fn normalizes_static_urls() {
        assert_eq!(
            normalize_base_url("http://llamacpp:8080/"),
            "http://llamacpp:8080"
        );
        assert_eq!(normalize_base_url("llamacpp:8080"), "http://llamacpp:8080");
        assert_eq!(normalize_base_url("https://h:1/a/"), "https://h:1/a");
    }

    #[test]
    fn merges_by_base_url_with_model_union() {
        fn ep(url: &str, models: &[&str], source: &str) -> DiscoveredEndpoint {
            DiscoveredEndpoint {
                id: format!("{source}-{url}"),
                name: "test".into(),
                base_url: url.into(),
                models: models.iter().map(|s| s.to_string()).collect(),
                source: source.into(),
                hive_id: None,
            }
        }
        let out = merge_endpoints(
            vec![ep("http://127.0.0.1:11434", &["a"], "ss-listener")],
            vec![
                ep("http://127.0.0.1:11434", &["b"], "static"),
                ep("http://llamacpp:8080", &["c"], "static"),
            ],
        );
        assert_eq!(out.len(), 2);
        let local = out.iter().find(|e| e.base_url.contains("11434")).unwrap();
        assert_eq!(local.models, vec!["a".to_string(), "b".to_string()]);
        assert_eq!(local.source, "static");
    }

    #[test]
    fn v1_suffixed_bases_dont_double_the_prefix() {
        assert_eq!(
            models_url("http://127.0.0.1:8180/general-stage1/v1"),
            "http://127.0.0.1:8180/general-stage1/v1/models"
        );
        assert_eq!(
            models_url("http://127.0.0.1:8180/general-stage1/v1/"),
            "http://127.0.0.1:8180/general-stage1/v1/models"
        );
        assert_eq!(
            models_url("http://127.0.0.1:8000"),
            "http://127.0.0.1:8000/v1/models"
        );
        assert_eq!(
            server_root("http://127.0.0.1:8180/general-stage1/v1"),
            "http://127.0.0.1:8180/general-stage1"
        );
        assert_eq!(
            server_root("http://127.0.0.1:8000"),
            "http://127.0.0.1:8000"
        );
        assert_eq!(
            join_upstream_path(
                "http://127.0.0.1:8180/general-stage1/v1",
                "/v1/chat/completions"
            ),
            "http://127.0.0.1:8180/general-stage1/v1/chat/completions"
        );
        assert_eq!(
            join_upstream_path("http://127.0.0.1:8000", "/v1/chat/completions"),
            "http://127.0.0.1:8000/v1/chat/completions"
        );
    }

    #[tokio::test]
    async fn static_probe_skips_unreachable() {
        // Live mock backend.
        let app = axum::Router::new().route(
            "/v1/models",
            axum::routing::get(|| async {
                axum::Json(serde_json::json!({
                    "object": "list",
                    "data": [{"id": "demo-model", "object": "model"}]
                }))
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        // Dead port: reserve then release.
        let tmp = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .unwrap();
        let dead = tmp.local_addr().unwrap().port();
        drop(tmp);

        let client = reqwest::Client::new();
        let found = probe_static(
            &client,
            &[
                format!("http://127.0.0.1:{port}"),
                format!("127.0.0.1:{dead}"),
            ],
        )
        .await;
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].models, vec!["demo-model".to_string()]);
        assert_eq!(found[0].source, "static");
    }

    #[tokio::test]
    async fn static_probe_supports_v1_prefixed_base() {
        // Gateway-style backend: OpenAI API lives under a path prefix.
        let app = axum::Router::new().route(
            "/general-stage1/v1/models",
            axum::routing::get(|| async {
                axum::Json(serde_json::json!({
                    "object": "list",
                    "data": [{"id": "teacher-general-stage1", "object": "model"}]
                }))
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let client = reqwest::Client::new();
        let found = probe_static(
            &client,
            &[format!("http://127.0.0.1:{port}/general-stage1/v1")],
        )
        .await;
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].models, vec!["teacher-general-stage1".to_string()]);
        assert_eq!(found[0].source, "static");
    }
}
