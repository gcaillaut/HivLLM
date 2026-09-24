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
    /// Hive members only: model → the hive-id paths through which this
    /// member reaches a real backend for it, each starting with the
    /// member's own id (path vector, see [`hive_paths`]).
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub paths: HashMap<String, Vec<Vec<String>>>,
    /// Model → the upstream's `/v1/models` object (`max_model_len`,
    /// `root`, …), re-served by the hive's `/v1/models`.
    #[serde(default, skip_serializing)]
    pub model_meta: HashMap<String, serde_json::Value>,
}

/// Request header carrying the comma-separated instance ids of hives that
/// already forwarded this request — or, on a `/v1/models` probe, the id
/// of the probing hive, so the answer leaves out routes through it.
pub const VIA_HEADER: &str = "x-hivllm-via";

/// Longest hive path kept: deeper chains are dropped, never trusted.
pub const MAX_HOPS: usize = 8;

/// Response header every hive stamps with its instance id, so peers can
/// recognise it whatever address they reach it by.
pub const HIVE_ID_HEADER: &str = "x-hivllm-id";

/// A successful `/v1/models` probe.
#[derive(Debug, Clone)]
pub struct Probed {
    pub models: Vec<String>,
    /// Instance id when the backend is a hive.
    pub hive_id: Option<String>,
    /// Hive backends only: see [`DiscoveredEndpoint::paths`].
    pub paths: HashMap<String, Vec<Vec<String>>>,
    /// See [`DiscoveredEndpoint::model_meta`].
    pub model_meta: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct ModelList {
    #[serde(default)]
    data: Vec<serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct ModelEntry {
    id: String,
    /// Set by HivLLM hives: routes behind this model.
    #[serde(default)]
    hivllm: Option<HivModelMeta>,
}

/// `hivllm` extension of a model object in a hive's `/v1/models`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HivModelMeta {
    /// Hive-id paths from the advertising hive (excluded) to a real
    /// backend: `[]` = one of its own backends serves it directly.
    #[serde(default)]
    pub paths: Vec<Vec<String>>,
}

/// Paths through hive member `hive_id` for one advertised model: each
/// advertised path prefixed with the member itself. Hives that predate
/// path vectors advertise none, which reads as a single opaque hop.
/// Over-long paths are dropped.
pub fn hive_paths(hive_id: &str, meta: Option<&HivModelMeta>) -> Vec<Vec<String>> {
    let advertised = match meta {
        Some(m) => m.paths.clone(),
        None => vec![Vec::new()],
    };
    advertised
        .into_iter()
        .map(|p| std::iter::once(hive_id.to_string()).chain(p).collect::<Vec<_>>())
        .filter(|p| p.len() <= MAX_HOPS)
        .collect()
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

/// Join a base URL with an API path without doubling a `/v1` prefix the
/// base already carries. OpenAI paths (`/v1/…`) go under the base; other
/// server routes (`/tokenize`, `/v2/rerank`, …) live next to `/v1`, at
/// the [`server_root`].
pub fn join_upstream_path(base_url: &str, api_path: &str) -> String {
    let b = base_url.trim_end_matches('/');
    if let Some(rest) = api_path.strip_prefix("/v1").filter(|r| r.is_empty() || r.starts_with('/')) {
        if b.ends_with("/v1") {
            return format!("{b}{rest}");
        }
        return format!("{b}{api_path}");
    }
    format!("{}{api_path}", server_root(b))
}

/// Probe a single base URL for OpenAI compat. Returns model ids (and, for
/// hives, their id and per-model paths) on success. `self_id` (the
/// probing hive) rides along as Via, so a hive answers only with routes
/// that don't lead back through the prober.
pub async fn probe_openai_endpoint(
    client: &reqwest::Client,
    base_url: &str,
    self_id: &str,
) -> Option<Probed> {
    let url = models_url(base_url);
    let resp = client
        .get(&url)
        .header(VIA_HEADER, self_id)
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
    let mut paths: HashMap<String, Vec<Vec<String>>> = HashMap::new();
    let mut model_meta = HashMap::new();
    let mut models = Vec::new();
    for raw in body.data {
        let Ok(entry) = serde_json::from_value::<ModelEntry>(raw.clone()) else {
            continue; // no string `id`
        };
        model_meta.insert(entry.id.clone(), raw);
        if let Some(hid) = &hive_id {
            let found = hive_paths(hid, entry.hivllm.as_ref());
            if found.is_empty() {
                continue; // only over-long routes: not usable
            }
            paths.entry(entry.id.clone()).or_default().extend(found);
        }
        models.push(entry.id);
    }
    models.sort();
    models.dedup();
    Some(Probed {
        models,
        hive_id,
        paths,
        model_meta,
    })
}

/// Longest a helper command (`ss`, `docker ps`) may take: a wedged Docker
/// daemon must not stall discovery.
const COMMAND_TIMEOUT: Duration = Duration::from_secs(5);

/// Stdout of `program args…`, or empty when it is missing, fails or
/// outlives `timeout` (then it is killed). Async: never blocks a runtime
/// worker thread.
async fn command_output(program: &str, args: &[&str], timeout: Duration) -> String {
    let child = tokio::process::Command::new(program)
        .args(args)
        .stdin(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .kill_on_drop(true)
        .output();
    match tokio::time::timeout(timeout, child).await {
        Ok(Ok(out)) if out.status.success() => String::from_utf8_lossy(&out.stdout).into_owned(),
        Ok(Ok(_)) | Ok(Err(_)) => String::new(),
        Err(_) => {
            tracing::warn!(program, ?timeout, "discovery command timed out, skipped");
            String::new()
        }
    }
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

/// `ss -tlnp` output -> map port -> process name (best effort: names
/// only show for processes this user may inspect).
fn parse_ss_processes(out: &str) -> HashMap<u16, String> {
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

/// `docker ps --format {{.Ports}}` output -> published host ports.
fn parse_docker_ports(out: &str) -> Vec<u16> {
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
    self_id: &str,
) -> Vec<DiscoveredEndpoint> {
    // One `ss -tlnp` covers both listening ports and process names;
    // both commands run concurrently, each bounded (missing = empty).
    let (ss_out, docker_out) = tokio::join!(
        command_output("ss", &["-tlnp"], COMMAND_TIMEOUT),
        command_output("docker", &["ps", "--format", "{{.Ports}}"], COMMAND_TIMEOUT),
    );
    let docker_ports: HashSet<u16> = parse_docker_ports(&docker_out).into_iter().collect();
    let proc_map = parse_ss_processes(&ss_out);

    let mut candidates: Vec<u16> = well_known_ports(extra_ports);
    candidates.extend(parse_ss_ports(&ss_out));
    candidates.extend(docker_ports.iter().copied());
    candidates.sort_unstable();
    candidates.dedup();
    candidates.retain(|p| !exclude_ports.contains(p));

    let results: Vec<Option<DiscoveredEndpoint>> = stream::iter(candidates)
        .map(|port| {
            let proc_map = &proc_map;
            let docker_ports = &docker_ports;
            async move {
                let base_url = format!("http://127.0.0.1:{port}");
                let probed = probe_openai_endpoint(client, &base_url, self_id).await?;
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
                    paths: probed.paths,
                    model_meta: probed.model_meta,
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
    self_id: &str,
) -> Vec<DiscoveredEndpoint> {
    let probed: Vec<Option<DiscoveredEndpoint>> = stream::iter(urls.iter().cloned())
        .map(|raw| {
            let client = client.clone();
            async move {
                let base_url = normalize_base_url(&raw);
                let probed = match probe_openai_endpoint(&client, &base_url, self_id).await {
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
                    paths: probed.paths,
                    model_meta: probed.model_meta,
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
                for (model, meta) in ep.model_meta {
                    existing.model_meta.entry(model).or_insert(meta);
                }
                for (model, paths) in ep.paths {
                    let slot = existing.paths.entry(model).or_default();
                    for p in paths {
                        if !slot.contains(&p) {
                            slot.push(p);
                        }
                    }
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
                paths: HashMap::new(),
                model_meta: HashMap::new(),
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
        assert_eq!(
            join_upstream_path("http://127.0.0.1:8180/general-stage1/v1", "/tokenize"),
            "http://127.0.0.1:8180/general-stage1/tokenize"
        );
        assert_eq!(
            join_upstream_path("http://127.0.0.1:8000", "/v2/rerank"),
            "http://127.0.0.1:8000/v2/rerank"
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
            "test-hive",
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
            "test-hive",
        )
        .await;
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].models, vec!["teacher-general-stage1".to_string()]);
        assert_eq!(found[0].source, "static");
    }

    #[test]
    fn hive_paths_prefix_the_member_and_drop_long_chains() {
        let meta = HivModelMeta {
            paths: vec![vec![], vec!["b".into()]],
        };
        assert_eq!(
            hive_paths("a", Some(&meta)),
            vec![vec!["a".to_string()], vec!["a".to_string(), "b".to_string()]]
        );
        // Pre-path-vector hive: one opaque hop.
        assert_eq!(hive_paths("a", None), vec![vec!["a".to_string()]]);
        let long = HivModelMeta {
            paths: vec![(0..MAX_HOPS).map(|i| format!("h{i}")).collect()],
        };
        assert!(hive_paths("a", Some(&long)).is_empty());
    }

    #[test]
    fn one_ss_tlnp_call_gives_ports_and_process_names() {
        let sample = "State Recv-Q Send-Q Local Address:Port Peer Address:Port Process\n\
LISTEN 0 4096 127.0.0.1:11434 0.0.0.0:* users:((\"ollama\",pid=812,fd=3))\n\
LISTEN 0 2048 0.0.0.0:8000 0.0.0.0:* users:((\"python3\",pid=99,fd=12))\n\
LISTEN 0 128 [::]:9090 [::]:*\n";
        assert_eq!(parse_ss_ports(sample), vec![8000, 9090, 11434]);
        let procs = parse_ss_processes(sample);
        assert_eq!(procs.get(&11434).map(String::as_str), Some("ollama"));
        assert_eq!(procs.get(&8000).map(String::as_str), Some("python3"));
        assert!(!procs.contains_key(&9090));
    }

    #[test]
    fn parses_docker_published_ports() {
        let out = "0.0.0.0:8000->8000/tcp, :::8000->8000/tcp\n127.0.0.1:11434->11434/tcp\n\n";
        assert_eq!(parse_docker_ports(out), vec![8000, 11434]);
    }

    #[tokio::test]
    async fn slow_or_missing_commands_yield_empty_output() {
        let start = std::time::Instant::now();
        assert_eq!(command_output("sleep", &["10"], Duration::from_millis(200)).await, "");
        assert!(start.elapsed() < Duration::from_secs(2));
        assert_eq!(command_output("hivllm-no-such-binary", &[], COMMAND_TIMEOUT).await, "");
        assert_eq!(command_output("echo", &["hi"], COMMAND_TIMEOUT).await, "hi\n");
    }
}
