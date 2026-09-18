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

/// Probe a single base URL for OpenAI compat. Returns model ids on success.
pub async fn probe_openai_endpoint(
    client: &reqwest::Client,
    base_url: &str,
) -> Option<Vec<String>> {
    let url = format!("{}/v1/models", base_url.trim_end_matches('/'));
    let resp = client
        .get(&url)
        .timeout(Duration::from_secs(2))
        .send()
        .await
        .ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let body: ModelList = resp.json().await.ok()?;
    let mut models: Vec<String> = body.data.into_iter().map(|m| m.id).collect();
    models.sort();
    models.dedup();
    Some(models)
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
                let models = probe_openai_endpoint(client, &base_url).await?;
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
                    models,
                    source: source.to_string(),
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
}
