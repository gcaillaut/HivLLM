//! Docker socket discovery: find OpenAI-compatible backends running as
//! containers on the hive's networks, and follow their lifecycle.
//!
//! Opt-in via labels (Traefik-style):
//! - `hivllm.enable=true` — probe this container (anything else is ignored).
//! - `hivllm.port=8080` — serving port override (else exposed ports +
//!   well-known LLM ports are tried).
//!
//! Containers are addressed by NAME (stable across recreates; IPs are not)
//! which requires a user-defined network (default `bridge` has no DNS).
//! [`DockerDiscovery::watch_events`] subscribes to container lifecycle
//! events so backends appear/disappear in ~seconds instead of on the next
//! periodic rescan (which stays as backstop).
//!
//! Needs the Engine socket mounted (`-v /var/run/docker.sock:…`) and
//! `--docker-socket`. Absent socket = this whole module stays idle.

use crate::discovery::{probe_openai_endpoint, well_known_ports, DiscoveredEndpoint};
use crate::hive::Hive;
use bollard::container::ListContainersOptions;
use bollard::models::EventMessageTypeEnum;
use bollard::system::EventsOptions;
use bollard::Docker;
use futures::{stream, StreamExt};
use std::collections::HashMap;
use std::time::Duration;

pub const ENABLE_LABEL: &str = "hivllm.enable";
pub const PORT_LABEL: &str = "hivllm.port";

/// Container lifecycle actions worth a rediscovery.
const LIFECYCLE_ACTIONS: &[&str] = &[
    "start", "die", "stop", "destroy", "kill", "pause", "unpause", "rename",
];

#[derive(Clone)]
pub struct DockerDiscovery {
    docker: Docker,
}

impl DockerDiscovery {
    /// Connect and verify with a ping so a bad path fails fast.
    pub async fn connect(socket_path: &str) -> Result<Self, bollard::errors::Error> {
        let docker = Docker::connect_with_socket(
            socket_path,
            120,
            bollard::API_DEFAULT_VERSION,
        )?;
        docker.ping().await?;
        Ok(Self { docker })
    }

    /// Probe every labeled container for `/v1/models`.
    pub async fn container_backends(
        &self,
        client: &reqwest::Client,
        self_id: &str,
    ) -> Vec<DiscoveredEndpoint> {
        let containers = match self
            .docker
            .list_containers::<String>(Some(ListContainersOptions {
                all: false,
                ..Default::default()
            }))
            .await
        {
            Ok(c) => c,
            Err(e) => {
                tracing::warn!(error = %e, "docker container list failed");
                return Vec::new();
            }
        };
        let jobs: Vec<(String, Vec<u16>)> = containers
            .iter()
            .filter_map(|c| {
                let labels = c.labels.as_ref()?;
                if !is_enabled(labels) {
                    return None;
                }
                let name = container_name(c)?;
                let mut exposed: Vec<u16> = c
                    .ports
                    .as_ref()
                    .map(|ports| {
                        ports
                            .iter()
                            .map(|p| p.private_port)
                            .filter(|p| *p > 0)
                            .collect()
                    })
                    .unwrap_or_default();
                exposed.sort_unstable();
                Some((name, candidate_ports(labels, &exposed)))
            })
            .collect();
        stream::iter(jobs)
            .map(|(name, ports)| {
                let client = client.clone();
                async move {
                    for port in ports {
                        let base_url = format!("http://{name}:{port}");
                        if let Some(probed) =
                            probe_openai_endpoint(&client, &base_url, self_id).await
                        {
                            return Some(DiscoveredEndpoint {
                                id: format!("docker-{name}-{port}"),
                                name: name.clone(),
                                base_url,
                                models: probed.models,
                                source: "docker".to_string(),
                                hive_id: probed.hive_id,
                                paths: probed.paths,
                                model_meta: probed.model_meta,
                            });
                        }
                    }
                    tracing::debug!(%name, "labeled container serves no OpenAI endpoint");
                    None
                }
            })
            .buffer_unordered(16)
            .collect::<Vec<_>>()
            .await
            .into_iter()
            .flatten()
            .collect()
    }

    /// Follow container lifecycle events; refresh the hive after each burst
    /// (coalesced on a 2s quiet window so `compose up` storms refresh once).
    /// Reconnects forever — never returns under normal operation.
    pub async fn watch_events(
        &self,
        hive: Hive,
        extra_ports: Vec<u16>,
        static_backends: Vec<String>,
    ) {
        loop {
            let mut events = self.docker.events::<String>(Some(EventsOptions {
                filters: HashMap::from([(
                    "type".to_string(),
                    vec!["container".to_string()],
                )]),
                ..Default::default()
            }));
            // Phase 1: wait for the first relevant event.
            let mut alive = true;
            let mut triggered = false;
            while !triggered && alive {
                match events.next().await {
                    Some(Ok(evt)) if is_lifecycle(&evt) => triggered = true,
                    Some(Ok(_)) => {}
                    Some(Err(e)) => {
                        tracing::warn!(error = %e, "docker events error, reconnecting");
                        alive = false;
                    }
                    None => {
                        tracing::warn!("docker events stream ended, reconnecting");
                        alive = false;
                    }
                }
            }
            if !alive {
                tokio::time::sleep(Duration::from_secs(5)).await;
                continue;
            }
            // Phase 2: coalesce the burst on 2s of quiet.
            loop {
                match tokio::time::timeout(Duration::from_secs(2), events.next()).await {
                    Ok(Some(Ok(evt))) if is_lifecycle(&evt) => continue,
                    Ok(Some(_)) => continue,
                    _ => break,
                }
            }
            tracing::info!("docker containers changed, refreshing hive");
            hive
                .refresh(&extra_ports, &static_backends, Some(self))
                .await;
            hive.log_view().await;
        }
    }
}

/// Labels gate discovery: exactly `hivllm.enable=true` opts in.
pub fn is_enabled(labels: &HashMap<String, String>) -> bool {
    labels.get(ENABLE_LABEL).is_some_and(|v| v == "true")
}

/// Explicit serving port from `hivllm.port=N`.
pub fn labeled_port(labels: &HashMap<String, String>) -> Option<u16> {
    labels.get(PORT_LABEL)?.parse().ok()
}

/// Ports to probe, in order: explicit label, exposed private ports,
/// well-known LLM ports. Deduplicated.
pub fn candidate_ports(labels: &HashMap<String, String>, exposed: &[u16]) -> Vec<u16> {
    let mut ports = Vec::new();
    if let Some(p) = labeled_port(labels) {
        ports.push(p);
    }
    let mut exp = exposed.to_vec();
    exp.sort_unstable();
    exp.dedup();
    for p in exp.into_iter().chain(well_known_ports(&[])) {
        if !ports.contains(&p) {
            ports.push(p);
        }
    }
    ports
}

/// First container name, without Docker's leading `/`.
pub fn container_name(
    c: &bollard::models::ContainerSummary,
) -> Option<String> {
    c.names
        .as_ref()?
        .first()
        .map(|n| n.trim_start_matches('/').to_string())
        .filter(|n| !n.is_empty())
}

fn is_lifecycle(evt: &bollard::models::EventMessage) -> bool {
    matches!(evt.typ, Some(EventMessageTypeEnum::CONTAINER))
        && evt
            .action
            .as_deref()
            .is_some_and(|a| LIFECYCLE_ACTIONS.contains(&a))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn labels(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn enable_label_is_exact() {
        assert!(is_enabled(&labels(&[(ENABLE_LABEL, "true")])));
        assert!(!is_enabled(&labels(&[(ENABLE_LABEL, "1")])));
        assert!(!is_enabled(&labels(&[(ENABLE_LABEL, "True")])));
        assert!(!is_enabled(&labels(&[])));
    }

    #[test]
    fn labeled_port_parses() {
        assert_eq!(
            labeled_port(&labels(&[(PORT_LABEL, "8080")])),
            Some(8080)
        );
        assert_eq!(labeled_port(&labels(&[(PORT_LABEL, "x")])), None);
        assert_eq!(labeled_port(&labels(&[])), None);
    }

    #[test]
    fn ports_prefer_label_then_exposed_then_well_known() {
        let ports = candidate_ports(&labels(&[(PORT_LABEL, "5000")]), &[8000, 8000]);
        assert_eq!(ports[0], 5000);
        assert!(ports.contains(&8000));
        assert!(ports.contains(&11434)); // well-known default
        assert_eq!(ports.len(), ports.iter().collect::<std::collections::HashSet<_>>().len());
    }

    #[test]
    fn ports_without_label_start_with_exposed() {
        let ports = candidate_ports(&labels(&[]), &[9000]);
        assert_eq!(ports[0], 9000);
    }
}
