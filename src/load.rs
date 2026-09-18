//! Load-aware routing.
//!
//! Abstraction: `backend → health/capabilities + current load → routing decision`.
//! - backend: [`DiscoveredEndpoint`](crate::discovery::DiscoveredEndpoint)
//!   (capabilities = its `models`, health = reachable on proxy attempt)
//! - current load: [`Load`], refreshed by polling provider probes
//!   (see [`Hive::refresh_load`](crate::hive::Hive::refresh_load))
//! - routing decision: [`Hive::candidates`](crate::hive::Hive::candidates)
//!   (lowest known load first, round-robin on ties / unknown)
//!
//! Provider-aware: every load mechanism is a [`LoadProbe`]. vLLM's
//! `GET /load` is just the first one — future servers (Ollama, LM Studio, …)
//! plug in their own probe without touching the routing code. Probes never
//! exclude a backend: anything but a clear signal is [`Load::Unknown`],
//! and unknown backends stay fully usable.

use crate::logging::BoxFuture;
use serde_json::Value;
use std::sync::Arc;
use std::time::Duration;

/// Load of one backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Load {
    /// Actively tracked request count (vLLM `server_load`) — NOT GPU utilization.
    Known(u64),
    /// No usable signal: no `/load` route, error, tracking disabled, or a
    /// non-vLLM server. Backend stays fully usable; routing falls back to
    /// round-robin for it.
    Unknown,
}

/// One mechanism for reading a backend's load. The first `Known` result
/// wins; anything else must yield `Unknown`, never an exclusion.
pub trait LoadProbe: Send + Sync {
    fn name(&self) -> &'static str;
    fn probe<'a>(&'a self, client: &'a reqwest::Client, base_url: &'a str) -> BoxFuture<'a, Load>;
}

/// vLLM `GET /load` → `{"server_load": N}`.
///
/// Only meaningful when vLLM runs with `--enable-server-load-tracking`
/// (otherwise the route 404s and we report `Unknown`). `N` counts
/// tracked/active requests, not GPU utilization.
pub struct VllmLoadProbe;

impl LoadProbe for VllmLoadProbe {
    fn name(&self) -> &'static str {
        "vllm-/load"
    }

    fn probe<'a>(&'a self, client: &'a reqwest::Client, base_url: &'a str) -> BoxFuture<'a, Load> {
        Box::pin(async move {
            let url = format!("{}/load", base_url.trim_end_matches('/'));
            let resp = match client
                .get(&url)
                .timeout(Duration::from_secs(2))
                .send()
                .await
            {
                Ok(r) => r,
                Err(_) => return Load::Unknown,
            };
            if !resp.status().is_success() {
                return Load::Unknown;
            }
            let body: Value = match resp.json().await {
                Ok(v) => v,
                Err(_) => return Load::Unknown,
            };
            let n = body.get("server_load").and_then(|v| {
                v.as_u64()
                    .or_else(|| v.as_f64().map(|f| f.max(0.0) as u64))
            });
            match n {
                Some(n) => Load::Known(n),
                None => Load::Unknown,
            }
        })
    }
}

/// Poll every probe for one backend; first `Known` wins, else `Unknown`.
pub async fn probe_backend(
    probes: &[Arc<dyn LoadProbe>],
    client: &reqwest::Client,
    base_url: &str,
) -> Load {
    for probe in probes {
        if let Load::Known(n) = probe.probe(client, base_url).await {
            return Load::Known(n);
        }
    }
    Load::Unknown
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{http::StatusCode, routing::get, Json, Router};

    async fn serve(router: Router) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
        url
    }

    fn client() -> reqwest::Client {
        reqwest::Client::new()
    }

    #[tokio::test]
    async fn vllm_load_parses_server_load() {
        let url = serve(Router::new().route(
            "/load",
            get(|| async { Json(serde_json::json!({"server_load": 3})) }),
        ))
        .await;
        assert_eq!(
            VllmLoadProbe.probe(&client(), &url).await,
            Load::Known(3)
        );
    }

    #[tokio::test]
    async fn vllm_load_errors_and_disabled_tracking_mean_unknown() {
        // 404: tracking disabled or not a vLLM server at all.
        let missing =
            serve(Router::new().route("/other", get(|| async { "nothing here" }))).await;
        // 500.
        let broken = serve(Router::new().route(
            "/load",
            get(|| async { (StatusCode::INTERNAL_SERVER_ERROR, "boom") }),
        ))
        .await;
        // 200 with an unexpected shape.
        let weird = serve(Router::new().route(
            "/load",
            get(|| async { Json(serde_json::json!({"foo": 1})) }),
        ))
        .await;
        for url in [missing, broken, weird] {
            assert_eq!(
                VllmLoadProbe.probe(&client(), &url).await,
                Load::Unknown,
                "{url}"
            );
        }
    }

    #[tokio::test]
    async fn unreachable_backend_is_unknown() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        drop(listener); // nothing listens here anymore
        assert_eq!(
            VllmLoadProbe.probe(&client(), &url).await,
            Load::Unknown
        );
    }

    struct FixedProbe(Load);

    impl LoadProbe for FixedProbe {
        fn name(&self) -> &'static str {
            "fixed"
        }

        fn probe<'a>(
            &'a self,
            _client: &'a reqwest::Client,
            _base_url: &'a str,
        ) -> BoxFuture<'a, Load> {
            let load = self.0;
            Box::pin(async move { load })
        }
    }

    #[tokio::test]
    async fn first_known_probe_wins() {
        let probes: Vec<Arc<dyn LoadProbe>> =
            vec![Arc::new(FixedProbe(Load::Unknown)), Arc::new(FixedProbe(Load::Known(7)))];
        assert_eq!(
            probe_backend(&probes, &client(), "http://127.0.0.1:9").await,
            Load::Known(7)
        );
        let none: Vec<Arc<dyn LoadProbe>> = vec![Arc::new(FixedProbe(Load::Unknown))];
        assert_eq!(
            probe_backend(&none, &client(), "http://127.0.0.1:9").await,
            Load::Unknown
        );
    }
}
