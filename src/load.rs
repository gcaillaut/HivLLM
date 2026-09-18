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
//! `/metrics` (always exported) is probed before `/load` (only truthful
//! with `--enable-server-load-tracking`); future servers (Ollama,
//! LM Studio, …) plug in their own probe without touching routing.
//! Probes never exclude a backend: anything but a clear signal is
//! [`Load::Unknown`], and unknown backends stay fully usable.

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
/// `model` scopes the reading; server-level probes (vLLM) may ignore it
/// (their whole box is busy for every model they serve).
pub trait LoadProbe: Send + Sync {
    fn name(&self) -> &'static str;
    fn probe<'a>(
        &'a self,
        client: &'a reqwest::Client,
        base_url: &'a str,
        model: &'a str,
    ) -> BoxFuture<'a, Load>;
}

/// vLLM `GET /metrics` (Prometheus text format): sum of
/// `vllm:num_requests_running` + `vllm:num_requests_waiting` across all
/// engines/models on the backend.
///
/// Unlike `/load`, these gauges are exported regardless of
/// `--enable-server-load-tracking`, so they stay truthful when `/load`
/// is stuck at a bogus `0`. Queue pressure counts: a backend with 32
/// running + 5 waiting reports 37.
pub struct VllmMetricsProbe;

impl LoadProbe for VllmMetricsProbe {
    fn name(&self) -> &'static str {
        "vllm-/metrics"
    }

    fn probe<'a>(
        &'a self,
        client: &'a reqwest::Client,
        base_url: &'a str,
        _model: &'a str,
    ) -> BoxFuture<'a, Load> {
        Box::pin(async move {
            let url = format!("{}/metrics", base_url.trim_end_matches('/'));
            let resp = match client
                .get(&url)
                .timeout(Duration::from_secs(3))
                .send()
                .await
            {
                Ok(r) => r,
                Err(_) => return Load::Unknown,
            };
            if !resp.status().is_success() {
                return Load::Unknown;
            }
            let text = match resp.text().await {
                Ok(t) => t,
                Err(_) => return Load::Unknown,
            };
            match parse_vllm_load(&text) {
                Some(n) => Load::Known(n),
                None => Load::Unknown,
            }
        })
    }
}

/// Sum `vllm:num_requests_running` + `vllm:num_requests_waiting` samples
/// from a Prometheus text exposition. Returns `None` when neither metric
/// is present (metrics disabled / not a vLLM server).
fn parse_vllm_load(text: &str) -> Option<u64> {
    let mut total = 0.0;
    let mut found = false;
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        // Split off the metric name; label values may contain spaces, so
        // the value is the first token AFTER the label block (or name).
        let (name, rest) = match line.find('{') {
            Some(b) => match line[b..].find('}') {
                Some(e) => (&line[..b], &line[b + e + 1..]),
                None => continue, // unbalanced labels
            },
            None => match line.find(char::is_whitespace) {
                Some(i) => (&line[..i], &line[i + 1..]),
                None => continue,
            },
        };
        if name != "vllm:num_requests_running" && name != "vllm:num_requests_waiting" {
            continue;
        }
        // A trailing Prometheus timestamp (if any) is ignored.
        let value: f64 = match rest.split_whitespace().next().and_then(|v| v.parse::<f64>().ok()) {
            Some(f) if f.is_finite() => f,
            _ => continue,
        };
        total += value;
        found = true;
    }
    found.then(|| total.max(0.0) as u64)
}

/// Another HivLLM hive: `GET /load?model=M` answers the aggregate
/// pressure for that model. The `"hivllm": {"exact": bool}` marker
/// distinguishes a real per-model aggregate from a plain vLLM server,
/// which ignores unknown query params and would otherwise answer its
/// box-level total — and `exact: false` aggregates are dropped
/// (`Unknown`), never laundered into exact: an unverified number must
/// not circulate hive-to-hive as fact.
pub struct HivLoadProbe;

impl LoadProbe for HivLoadProbe {
    fn name(&self) -> &'static str {
        "hiv-/load?model"
    }

    fn probe<'a>(
        &'a self,
        client: &'a reqwest::Client,
        base_url: &'a str,
        model: &'a str,
    ) -> BoxFuture<'a, Load> {
        Box::pin(async move {
            let url = format!("{}/load", base_url.trim_end_matches('/'));
            let resp = match client
                .get(&url)
                .query(&[("model", model)])
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
            // Only an explicit exact flag counts. A bare `"hivllm": true`
            // (or any unverified number) is an aggregate that may smear
            // another model's pressure into this one — trusting it as
            // exact would launder approximations into facts across hops.
            let exact = body
                .get("hivllm")
                .and_then(|h| h.get("exact"))
                .and_then(|e| e.as_bool())
                .unwrap_or(false);
            if !exact {
                return Load::Unknown;
            }
            match body.get("server_load").and_then(|v| {
                v.as_u64()
                    .or_else(|| v.as_f64().map(|f| f.max(0.0) as u64))
            }) {
                Some(n) => Load::Known(n),
                None => Load::Unknown,
            }
        })
    }
}

/// vLLM `GET /load` → `{"server_load": N}`.
///
/// Only meaningful when vLLM runs with `--enable-server-load-tracking`
/// (otherwise the route 404s — or answers a permanent bogus `0`, which
/// [`effective_load`] treats as unverified). Responses carrying the
/// `"hivllm"` marker are NOT vLLM boxes and defer to [`HivLoadProbe`].
pub struct VllmLoadProbe;

impl LoadProbe for VllmLoadProbe {
    fn name(&self) -> &'static str {
        "vllm-/load"
    }

    fn probe<'a>(
        &'a self,
        client: &'a reqwest::Client,
        base_url: &'a str,
        _model: &'a str,
    ) -> BoxFuture<'a, Load> {
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
            if body.get("hivllm").is_some() {
                return Load::Unknown;
            }
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

/// Poll every probe for one backend + model; first `Known` wins, else `Unknown`.
pub async fn probe_backend(
    probes: &[Arc<dyn LoadProbe>],
    client: &reqwest::Client,
    base_url: &str,
    model: &str,
) -> Load {
    for probe in probes {
        if let Load::Known(n) = probe.probe(client, base_url, model).await {
            return Load::Known(n);
        }
    }
    Load::Unknown
}

/// Effective load unifying both signals: the worst of what the server
/// reports and what the hive itself observes in flight.
///
/// This is what the load balancer routes on AND what the hive view
/// displays — one number, no split brain. The `bool` tells whether the
/// number is an exact server report (`true`) or a hive-observed
/// approximation (`false`, displayed with a `~` prefix).
///
/// Rationale: `server_load` is only trustworthy when the server runs with
/// `--enable-server-load-tracking`; without it some versions answer a
/// permanent bogus `0`. A positive sighting proves tracking works, but a
/// bare `0` proves nothing (idle and untracked look identical), so zero
/// is always reported as an approximation. Hive-observed in-flight
/// requests are always real, so `max()` degrades gracefully.
pub fn effective_load(server: Load, inflight: u64) -> (u64, bool) {
    match server {
        Load::Known(n) if n > 0 && n >= inflight => (n, true),
        Load::Known(n) => (n.max(inflight), false),
        Load::Unknown => (inflight, false),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{http::StatusCode, routing::get, Json, Router};
    use std::collections::HashMap;

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
            VllmLoadProbe.probe(&client(), &url, "m").await,
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
                VllmLoadProbe.probe(&client(), &url, "m").await,
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
            VllmLoadProbe.probe(&client(), &url, "m").await,
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
            _model: &'a str,
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
            probe_backend(&probes, &client(), "http://127.0.0.1:9", "m").await,
            Load::Known(7)
        );
        let none: Vec<Arc<dyn LoadProbe>> = vec![Arc::new(FixedProbe(Load::Unknown))];
        assert_eq!(
            probe_backend(&none, &client(), "http://127.0.0.1:9", "m").await,
            Load::Unknown
        );
    }

    #[test]
    fn effective_load_takes_the_worst_of_both_signals() {
        // Positive server sighting dominates and proves tracking works.
        assert_eq!(effective_load(Load::Known(5), 2), (5, true));
        // A bare 0 proves nothing (idle and untracked look identical).
        assert_eq!(effective_load(Load::Known(0), 0), (0, false));
        // Broken/absent tracking (stuck at 0) falls back to hive-observed.
        assert_eq!(effective_load(Load::Known(0), 3), (3, false));
        assert_eq!(effective_load(Load::Unknown, 2), (2, false));
        assert_eq!(effective_load(Load::Unknown, 0), (0, false));
    }

    const SAMPLE_METRICS: &str = r#"
# HELP vllm:num_requests_running Number of requests currently running.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="olala-7a1b-50k-antidoom-fix"} 32.0
vllm:num_requests_running{engine="1",model_name="olala-7a1b-50k-antidoom-fix"} 4.0
# HELP vllm:num_requests_waiting Number of requests waiting to be processed.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{engine="0",model_name="olala-7a1b-50k-antidoom-fix"} 0.0
vllm:num_requests_waiting{engine="1",model_name="olala-7a1b-50k-antidoom-fix"} 3.0
# HELP vllm:kv_cache_usage_perc KV-cache usage. 1 means 100 percent usage.
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{engine="0",model_name="olala-7a1b-50k-antidoom-fix"} 0.6311975985922782
"#;

    #[test]
    fn metrics_parser_sums_running_and_waiting() {
        assert_eq!(parse_vllm_load(SAMPLE_METRICS), Some(39));
    }

    #[test]
    fn metrics_parser_ignores_missing_and_broken_series() {
        assert_eq!(parse_vllm_load("# nothing here\nfoo 1\n"), None);
        assert_eq!(parse_vllm_load("vllm:num_requests_running NaN\n"), None);
        // Timestamps and label values with spaces don't confuse it.
        assert_eq!(
            parse_vllm_load("vllm:num_requests_waiting{note=\"a b\"} 2.0 1759999999999\n"),
            Some(2)
        );
    }

    #[tokio::test]
    async fn metrics_probe_reads_running_plus_waiting() {
        let url = serve(Router::new().route(
            "/metrics",
            get(|| async {
                (
                    [(axum::http::header::CONTENT_TYPE, "text/plain")],
                    SAMPLE_METRICS,
                )
            }),
        ))
        .await;
        assert_eq!(
            VllmMetricsProbe.probe(&client(), &url, "m").await,
            Load::Known(39)
        );
    }

    #[tokio::test]
    async fn metrics_probe_missing_endpoint_is_unknown() {
        let url = serve(Router::new().route("/other", get(|| async { "x" }))).await;
        assert_eq!(
            VllmMetricsProbe.probe(&client(), &url, "m").await,
            Load::Unknown
        );
    }

    #[tokio::test]
    async fn hiv_probe_reads_marked_per_model_aggregate() {
        let url = serve(Router::new().route(
            "/load",
            get(|axum::extract::Query(q): axum::extract::Query<HashMap<String, String>>| async move {
                let model = q.get("model").cloned().unwrap_or_default();
                let n = if model == "hot" { 12 } else { 0 };
                Json(serde_json::json!({ "server_load": n, "hivllm": { "exact": true } }))
            }),
        ))
        .await;
        assert_eq!(
            HivLoadProbe.probe(&client(), &url, "hot").await,
            Load::Known(12)
        );
        // Exact zero is still exact (positively sighted).
        assert_eq!(
            HivLoadProbe.probe(&client(), &url, "cold").await,
            Load::Known(0)
        );
    }

    #[tokio::test]
    async fn hiv_probe_drops_inexact_aggregates() {
        // An unverified aggregate must not circulate as fact — including
        // the legacy flat marker, which carries no per-model provenance.
        for body in [
            serde_json::json!({ "server_load": 31, "hivllm": { "exact": false } }),
            serde_json::json!({ "server_load": 31, "hivllm": true }),
        ] {
            let url = serve(Router::new().route(
                "/load",
                get({
                    let body = body.clone();
                    || async move { Json(body.clone()) }
                }),
            ))
            .await;
            assert_eq!(
                HivLoadProbe.probe(&client(), &url, "m").await,
                Load::Unknown,
                "{body}"
            );
        }
    }

    #[tokio::test]
    async fn bare_load_probe_defers_to_hiv_marker() {
        // A marked body is hive business even on the bare route.
        let url = serve(Router::new().route(
            "/load",
            get(|| async {
                Json(serde_json::json!({ "server_load": 7, "hivllm": { "exact": true } }))
            }),
        ))
        .await;
        assert_eq!(
            VllmLoadProbe.probe(&client(), &url, "m").await,
            Load::Unknown
        );
    }

    #[tokio::test]
    async fn hiv_probe_rejects_unmarked_server_load() {
        // A plain vLLM server ignores the ?model= param and answers its
        // box-level total — without the marker it must not count as a
        // per-model hive aggregate.
        let url = serve(Router::new().route(
            "/load",
            get(|| async { Json(serde_json::json!({ "server_load": 5 })) }),
        ))
        .await;
        assert_eq!(
            HivLoadProbe.probe(&client(), &url, "m").await,
            Load::Unknown
        );
    }
}
