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
    sync::{Arc, Mutex as StdMutex},
    time::{Duration, Instant},
};
use tokio::sync::{oneshot, Mutex, RwLock};

use crate::discovery::{
    discover, join_upstream_path, merge_endpoints, probe_static, Credentials, DiscoveredEndpoint,
    HivModelMeta, HIVE_ID_HEADER, VIA_HEADER,
};
use crate::docker::DockerDiscovery;
use crate::load::{effective_load, probe_backend, HivLoadProbe, Load, LoadProbe, VllmLoadProbe, VllmMetricsProbe};
use crate::logging::{
    extract_response, read_recent, truncate_text, truncate_value, LogEntry, LoggedResponse, RequestLogger, StreamAcc,
    StreamSummary, TeeStream, Truncate,
};

/// Candidate ordering key: (cooling after a failure, effective load).
type RankKey = (bool, u64);

#[derive(Clone)]
pub struct Hive {
    pub client: reqwest::Client,
    /// Port the hive itself listens on — always excluded from discovery.
    own_port: u16,
    /// This hive's instance id: stamped on every response
    /// (`x-hivllm-id`) and appended to the Via path of forwarded requests.
    /// Unique per process unless pinned with `--hive-id`, so peers
    /// recognise this hive whatever address they reach it by.
    own_id: String,
    logger: RequestLogger,
    log_truncate: Truncate,
    log_max_chars: usize,
    /// Per-model probes for members that are hives (`/load?model=`).
    hive_probes: Vec<Arc<dyn LoadProbe>>,
    /// Server-level probes for plain backends, run once per server and
    /// applied to all its models (a busy box is busy for every model).
    server_probes: Vec<Arc<dyn LoadProbe>>,
    /// Appended for single-model servers only: bare vLLM `/load` can't be
    /// attributed to one model of many.
    single_model_probes: Vec<Arc<dyn LoadProbe>>,
    /// (endpoint id, model) → last polled load. Rebuilt by every
    /// [`Hive::refresh_load`].
    loads: Arc<RwLock<HashMap<(String, String), Load>>>,
    /// base_url → requests currently being served by the hive. Used as a
    /// load approximation for backends without load tracking. Only
    /// touched through [`InflightGuard`], so cancelled requests can't leak.
    inflight: Arc<StdMutex<HashMap<String, u64>>>,
    /// base_url → instant until which the backend is ranked last, set when
    /// a request to it failed at the transport level. Cooling backends stay
    /// usable (tried after everyone else), they just stop looking idle.
    cooldown: Arc<StdMutex<HashMap<String, Instant>>>,
    /// Forward the client's `Authorization` header to backends (opt-in:
    /// by default a token never leaves the hive).
    forward_auth: bool,
    /// Per-backend API keys (config file). A backend's own key always
    /// wins over a forwarded client token.
    credentials: Arc<Credentials>,
    /// Largest accepted request body (`None` = unlimited). Axum's 2 MB
    /// default rejects long contexts and base64 images.
    max_body: Option<usize>,
    /// Last rendered hive view (change detection for stdout logging).
    last_view: Arc<Mutex<String>>,
    endpoints: Arc<RwLock<Vec<DiscoveredEndpoint>>>,
    /// Serialises discovery passes (periodic rescan, Docker events): a slow
    /// pass finishing after a newer one would otherwise overwrite fresh
    /// membership with stale results.
    scan_lock: Arc<Mutex<()>>,
    /// base_url → consecutive scans a member has been missing from.
    missed_scans: Arc<StdMutex<HashMap<String, u32>>>,
    /// Consecutive missed scans before a member leaves (1 = at once).
    drop_after: u32,
    /// model -> round-robin counter, advanced once per routing decision.
    /// Every run of equal effective loads rotates by it, so ties take
    /// turns; all-idle degrades to plain round-robin.
    rr: Arc<Mutex<HashMap<String, usize>>>,
}

impl Hive {
    pub fn new(own_port: u16) -> Self {
        Self {
            client: upstream_client(DEFAULT_CONNECT_TIMEOUT, Some(DEFAULT_READ_TIMEOUT)),
            own_port,
            own_id: new_hive_id(),
            logger: RequestLogger::new(),
            log_truncate: Truncate::None,
            log_max_chars: 2000,
            hive_probes: vec![Arc::new(HivLoadProbe)],
            server_probes: vec![Arc::new(VllmMetricsProbe)],
            single_model_probes: vec![Arc::new(VllmLoadProbe)],
            loads: Arc::new(RwLock::new(HashMap::new())),
            inflight: Arc::new(StdMutex::new(HashMap::new())),
            cooldown: Arc::new(StdMutex::new(HashMap::new())),
            forward_auth: false,
            credentials: Arc::new(Credentials::default()),
            max_body: Some(DEFAULT_MAX_BODY),
            last_view: Arc::new(Mutex::new(String::new())),
            endpoints: Arc::new(RwLock::new(Vec::new())),
            scan_lock: Arc::new(Mutex::new(())),
            missed_scans: Arc::new(StdMutex::new(HashMap::new())),
            drop_after: DEFAULT_DROP_AFTER,
            rr: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn with_logger(mut self, logger: RequestLogger) -> Self {
        self.logger = logger;
        self
    }

    /// Pin the instance id (`--hive-id`). Must be a valid header value
    /// without commas (Via paths are comma-separated).
    pub fn with_hive_id(mut self, id: String) -> Self {
        self.own_id = id;
        self
    }

    /// Upstream timeouts: `connect` bounds TCP/TLS setup (fast failover to
    /// the next candidate), `read` bounds the silence between two reads
    /// (`None` = wait forever). There is deliberately no total timeout:
    /// a long generation that keeps streaming is never cut off.
    pub fn with_timeouts(mut self, connect: Duration, read: Option<Duration>) -> Self {
        self.client = upstream_client(connect, read);
        self
    }

    /// Consecutive failed scans before a member leaves the hive (min 1).
    pub fn with_drop_after(mut self, scans: u32) -> Self {
        self.drop_after = scans.max(1);
        self
    }

    pub fn with_max_body(mut self, max: Option<usize>) -> Self {
        self.max_body = max;
        self
    }

    pub fn with_credentials(mut self, credentials: Credentials) -> Self {
        self.credentials = Arc::new(credentials);
        self
    }

    pub fn with_forward_auth(mut self, forward: bool) -> Self {
        self.forward_auth = forward;
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

    pub async fn refresh(
        &self,
        extra_ports: &[u16],
        static_backends: &[String],
        docker: Option<&DockerDiscovery>,
    ) {
        let me = self.own_id.as_str();
        let creds = self.credentials.as_ref();
        let own_port = [self.own_port];
        self.scan_and_install(async {
            let docked = async {
                match docker {
                    Some(d) => d.container_backends(&self.client, me, creds).await,
                    None => Vec::new(),
                }
            };
            let (found, pinned, docked) = tokio::join!(
                discover(&self.client, extra_ports, &own_port, me, creds),
                probe_static(&self.client, static_backends, me, creds),
                docked,
            );
            merge_endpoints(merge_endpoints(found, pinned), docked)
        })
        .await;
    }

    /// Run one discovery pass and install its result, one pass at a time:
    /// a pass starts probing only after the previous one is installed, so
    /// the last write is always the freshest view.
    async fn scan_and_install<F>(&self, scan: F)
    where
        F: std::future::Future<Output = Vec<DiscoveredEndpoint>>,
    {
        let _one_pass_at_a_time = self.scan_lock.lock().await;
        let found = scan.await;
        self.set_endpoints(found).await;
    }

    /// Install a freshly probed member list, minus anything that leads
    /// back into this hive (see [`Hive::without_self`]).
    pub async fn set_endpoints(&self, endpoints: Vec<DiscoveredEndpoint>) {
        let fresh = self.without_self(endpoints);
        let mut current = self.endpoints.write().await;
        let kept = self.carry_over_missing(&current, &fresh);
        let mut all = fresh;
        all.extend(kept);
        all.sort_by(|a, b| a.base_url.cmp(&b.base_url));
        let served: HashSet<&String> = all.iter().flat_map(|e| &e.models).collect();
        self.rr.lock().await.retain(|m, _| served.contains(m));
        *current = all;
    }

    /// Members absent from this scan that are kept anyway, with their last
    /// known models, until they miss `drop_after` scans in a row: a busy
    /// backend answering one `/v1/models` probe late must not vanish (and
    /// 404 its model) until the next scan. Members that answered are
    /// always taken as they are now. Docker members leave at once: the
    /// Engine API is authoritative on whether a container runs.
    fn carry_over_missing(
        &self,
        previous: &[DiscoveredEndpoint],
        fresh: &[DiscoveredEndpoint],
    ) -> Vec<DiscoveredEndpoint> {
        let present: HashSet<&str> = fresh.iter().map(|e| e.base_url.as_str()).collect();
        let mut missed = lock(&self.missed_scans);
        missed.retain(|url, _| !present.contains(url.as_str()));
        let mut kept = Vec::new();
        for ep in previous.iter().filter(|e| !present.contains(e.base_url.as_str())) {
            let n = missed.entry(ep.base_url.clone()).or_insert(0);
            *n += 1;
            if ep.source != "docker" && *n < self.drop_after {
                tracing::debug!(base_url = %ep.base_url, missed = *n, "member missed a scan, kept");
                kept.push(ep.clone());
            } else {
                tracing::info!(base_url = %ep.base_url, missed = *n, "member left the hive");
                missed.remove(&ep.base_url);
            }
        }
        kept
    }

    /// Drop what leads back into this hive:
    /// - endpoints that ARE this hive reached by another address (a static
    ///   URL, a LAN IP, a container name) — the own-port exclusion only
    ///   covers localhost discovery;
    /// - hive-member paths through this hive, and models left with no
    ///   other path. Without this, hives that discover each other keep a
    ///   dead backend's model alive forever by re-advertising it in a
    ///   cycle.
    fn without_self(&self, endpoints: Vec<DiscoveredEndpoint>) -> Vec<DiscoveredEndpoint> {
        let me = self.own_id.as_str();
        endpoints
            .into_iter()
            .filter(|ep| {
                let is_self = ep.hive_id.as_deref() == Some(me);
                if is_self {
                    tracing::debug!(base_url = %ep.base_url, "skipping endpoint: it is this hive");
                }
                !is_self
            })
            .map(|mut ep| {
                if let Some(hid) = ep.hive_id.clone() {
                    let models = std::mem::take(&mut ep.models);
                    for m in models {
                        let paths: Vec<Vec<String>> = member_paths(&hid, &ep, &m)
                            .into_iter()
                            .filter(|p| !p.iter().any(|h| h == me))
                            .collect();
                        if paths.is_empty() {
                            ep.paths.remove(&m);
                        } else {
                            ep.paths.insert(m.clone(), paths);
                            ep.models.push(m);
                        }
                    }
                }
                ep
            })
            .collect()
    }

    /// Paths to advertise for `model` to a requester whose Via path is
    /// `via`: every route of every member, minus routes through the
    /// requester's path or this hive, shortest first, capped. Relative to
    /// this hive (it is not included); `[]` = served by a direct backend.
    /// Empty = this hive can't serve `model` to that requester.
    fn advertised_paths(
        endpoints: &[DiscoveredEndpoint],
        model: &str,
        avoid: &HashSet<String>,
    ) -> Vec<Vec<String>> {
        let mut out: Vec<Vec<String>> = Vec::new();
        for ep in endpoints.iter().filter(|e| e.models.iter().any(|m| m == model)) {
            for p in routes_of(ep, model) {
                if p.len() < crate::discovery::MAX_HOPS && loop_free(&p, avoid) && !out.contains(&p) {
                    out.push(p);
                }
            }
        }
        out.sort_by_key(Vec::len);
        out.truncate(MAX_ADVERTISED_PATHS);
        out
    }

    pub async fn snapshot(&self) -> Vec<DiscoveredEndpoint> {
        self.endpoints.read().await.clone()
    }

    /// This hive's instance id (see [`Hive::with_hive_id`]).
    pub fn own_id(&self) -> String {
        self.own_id.clone()
    }

    /// Aggregate load of the whole hive (`model = None`) or of one model,
    /// as `(number, exact)`. Served as `GET /load[?model=]` with marker
    /// `"hivllm": {"exact": bool}` so upstream hives route on real numbers.
    ///
    /// The minimum effective load over members: the load the next request
    /// would face, since the hive routes it to its least-loaded member.
    /// Exact iff that member's number is (ties prefer exact), so
    /// approximations never launder into exact across hops. No members →
    /// `(0, false)`.
    pub async fn aggregate_load(&self, model: Option<&str>) -> (u64, bool) {
        let endpoints = self.endpoints.read().await;
        let loads = self.loads.read().await;
        let inflight = lock(&self.inflight);
        endpoints
            .iter()
            .flat_map(|ep| ep.models.iter().map(move |m| (ep, m)))
            .filter(|(_, m)| model.is_none_or(|wanted| wanted == m.as_str()))
            .map(|(ep, m)| {
                let server = loads
                    .get(&(ep.id.clone(), m.clone()))
                    .copied()
                    .unwrap_or(Load::Unknown);
                let flying = inflight.get(&ep.base_url).copied().unwrap_or(0);
                effective_load(server, flying)
            })
            .min_by_key(|&(n, exact)| (n, !exact))
            .unwrap_or((0, false))
    }

    /// Poll load probes (concurrently) and replace the load map: once per
    /// plain server (`/metrics`, then bare `/load` for single-model
    /// servers), once per model for hive members (only their per-model
    /// aggregate means anything). Backends that fail probing stay
    /// `Unknown` — usable, routed round-robin. Run on a short interval;
    /// never per-request.
    pub async fn refresh_load(&self) {
        let endpoints = self.endpoints.read().await.clone();
        // (endpoint id, base_url, probe chain, model to ask about, models the answer covers)
        type Job = (String, String, Vec<Arc<dyn LoadProbe>>, String, Vec<String>);
        let mut jobs: Vec<Job> = Vec::new();
        for ep in &endpoints {
            if ep.hive_id.is_some() {
                for m in &ep.models {
                    let chain = self.hive_probes.clone();
                    jobs.push((ep.id.clone(), ep.base_url.clone(), chain, m.clone(), vec![m.clone()]));
                }
            } else if !ep.models.is_empty() {
                let mut chain = self.server_probes.clone();
                if ep.models.len() == 1 {
                    chain.extend(self.single_model_probes.iter().cloned());
                }
                let asked = ep.models[0].clone();
                jobs.push((ep.id.clone(), ep.base_url.clone(), chain, asked, ep.models.clone()));
            }
        }
        let probes: Vec<_> = jobs
            .into_iter()
            .map(|(id, base_url, chain, asked, covers)| {
                let client = self.client.clone();
                let key = self.credentials.for_url(&base_url).map(str::to_string);
                async move {
                    let load = probe_backend(&chain, &client, &base_url, &asked, key.as_deref()).await;
                    (id, covers, load)
                }
            })
            .collect();
        let results: Vec<(String, Vec<String>, Load)> =
            stream::iter(probes).buffer_unordered(32).collect().await;
        let mut probed: HashMap<(String, String), Load> = HashMap::new();
        for (id, covers, load) in results {
            for m in covers {
                probed.insert((id.clone(), m), load);
            }
        }
        let known = probed.values().filter(|l| !matches!(l, Load::Unknown)).count();
        let probe_names: Vec<&str> = self
            .hive_probes
            .iter()
            .chain(&self.server_probes)
            .chain(&self.single_model_probes)
            .map(|p| p.name())
            .collect();
        tracing::debug!(?probe_names, known, total = probed.len(), "hive load refresh");
        *self.loads.write().await = probed;
        self.log_view().await;
    }

    /// Count one request against `base_url` until the guard is dropped.
    fn track_inflight(&self, base_url: &str) -> InflightGuard {
        *lock(&self.inflight).entry(base_url.to_string()).or_insert(0) += 1;
        InflightGuard {
            inflight: self.inflight.clone(),
            base_url: base_url.to_string(),
        }
    }

    /// Rank `base_url` last for [`FAILURE_COOLDOWN`].
    fn mark_failed(&self, base_url: &str) {
        lock(&self.cooldown).insert(base_url.to_string(), Instant::now() + FAILURE_COOLDOWN);
    }

    fn mark_healthy(&self, base_url: &str) {
        lock(&self.cooldown).remove(base_url);
    }

    /// Backends still cooling down after a failure (expired entries pruned).
    fn cooling(&self) -> HashSet<String> {
        let now = Instant::now();
        let mut cooldown = lock(&self.cooldown);
        cooldown.retain(|_, until| *until > now);
        cooldown.keys().cloned().collect()
    }

    /// Split `http://127.0.0.1:9037` into `("127.0.0.1", Some(9037), "")`;
    /// path-prefixed bases like `http://127.0.0.1:8180/general-stage1/v1`
    /// yield `("127.0.0.1", Some(8180), "/general-stage1/v1")` so the port
    /// still parses and the view can tell same-port prefixes apart.
    fn split_host_port(base_url: &str) -> (String, Option<u16>, String) {
        let no_scheme = base_url.split("://").last().unwrap_or(base_url);
        // Authority is up to the first `/`; the rest is the serving path.
        let (authority, path) = match no_scheme.find('/') {
            Some(i) => (&no_scheme[..i], no_scheme[i..].to_string()),
            None => (no_scheme, String::new()),
        };
        match authority.rfind(':') {
            Some(i) => (
                authority[..i].to_string(),
                authority[i + 1..].parse().ok(),
                path,
            ),
            None => (authority.to_string(), None, path),
        }
    }

    /// Per-model backends with the SAME effective loads the balancer routes
    /// on. Backs both the stdout view and `GET /api/hive/backends`.
    pub async fn backends_view(&self) -> BackendsView {
        let cooling = self.cooling();
        let endpoints = self.endpoints.read().await;
        let loads = self.loads.read().await;
        let inflight = lock(&self.inflight);
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
                let (ip, port, path) = Self::split_host_port(&ep.base_url);
                models.entry(m.clone()).or_default().push(BackendInfo {
                    ip,
                    port,
                    path,
                    load: num,
                    exact,
                    cooling: cooling.contains(&ep.base_url),
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

    /// Last `limit` query-log entries, newest first, reaching into rolled
    /// archives when the live file is short. Empty when no file-backed
    /// sink is configured.
    pub async fn recent_queries(&self, limit: usize) -> Vec<Value> {
        let Some(path) = self.logger.file_sink_path() else {
            return Vec::new();
        };
        // Show everything logged so far, including the caller's last query.
        self.logger.flush().await;
        read_recent(&path, limit).await
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
                let where_ = format!("{}:{port}{}", b.ip, b.path);
                if b.exact {
                    out.push_str(&format!("\n    {where_} load={}", b.load));
                } else {
                    out.push_str(&format!("\n    {where_} load=~{}", b.load));
                }
                if b.cooling {
                    out.push_str(" (failing, ranked last)");
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
    /// rotate round-robin. Backends that just failed a request go last
    /// (an unreachable backend reports no load, so it would otherwise look
    /// idle and be tried first). The proxy tries candidates in order and
    /// fails over to the next one when a backend is unreachable.
    pub async fn candidates(&self, model: &str) -> Vec<String> {
        let cooling = self.cooling();
        let endpoints = self.endpoints.read().await;
        let loads = self.loads.read().await;
        let mut ranked: Vec<(String, RankKey)> = {
            // Scoped: a std guard must not live across the `rr` await below.
            let inflight = lock(&self.inflight);
            endpoints
                .iter()
                .filter(|e| e.models.iter().any(|m| m == model))
                .map(|ep| {
                    let server = loads
                        .get(&(ep.id.clone(), model.to_string()))
                        .copied()
                        .unwrap_or(Load::Unknown);
                    let flying = inflight.get(&ep.base_url).copied().unwrap_or(0);
                    let (eff, _) = effective_load(server, flying);
                    (ep.base_url.clone(), (cooling.contains(&ep.base_url), eff))
                })
                .collect()
        };
        drop(loads);
        drop(endpoints);
        ranked.sort_by_key(|(_, n)| *n); // stable: ties keep discovery order
        let turn = {
            let mut rr = self.rr.lock().await;
            let counter = rr.entry(model.to_string()).or_insert(0);
            let turn = *counter;
            *counter = counter.wrapping_add(1);
            turn
        };
        let mut out = Vec::with_capacity(ranked.len());
        // Rotate each run of equal (cooling, effective load) independently.
        let mut i = 0;
        while i < ranked.len() {
            let key = ranked[i].1;
            let mut j = i + 1;
            while j < ranked.len() && ranked[j].1 == key {
                j += 1;
            }
            let group = &mut ranked[i..j];
            if group.len() > 1 {
                group.rotate_left(turn % group.len());
            }
            out.extend(group.iter().map(|(u, _)| u.clone()));
            i = j;
        }
        out
    }

    /// [`Hive::candidates`] minus backends that lead back to a visited
    /// hive: members whose id (or, from pre-id hives, whose base_url) is on
    /// the Via path, and hive members whose every route for `model` runs
    /// through a visited hive. A request that arrives with nothing left is
    /// a loop — the proxy answers 502 instead of forwarding forever.
    #[cfg(test)]
    pub async fn candidates_excluding(
        &self,
        model: &str,
        visited: &HashSet<String>,
    ) -> Vec<String> {
        let all = self.candidates(model).await;
        self.without_visited(model, all, visited).await
    }

    /// `candidates` (from [`Hive::candidates`]) minus those leading back
    /// to a visited hive (see [`Hive::candidates_excluding`]).
    async fn without_visited(
        &self,
        model: &str,
        candidates: Vec<String>,
        visited: &HashSet<String>,
    ) -> Vec<String> {
        let excluded: HashSet<String> = self
            .endpoints
            .read()
            .await
            .iter()
            .filter(|ep| {
                visited.contains(&ep.base_url)
                    || !routes_of(ep, model).iter().any(|p| loop_free(p, visited))
            })
            .map(|ep| ep.base_url.clone())
            .collect();
        candidates.into_iter().filter(|u| !excluded.contains(u)).collect()
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
    /// Serving path below host:port (`""` for bare roots,
    /// `"/general-stage1/v1"` for path-routed backends).
    #[serde(default)]
    pub path: String,
    pub load: u64,
    pub exact: bool,
    /// A recent request failed at the transport level: ranked last until
    /// the cooldown expires or a request succeeds.
    pub cooling: bool,
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

/// OpenAI model objects for everything this hive can serve to a caller
/// whose Via path is in `headers`. Built on the upstream's own object
/// (`max_model_len`, `root`, … survive) with the hive's fields on top:
/// `owned_by` names the member, and `hivllm.paths` carries the model's
/// loop-free routes so peer hives can filter one hop further. Models the
/// caller could only reach back through itself are left out.
async fn model_objects(hive: &Hive, headers: &HeaderMap) -> Vec<Value> {
    let endpoints = hive.snapshot().await;
    let mut avoid: HashSet<String> = parse_via(headers).into_iter().collect();
    avoid.insert(hive.own_id());
    let mut seen = HashSet::new();
    let mut data = Vec::new();
    for ep in &endpoints {
        for m in &ep.models {
            if !seen.insert(m.clone()) {
                continue;
            }
            let paths = Hive::advertised_paths(&endpoints, m, &avoid);
            if paths.is_empty() {
                continue;
            }
            let mut obj = match ep.model_meta.get(m) {
                Some(Value::Object(o)) => o.clone(),
                _ => serde_json::Map::new(),
            };
            obj.insert("id".into(), Value::String(m.clone()));
            obj.insert("object".into(), Value::String("model".into()));
            obj.entry("created").or_insert(Value::from(0));
            obj.insert("owned_by".into(), Value::String(ep.name.clone()));
            obj.insert(
                "hivllm".into(),
                serde_json::to_value(HivModelMeta { paths }).unwrap_or_default(),
            );
            data.push(Value::Object(obj));
        }
    }
    data.sort_by(|a, b| a["id"].as_str().cmp(&b["id"].as_str()));
    data
}

/// Aggregated model list (see [`model_objects`]).
pub async fn list_models(State(hive): State<Hive>, headers: HeaderMap) -> impl IntoResponse {
    Json(serde_json::json!({ "object": "list", "data": model_objects(&hive, &headers).await }))
}

/// `GET /v1/models/{id}`: one model object, or 404.
pub async fn get_model(
    State(hive): State<Hive>,
    axum::extract::Path(id): axum::extract::Path<String>,
    headers: HeaderMap,
) -> Response {
    match model_objects(&hive, &headers).await.into_iter().find(|m| m["id"] == id.as_str()) {
        Some(m) => Json(m).into_response(),
        None => json_error(StatusCode::NOT_FOUND, format!("model `{id}` not found in hive")),
    }
}

#[derive(Debug, Deserialize, Default)]
pub struct ModelQuery {
    /// `?model=` override. Body `model` field takes precedence if both set?
    /// We let the query param win when present (explicit routing request).
    pub model: Option<String>,
}


/// Default consecutive failed scans before a member leaves.
pub const DEFAULT_DROP_AFTER: u32 = 3;

/// Default largest request body: room for long contexts and a few
/// base64 images.
pub const DEFAULT_MAX_BODY: usize = 64 * 1024 * 1024;

/// Upstream response headers to hand the client: everything but
/// hop-by-hop headers (this hop's framing is axum's), `content-length`
/// (re-computed), the upstream's CORS policy (the hive enforces its own —
/// a backend's `access-control-allow-origin: *` must not bypass it) and
/// its hive id (replaced by ours). Upstreams that omit `content-type` get
/// `text/event-stream` for successful streams, JSON otherwise.
fn client_headers(resp: &reqwest::Response, sse: bool) -> HeaderMap {
    const DROP: &[&str] = &[
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "content-length",
        HIVE_ID_HEADER,
    ];
    let mut out = HeaderMap::new();
    for (name, value) in resp.headers() {
        let n = name.as_str();
        if DROP.contains(&n) || n.starts_with("access-control-") {
            continue;
        }
        out.append(name.clone(), value.clone());
    }
    let fallback = if sse { "text/event-stream" } else { "application/json" };
    out.entry(axum::http::header::CONTENT_TYPE)
        .or_insert(axum::http::HeaderValue::from_static(fallback));
    if sse {
        out.entry(axum::http::header::CACHE_CONTROL)
            .or_insert(axum::http::HeaderValue::from_static("no-cache"));
    }
    out
}

/// Upstream statuses worth another candidate: the backend is rate
/// limiting (429), broken (500), a bad gateway — e.g. a downstream hive's
/// `loop detected` — (502), or unavailable / still loading (503). Never
/// 504 (the generation may still be running downstream) nor other 4xx
/// (the client's mistake, every backend would repeat it).
fn retryable(status: StatusCode) -> bool {
    matches!(
        status,
        StatusCode::TOO_MANY_REQUESTS
            | StatusCode::INTERNAL_SERVER_ERROR
            | StatusCode::BAD_GATEWAY
            | StatusCode::SERVICE_UNAVAILABLE
    )
}

/// Most routes advertised per model: shortest first, so a dense mesh
/// can't blow up the model list.
const MAX_ADVERTISED_PATHS: usize = 8;

/// Hive-id paths from this hive through `ep` to a real backend of
/// `model`: `[[]]` for a plain backend (served directly), the member's
/// recorded paths for a hive (a single opaque hop if none recorded).
fn routes_of(ep: &DiscoveredEndpoint, model: &str) -> Vec<Vec<String>> {
    match &ep.hive_id {
        None => vec![Vec::new()],
        Some(hid) => member_paths(hid, ep, model),
    }
}

fn member_paths(hid: &str, ep: &DiscoveredEndpoint, model: &str) -> Vec<Vec<String>> {
    ep.paths
        .get(model)
        .cloned()
        .unwrap_or_else(|| vec![vec![hid.to_string()]])
}

fn loop_free(path: &[String], avoid: &HashSet<String>) -> bool {
    !path.iter().any(|h| avoid.contains(h))
}

/// Random instance id: unique per process (the std hasher is seeded from
/// OS randomness), so two hives never share one even on the same port.
fn new_hive_id() -> String {
    use std::hash::{BuildHasher, Hasher};
    let mut h = std::collections::hash_map::RandomState::new().build_hasher();
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or_default();
    h.write_u128(nanos);
    h.write_u32(std::process::id());
    format!("hive-{:016x}", h.finish())
}

/// Whether `id` can be a hive id: a header value that can't be confused
/// with Via-path separators.
pub fn valid_hive_id(id: &str) -> bool {
    !id.is_empty() && id.chars().all(|c| c.is_ascii_graphic() && c != ',')
}

/// Every hive route, each response stamped with this hive's id.
pub fn routes(hive: Hive) -> axum::Router {
    use axum::routing::{get, post};
    let id = axum::http::HeaderValue::from_str(&hive.own_id).expect("valid hive id");
    let body_limit = match hive.max_body {
        Some(max) => axum::extract::DefaultBodyLimit::max(max),
        None => axum::extract::DefaultBodyLimit::disable(),
    };
    axum::Router::new()
        .route("/health", get(health))
        .route("/load", get(server_load))
        .route("/v1/models", get(list_models))
        .route("/v1/models/:id", get(get_model))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/completions", post(completions))
        .route("/v1/embeddings", post(embeddings))
        .route("/v1/responses", model_route("responses", "/v1/responses"))
        .route("/v1/rerank", model_route("rerank", "/v1/rerank"))
        .route("/v2/rerank", model_route("rerank", "/v2/rerank"))
        .route("/rerank", model_route("rerank", "/rerank"))
        .route("/v1/score", model_route("score", "/v1/score"))
        .route("/score", model_route("score", "/score"))
        .route("/pooling", model_route("pooling", "/pooling"))
        .route("/classify", model_route("classify", "/classify"))
        .route("/tokenize", model_route("tokenize", "/tokenize"))
        .route("/detokenize", model_route("detokenize", "/detokenize"))
        .route("/v1/audio/speech", model_route("speech", "/v1/audio/speech"))
        .route(
            "/v1/audio/transcriptions",
            model_route("transcriptions", "/v1/audio/transcriptions"),
        )
        .route(
            "/v1/audio/translations",
            model_route("translations", "/v1/audio/translations"),
        )
        .route("/api/hive/endpoints", get(list_endpoints))
        .route("/api/hive/backends", get(list_backends))
        .route("/api/hive/queries", get(list_queries))
        .with_state(hive)
        .layer(body_limit)
        .layer(axum::middleware::map_response(move |mut resp: Response| {
            let id = id.clone();
            async move {
                resp.headers_mut().insert(HIVE_ID_HEADER, id);
                resp
            }
        }))
}

/// Upstream connect timeout default: a dead host fails over in seconds.
pub const DEFAULT_CONNECT_TIMEOUT: Duration = Duration::from_secs(5);
/// Upstream read (idle) timeout default. Non-streaming backends stay
/// silent until the whole generation is done, so this also caps how long
/// a non-streaming generation may take.
pub const DEFAULT_READ_TIMEOUT: Duration = Duration::from_secs(600);
/// How long a backend that failed a request stays ranked last.
const FAILURE_COOLDOWN: Duration = Duration::from_secs(15);

fn upstream_client(connect: Duration, read: Option<Duration>) -> reqwest::Client {
    let mut builder = reqwest::Client::builder().connect_timeout(connect);
    if let Some(read) = read {
        builder = builder.read_timeout(read);
    }
    builder.build().expect("reqwest client")
}

/// Poison-tolerant lock: these maps hold plain counters, always valid.
fn lock<T>(m: &StdMutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// One request counted against a backend's in-flight load; the count is
/// released on drop, whichever way the request ends.
struct InflightGuard {
    inflight: Arc<StdMutex<HashMap<String, u64>>>,
    base_url: String,
}

impl Drop for InflightGuard {
    fn drop(&mut self) {
        let mut inflight = lock(&self.inflight);
        if let Some(n) = inflight.get_mut(&self.base_url) {
            *n = n.saturating_sub(1);
            if *n == 0 {
                inflight.remove(&self.base_url); // no dead entries piling up
            }
        }
    }
}

fn json_error(status: StatusCode, message: String) -> Response {
    let payload = serde_json::json!({
        "error": { "message": message, "type": "hive_error" }
    });
    (status, Json(payload)).into_response()
}

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

/// A routable request: its model, what to forward, and what to log.
struct ParsedRequest {
    model: String,
    /// Logged form of the request (the JSON body, or a form description).
    request: Value,
    body: Bytes,
    content_type: String,
    stream: bool,
}

/// Read the model and forwarding details from a JSON body, or from a
/// `multipart/form-data` upload (audio). `?model=` wins over the body;
/// for JSON the forwarded body is rewritten so upstreams see it too
/// (multipart bodies are forwarded untouched). Errors carry what to log.
fn parse_request(
    query: &ModelQuery,
    headers: &HeaderMap,
    body: Bytes,
) -> Result<ParsedRequest, (String, Value)> {
    let content_type = headers
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("application/json")
        .to_string();
    let query_model = query.model.clone().filter(|m| !m.is_empty());
    if let Some(boundary) = crate::multipart::boundary(&content_type) {
        let parts = crate::multipart::parse(&body, &boundary);
        let request = crate::multipart::describe(&parts);
        let Some(model) =
            query_model.or_else(|| crate::multipart::field(&parts, "model").map(str::to_string))
        else {
            return Err(("missing `model`: set form field `model` or ?model=...".into(), request));
        };
        let stream = crate::multipart::field(&parts, "stream") == Some("true");
        return Ok(ParsedRequest {
            model,
            request,
            body,
            content_type,
            stream,
        });
    }
    let mut value: Value = serde_json::from_slice(&body)
        .map_err(|_| ("invalid JSON body".to_string(), Value::Null))?;
    let Some(model) = query_model.or_else(|| value.get("model")?.as_str().map(str::to_string))
    else {
        let msg = "missing `model`: set JSON body {\"model\": \"...\"} or ?model=...";
        return Err((msg.to_string(), value));
    };
    if let Some(obj) = value.as_object_mut() {
        obj.insert("model".to_string(), Value::String(model.clone()));
    }
    let stream = value.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let body = Bytes::from(serde_json::to_vec(&value).unwrap_or_default());
    Ok(ParsedRequest {
        model,
        request: value,
        body,
        content_type: "application/json".to_string(),
        stream,
    })
}

/// `POST` route forwarded as-is to a backend serving the request's model
/// (JSON body or multipart form), logged under `label`.
fn model_route(label: &'static str, path: &'static str) -> axum::routing::MethodRouter<Hive> {
    axum::routing::post(
        move |State(hive): State<Hive>, Query(query): Query<ModelQuery>, headers: HeaderMap, body: Bytes| {
            proxy_by_model(hive, query, headers, body, label, path)
        },
    )
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
    let parsed = match parse_request(&query, &headers, body) {
        Ok(p) => p,
        Err((msg, request)) => {
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
                    request,
                    response: None,
                },
                start,
            )
            .await;
            return json_error(StatusCode::BAD_REQUEST, msg);
        }
    };
    let ParsedRequest {
        model,
        request: value,
        body: fwd_body,
        content_type,
        stream: stream_mode,
    } = parsed;

    let via_path = parse_via(&headers);
    let visited: HashSet<String> = via_path.iter().cloned().collect();
    // One routing decision (the round-robin turn advances once), then
    // the loop filter.
    let all = hive.candidates(&model).await;
    let served_somewhere = !all.is_empty();
    let candidates = hive.without_visited(&model, all, &visited).await;
    if candidates.is_empty() {
        // Nothing left to try: either the model is unknown (404), or every
        // backend is already on the Via path — a hive-of-hives loop (502).
        if served_somewhere {
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


    // The client's token is only passed on when the operator opted in:
    // otherwise it would reach every candidate (random discovered ports,
    // remote static hosts, containers), not just the one it was meant for.
    let auth: Option<String> = headers
        .get(axum::http::header::AUTHORIZATION)
        .filter(|_| hive.forward_auth)
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
    // the next one instead of failing the request. A timeout does NOT fail
    // over: the backend accepted the request and may still be generating,
    // so retrying elsewhere would run the same generation twice.
    let mut last_error = String::new();
    for (attempt, upstream) in candidates.iter().enumerate() {
        let is_last = attempt + 1 == candidates.len();
        let url = join_upstream_path(upstream, upstream_path);
        let mut req = hive
            .client
            .post(&url)
            .header(axum::http::header::CONTENT_TYPE, content_type.as_str())
            .header(VIA_HEADER, fwd_via.clone())
            .body(fwd_body.clone());
        if let Some(key) = hive.credentials.for_url(upstream) {
            req = req.bearer_auth(key);
        } else if let Some(a) = &auth {
            req = req.header(reqwest::header::AUTHORIZATION, a.clone());
        }

        // Counted before sending: non-streaming backends only answer
        // headers once generation is done, so counting after `send` would
        // miss the whole busy period. Dropped on every exit path —
        // including a client disconnect cancelling this handler.
        let inflight = hive.track_inflight(upstream);
        let resp = match req.send().await {
            Ok(r) => {
                hive.mark_healthy(upstream);
                r
            }
            Err(e) if e.is_timeout() && !e.is_connect() => {
                let msg = format!("upstream `{upstream}` timed out: {e}");
                tracing::warn!(%url, error = %e, "upstream timed out, not retrying elsewhere");
                log_query(
                    &hive,
                    route,
                    Outcome {
                        model,
                        upstream: Some(upstream.clone()),
                        stream: stream_mode,
                        status: StatusCode::GATEWAY_TIMEOUT,
                        usage: None,
                        error: Some(msg.clone()),
                        request: value,
                        response: None,
                    },
                    start,
                )
                .await;
                return json_error(StatusCode::GATEWAY_TIMEOUT, msg);
            }
            Err(e) => {
                tracing::warn!(%url, error = %e, "upstream failed, trying next hive member");
                hive.mark_failed(upstream);
                last_error = e.to_string();
                continue;
            }
        };

        let status = StatusCode::from_u16(resp.status().as_u16())
            .unwrap_or(StatusCode::BAD_GATEWAY);

        // Overloaded / broken / unavailable backends produced nothing:
        // try the next candidate. The last one's answer is passed through
        // as-is, so the client sees the real error.
        if !is_last && retryable(status) {
            let detail = truncate_text(&resp.text().await.unwrap_or_default(), 300);
            tracing::warn!(%url, %status, %detail, "upstream error, trying next hive member");
            if matches!(status, StatusCode::SERVICE_UNAVAILABLE | StatusCode::TOO_MANY_REQUESTS) {
                hive.mark_failed(upstream);
            }
            last_error = format!("`{upstream}` answered {status}: {detail}");
            continue;
        }

        if stream_mode {
            // Tee the SSE bytes: client streams untouched while we rebuild the
            // response for the log. The entry is emitted when the stream ends,
            // so logged latency covers the full generation.
            let resp_headers = client_headers(&resp, status.is_success());
            let acc = Arc::new(std::sync::Mutex::new(StreamAcc::default()));
            let (tx, rx) = oneshot::channel::<StreamSummary>();
            let body = Body::from_stream(TeeStream::new(resp.bytes_stream(), acc.clone(), tx));
            let hive2 = hive.clone();
            let (model, value, upstream) = (model.clone(), value.clone(), upstream.clone());
            tokio::spawn(async move {
                let (summary, ended) = match rx.await {
                    Ok(s) => (s, None),
                    // Body dropped before its end: the client went away.
                    Err(_) => (
                        acc.lock().map(|a| a.summary()).unwrap_or_default(),
                        Some("stream ended before completion".to_string()),
                    ),
                };
                let error = summary
                    .error
                    .clone()
                    .map(|e| format!("stream broken: {e}"))
                    .or(ended)
                    .or_else(|| (!status.is_success()).then(|| format!("upstream status {status}")));
                log_query(
                    &hive2,
                    route,
                    Outcome {
                        model,
                        upstream: Some(upstream.clone()),
                        stream: true,
                        status,
                        usage: summary.usage,
                        error,
                        request: value,
                        response: Some(summary.response),
                    },
                    start,
                )
                .await;
                drop(inflight);
            });
            return (status, resp_headers, body).into_response();
        }

        let resp_headers = client_headers(&resp, false);
        // The request was processed: a failed body read is reported, not
        // retried elsewhere (and never passed off as an empty success).
        let bytes = match resp.bytes().await {
            Ok(b) => b,
            Err(e) => {
                let (status, what) = if e.is_timeout() {
                    (StatusCode::GATEWAY_TIMEOUT, "timed out")
                } else {
                    (StatusCode::BAD_GATEWAY, "failed")
                };
                let msg = format!("reading response from `{upstream}` {what}: {e}");
                tracing::warn!(%url, error = %e, "upstream response body {what}");
                log_query(
                    &hive,
                    route,
                    Outcome {
                        model,
                        upstream: Some(upstream.clone()),
                        stream: false,
                        status,
                        usage: None,
                        error: Some(msg.clone()),
                        request: value,
                        response: None,
                    },
                    start,
                )
                .await;
                return json_error(status, msg);
            }
        };
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
        drop(inflight);
        return (status, resp_headers, body_from_bytes(bytes)).into_response();
    }

    // Every candidate failed (the last one at the transport level).
    let msg = format!(
        "no hive member could serve model `{model}` ({} tried), last error: {last_error}",
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
            hive_id: None,
            paths: HashMap::new(),
            model_meta: HashMap::new(),
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
            std::mem::forget(hive.track_inflight("http://127.0.0.1:9001")); // held open
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
        std::mem::forget(hive.track_inflight("http://127.0.0.1:9003")); // held open
        std::mem::forget(hive.track_inflight("http://127.0.0.1:9003")); // held open
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
        std::mem::forget(hive.track_inflight("http://127.0.0.1:9002")); // held open
        std::mem::forget(hive.track_inflight("http://127.0.0.1:9002")); // held open
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
        let dir = crate::logging::test_dir("recent");
        let path = dir.join("q.jsonl");
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
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn recent_queries_empty_without_file_sink() {
        assert!(Hive::new(0).recent_queries(10).await.is_empty());
    }

    #[tokio::test]
    async fn aggregate_load_is_what_the_next_request_faces() {
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
        assert_eq!(hive.aggregate_load(None).await, (10, true));
    }

    #[tokio::test]
    async fn aggregate_load_empty_hive_is_zero() {
        assert_eq!(Hive::new(0).aggregate_load(None).await, (0, false));
    }

    #[tokio::test]
    async fn measured_idle_members_make_the_hive_exactly_idle() {
        // One saturated backend next to a measured-idle one: a new request
        // lands on the idle one, so upstream hives should see exactly 0.
        let hive = hive_with(
            vec![ep("idle", 9001, &["m"]), ep("hot", 9002, &["m"])],
            vec![(("idle", "m"), Load::Measured(0)), (("hot", "m"), Load::Measured(32))],
        )
        .await;
        assert_eq!(hive.aggregate_load(None).await, (0, true));
    }

    #[tokio::test]
    async fn unverified_minimum_stays_approximate() {
        // `/load`'s bare 0 (idle or untracked?) and an untracked member:
        // the next request may land on either, so the answer is ~0 —
        // never passed off as exact, even next to an exact 32.
        let hive = hive_with(
            vec![
                ep("maybe", 9001, &["m"]),
                ep("hot", 9002, &["m"]),
                ep("mystery", 9003, &["m"]),
            ],
            vec![(("maybe", "m"), Load::Known(0)), (("hot", "m"), Load::Known(32))],
        )
        .await;
        assert_eq!(hive.aggregate_load(None).await, (0, false));
        // Ties between an exact and an approximate minimum prefer exact.
        let tie = hive_with(
            vec![ep("a", 9001, &["m"]), ep("b", 9002, &["m"])],
            vec![(("a", "m"), Load::Known(3)), (("b", "m"), Load::Known(0))],
        )
        .await;
        for _ in 0..3 {
            std::mem::forget(tie.track_inflight("http://127.0.0.1:9002")); // held open
        }
        assert_eq!(tie.aggregate_load(None).await, (3, true));
    }

    #[tokio::test]
    async fn aggregate_load_scopes_to_one_model() {
        // "hot" is pressured on m2 only; "cold" serves m1 measured-idle.
        // ?model=m2 must report hot's 12, not something smeared with m1.
        let hive = hive_with(
            vec![
                ep("hot", 9001, &["m1", "m2"]),
                ep("cold", 9002, &["m1"]),
            ],
            vec![
                (("hot", "m1"), Load::Measured(4)),
                (("hot", "m2"), Load::Measured(12)),
                (("cold", "m1"), Load::Measured(0)),
            ],
        )
        .await;
        assert_eq!(hive.aggregate_load(Some("m2")).await, (12, true));
        assert_eq!(hive.aggregate_load(Some("m1")).await, (0, true));
        assert_eq!(hive.aggregate_load(Some("nope")).await, (0, false));
    }

    #[tokio::test]
    async fn load_endpoint_accepts_model_param() {
        let hive = hive_with(
            vec![ep("hot", 9001, &["m1", "m2"])],
            vec![(("hot", "m2"), Load::Known(12))],
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
                hive_id: None,
                paths: HashMap::new(),
                model_meta: HashMap::new(),
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
        let probe = hive.clone();

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
        // The failure is remembered: the dead backend no longer looks idle.
        assert_eq!(probe.candidates("m").await, urls(&[live_port, dead_port]));
        assert!(probe.render_view().await.contains("(failing, ranked last)"));
    }

    /// Serve `hive` on an ephemeral port with the proxy routes; returns its URL.
    async fn serve_proxy(hive: Hive) -> String {
        let app = routes(hive);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        url
    }

    /// Mock backend answering a canned completion after `delay`; counts hits
    /// and records the Authorization header it last received.
    async fn slow_backend(
        delay: Duration,
    ) -> (u16, Arc<std::sync::atomic::AtomicUsize>, Arc<StdMutex<Option<String>>>) {
        let hits = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let auth = Arc::new(StdMutex::new(None));
        let (h, a) = (hits.clone(), auth.clone());
        let mock = Router::new().route(
            "/v1/chat/completions",
            post(move |headers: HeaderMap| {
                let (h, a) = (h.clone(), a.clone());
                async move {
                    h.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    *lock(&a) = headers
                        .get("authorization")
                        .map(|v| v.to_str().unwrap().to_string());
                    tokio::time::sleep(delay).await;
                    Json(serde_json::json!({"id": "slow", "choices": []}))
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move { axum::serve(listener, mock).await.unwrap() });
        (port, hits, auth)
    }

    fn inflight_of(hive: &Hive, port: u16) -> u64 {
        lock(&hive.inflight)
            .get(&format!("http://127.0.0.1:{port}"))
            .copied()
            .unwrap_or(0)
    }

    #[tokio::test]
    async fn inflight_counts_while_non_streaming_upstream_generates() {
        // Non-streaming backends send headers only when done: the request
        // must count as in flight for the whole generation, not after it.
        let (port, hits, _) = slow_backend(Duration::from_millis(1500)).await;
        let hive = hive_with(vec![ep("a", port, &["m"])], vec![]).await;
        let url = serve_proxy(hive.clone()).await;
        let req = tokio::spawn(async move {
            reqwest::Client::new()
                .post(format!("{url}/v1/chat/completions"))
                .json(&serde_json::json!({"model": "m", "messages": []}))
                .send()
                .await
                .unwrap()
                .status()
        });
        // Wait until the backend is working on it, then look.
        while hits.load(std::sync::atomic::Ordering::SeqCst) == 0 {
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert_eq!(inflight_of(&hive, port), 1);
        assert_eq!(req.await.unwrap(), 200);
        assert_eq!(inflight_of(&hive, port), 0);
        assert!(lock(&hive.inflight).is_empty(), "idle backends leave no entry");
    }

    #[tokio::test]
    async fn client_disconnect_releases_inflight() {
        let (port, _, _) = slow_backend(Duration::from_secs(3)).await;
        let hive = hive_with(vec![ep("a", port, &["m"])], vec![]).await;
        let url = serve_proxy(hive.clone()).await;
        // Client gives up long before the backend answers.
        let gave_up = reqwest::Client::new()
            .post(format!("{url}/v1/chat/completions"))
            .timeout(Duration::from_millis(300))
            .json(&serde_json::json!({"model": "m", "messages": []}))
            .send()
            .await;
        assert!(gave_up.is_err());
        let mut released = false;
        for _ in 0..40 {
            if inflight_of(&hive, port) == 0 {
                released = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        assert!(released, "in-flight count leaked after client disconnect");
    }

    #[tokio::test]
    async fn timeout_is_answered_504_without_retrying_elsewhere() {
        // Slow backend ranks first (lower load); timing out on it must not
        // re-run the generation on the fast one.
        let (slow, slow_hits, _) = slow_backend(Duration::from_secs(3)).await;
        let (fast, fast_hits, _) = slow_backend(Duration::ZERO).await;
        let hive = hive_with(
            vec![ep("slow", slow, &["m"]), ep("fast", fast, &["m"])],
            vec![(("slow", "m"), Load::Known(1)), (("fast", "m"), Load::Known(5))],
        )
        .await
        .with_timeouts(Duration::from_secs(1), Some(Duration::from_millis(300)));
        let url = serve_proxy(hive).await;
        let resp = reqwest::Client::new()
            .post(format!("{url}/v1/chat/completions"))
            .json(&serde_json::json!({"model": "m", "messages": []}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 504);
        assert_eq!(slow_hits.load(std::sync::atomic::Ordering::SeqCst), 1);
        assert_eq!(fast_hits.load(std::sync::atomic::Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn authorization_is_forwarded_only_when_opted_in() {
        for forward in [false, true] {
            let (port, _, seen) = slow_backend(Duration::ZERO).await;
            let hive = hive_with(vec![ep("a", port, &["m"])], vec![])
                .await
                .with_forward_auth(forward);
            let url = serve_proxy(hive).await;
            let resp = reqwest::Client::new()
                .post(format!("{url}/v1/chat/completions"))
                .bearer_auth("secret")
                .json(&serde_json::json!({"model": "m", "messages": []}))
                .send()
                .await
                .unwrap();
            assert_eq!(resp.status(), 200);
            let expected = forward.then(|| "Bearer secret".to_string());
            assert_eq!(*lock(&seen), expected, "forward_auth={forward}");
        }
    }

    #[tokio::test]
    async fn hive_ping_pong_terminates_with_loop_detected() {
        // Two hives that strictly prefer each other, each reaching the other
        // by an address unrelated to its id (`localhost` here, a LAN name
        // across hosts). Ids are learned by probing, as in production.
        // A forwards to B with Via:[A]; B's only candidate is A, already
        // visited, so B answers 502 — a fast error, not an endless loop.
        let la = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let lb = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url_a = format!("http://localhost:{}", la.local_addr().unwrap().port());
        let url_b = format!("http://localhost:{}", lb.local_addr().unwrap().port());
        let (ha, hb) = (Hive::new(0), Hive::new(0));
        assert_ne!(ha.own_id(), hb.own_id());
        for (l, h) in [(la, ha.clone()), (lb, hb.clone())] {
            let app = routes(h);
            tokio::spawn(async move { axum::serve(l, app).await.unwrap() });
        }
        // Both advertise model "m" through a local stub member so each has
        // something to list; the stubs are then swapped for the peer.
        let stub = |h: &Hive| {
            let h = h.clone();
            async move {
                *h.endpoints.write().await = vec![ep("stub", 1, &["m"])];
            }
        };
        stub(&ha).await;
        stub(&hb).await;
        let peer_b = probe_static(&ha.client, std::slice::from_ref(&url_b), &ha.own_id(), &Credentials::default()).await;
        let peer_a = probe_static(&hb.client, std::slice::from_ref(&url_a), &hb.own_id(), &Credentials::default()).await;
        assert_eq!(peer_b[0].hive_id.as_deref(), Some(hb.own_id().as_str()));
        assert_eq!(peer_a[0].hive_id.as_deref(), Some(ha.own_id().as_str()));
        *ha.endpoints.write().await = peer_b;
        *hb.endpoints.write().await = peer_a;

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

    #[tokio::test]
    async fn hive_drops_itself_when_reached_by_another_address() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let hive = hive_with(vec![ep("stub", 1, &["m"])], vec![]).await;
        let app = routes(hive.clone());
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        // e.g. `--static-backends http://localhost:8335` on the hive itself.
        let found = probe_static(&hive.client, &[format!("http://localhost:{port}")], &hive.own_id(), &Credentials::default()).await;
        assert_eq!(found.len(), 1);
        assert!(hive.without_self(found).is_empty());
    }

    #[test]
    fn hive_ids_are_unique_and_valid() {
        let (a, b) = (new_hive_id(), new_hive_id());
        assert_ne!(a, b);
        assert!(valid_hive_id(&a));
        assert!(!valid_hive_id("a,b"));
        assert!(!valid_hive_id("a b"));
        assert!(!valid_hive_id(""));
    }

    /// Mock backend whose `/v1/models` lists "m" while `up` is true.
    async fn switchable_backend(up: Arc<std::sync::atomic::AtomicBool>) -> String {
        let mock = Router::new()
            .route(
                "/v1/models",
                get(move || {
                    let up = up.clone();
                    async move {
                        let data = if up.load(std::sync::atomic::Ordering::SeqCst) {
                            serde_json::json!([{"id": "m"}])
                        } else {
                            serde_json::json!([])
                        };
                        Json(serde_json::json!({ "data": data }))
                    }
                }),
            )
            .route(
                "/v1/chat/completions",
                post(|| async { Json(serde_json::json!({"id": "real", "choices": []})) }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move { axum::serve(listener, mock).await.unwrap() });
        url
    }

    /// Serve a hive with every route; returns its URL.
    async fn serve_hive(hive: &Hive) -> String {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://localhost:{}", listener.local_addr().unwrap().port());
        let app = routes(hive.clone());
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        url
    }

    /// One discovery pass over `statics` only (no localhost port scan).
    async fn rescan(hive: &Hive, statics: &[String]) {
        let found = probe_static(&hive.client, statics, &hive.own_id(), &Credentials::default()).await;
        hive.set_endpoints(found).await;
    }

    async fn serves_m(hive: &Hive) -> bool {
        hive.snapshot().await.iter().any(|e| e.models.iter().any(|m| m == "m"))
    }

    #[tokio::test]
    async fn dead_model_does_not_survive_in_a_hive_ring() {
        // Ring A → B → C → A (each probes the next); only A has the real
        // backend R. Without path vectors each hive would keep re-learning
        // "m" from its neighbour forever after R drops it.
        let up = Arc::new(std::sync::atomic::AtomicBool::new(true));
        let r = switchable_backend(up.clone()).await;
        let (a, b, c) = (Hive::new(0), Hive::new(0), Hive::new(0));
        let (ua, ub, uc) = (serve_hive(&a).await, serve_hive(&b).await, serve_hive(&c).await);
        let round = || async {
            rescan(&a, &[r.clone(), ub.clone()]).await;
            rescan(&b, std::slice::from_ref(&uc)).await;
            rescan(&c, std::slice::from_ref(&ua)).await;
        };
        for _ in 0..3 {
            round().await;
        }
        // "m" travelled around: B reaches it through C then A.
        assert!(serves_m(&b).await);
        let b_paths = &b.snapshot().await[0].paths["m"];
        assert_eq!(b_paths, &vec![vec![c.own_id(), a.own_id()]]);
        // A never learns its own model back through B.
        assert_eq!(a.candidates("m").await, vec![r.clone()]);
        // And requests follow the ring to the real backend.
        let resp = reqwest::Client::new()
            .post(format!("{ub}/v1/chat/completions"))
            .json(&serde_json::json!({"model": "m", "messages": []}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        assert_eq!(resp.json::<Value>().await.unwrap()["id"], "real");

        // R stops serving "m". A drops it on its next scan; stale routes
        // elsewhere die one hop per round (B scans C before C rescans A,
        // so B's two-hop route lingers one extra round) — then never
        // come back.
        up.store(false, std::sync::atomic::Ordering::SeqCst);
        round().await;
        assert!(!serves_m(&a).await, "the hive that lost the backend must drop it at once");
        assert!(!serves_m(&c).await);
        round().await;
        for _ in 0..3 {
            for (name, h) in [("a", &a), ("b", &b), ("c", &c)] {
                assert!(!serves_m(h).await, "hive {name} still serves a dead model");
            }
            round().await;
        }
    }

    #[test]
    fn advertised_paths_skip_loops_and_stay_bounded() {
        let hive_member = |hid: &str, paths: Vec<Vec<&str>>| DiscoveredEndpoint {
            id: hid.into(),
            name: hid.into(),
            base_url: format!("http://{hid}"),
            models: vec!["m".into()],
            source: "test".into(),
            hive_id: Some(hid.into()),
            model_meta: HashMap::new(),
            paths: HashMap::from([(
                "m".to_string(),
                paths
                    .into_iter()
                    .map(|p| p.into_iter().map(String::from).collect())
                    .collect(),
            )]),
        };
        let eps = vec![
            ep("direct", 9001, &["m"]),
            hive_member("x", vec![vec!["x"], vec!["x", "req"]]),
        ];
        let avoid: HashSet<String> = ["req".to_string(), "me".to_string()].into();
        // Direct first, then x; the route through the requester is dropped.
        assert_eq!(
            Hive::advertised_paths(&eps, "m", &avoid),
            vec![Vec::<String>::new(), vec!["x".to_string()]]
        );
        let many: Vec<DiscoveredEndpoint> = (0..20)
            .map(|i| hive_member(&format!("h{i}"), vec![vec![&format!("h{i}")]]))
            .collect();
        assert_eq!(Hive::advertised_paths(&many, "m", &avoid).len(), MAX_ADVERTISED_PATHS);
    }

    /// Mock backend answering every completion with `status` (an error
    /// body unless 200; SSE when the request streams). Counts hits.
    async fn status_backend(status: u16) -> (u16, Arc<std::sync::atomic::AtomicUsize>) {
        let hits = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let h = hits.clone();
        let mock = Router::new().route(
            "/v1/chat/completions",
            post(move |Json(body): Json<Value>| {
                let h = h.clone();
                async move {
                    h.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    let code = StatusCode::from_u16(status).unwrap();
                    if status != 200 {
                        let err = serde_json::json!({"error": {"message": format!("mock {status}")}});
                        return (code, Json(err)).into_response();
                    }
                    if body["stream"] == true {
                        let sse = "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n";
                        return ([("content-type", "text/event-stream")], sse).into_response();
                    }
                    Json(serde_json::json!({"id": "ok", "choices": []})).into_response()
                }
            }),
        )
        // Mocks accept anything: size limits under test are the hive's.
        .layer(axum::extract::DefaultBodyLimit::disable());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move { axum::serve(listener, mock).await.unwrap() });
        (port, hits)
    }

    /// Hive over `first` (ranked first by load) then `second`.
    async fn two_backends(first: u16, second: u16) -> Hive {
        hive_with(
            vec![ep("first", first, &["m"]), ep("second", second, &["m"])],
            vec![(("first", "m"), Load::Known(1)), (("second", "m"), Load::Known(5))],
        )
        .await
    }

    async fn chat(url: &str, stream: bool) -> reqwest::Response {
        reqwest::Client::new()
            .post(format!("{url}/v1/chat/completions"))
            .json(&serde_json::json!({"model": "m", "messages": [], "stream": stream}))
            .send()
            .await
            .unwrap()
    }

    fn hits(n: &std::sync::atomic::AtomicUsize) -> usize {
        n.load(std::sync::atomic::Ordering::SeqCst)
    }

    #[tokio::test]
    async fn retryable_errors_fail_over_to_the_next_backend() {
        for (code, cools) in [(503, true), (429, true), (500, false), (502, false)] {
            let (bad, bad_hits) = status_backend(code).await;
            let (good, good_hits) = status_backend(200).await;
            let hive = two_backends(bad, good).await;
            let url = serve_proxy(hive.clone()).await;
            let resp = chat(&url, false).await;
            assert_eq!(resp.status(), 200, "{code}");
            assert_eq!((hits(&bad_hits), hits(&good_hits)), (1, 1), "{code}");
            // Overload signals rank the backend last; request-specific
            // errors don't.
            let order = hive.candidates("m").await;
            let bad_first = order[0] == format!("http://127.0.0.1:{bad}");
            assert_eq!(!bad_first, cools, "{code}: {order:?}");
        }
    }

    #[tokio::test]
    async fn streams_fail_over_before_any_byte_is_sent() {
        let (bad, _) = status_backend(503).await;
        let (good, _) = status_backend(200).await;
        let url = serve_proxy(two_backends(bad, good).await).await;
        let resp = chat(&url, true).await;
        assert_eq!(resp.status(), 200);
        assert!(resp.text().await.unwrap().contains("\"ok\""));
    }

    #[tokio::test]
    async fn client_errors_and_gateway_timeouts_are_not_retried() {
        for code in [400, 404, 504] {
            let (bad, _) = status_backend(code).await;
            let (good, good_hits) = status_backend(200).await;
            let url = serve_proxy(two_backends(bad, good).await).await;
            let resp = chat(&url, false).await;
            assert_eq!(resp.status().as_u16(), code);
            assert_eq!(hits(&good_hits), 0, "{code} must not be retried");
        }
    }

    #[tokio::test]
    async fn last_candidate_error_is_passed_through() {
        let (a, a_hits) = status_backend(503).await;
        let (b, b_hits) = status_backend(500).await;
        let url = serve_proxy(two_backends(a, b).await).await;
        let resp = chat(&url, false).await;
        assert_eq!(resp.status(), 500);
        let body: Value = resp.json().await.unwrap();
        assert_eq!(body["error"]["message"], "mock 500");
        assert_eq!((hits(&a_hits), hits(&b_hits)), (1, 1));
    }

    fn big_chat(bytes: usize) -> Value {
        serde_json::json!({"model": "m", "messages": [{"role": "user", "content": "x".repeat(bytes)}]})
    }

    #[tokio::test]
    async fn large_bodies_pass_and_the_limit_is_configurable() {
        let (port, backend_hits) = status_backend(200).await;
        let default = serve_proxy(hive_with(vec![ep("a", port, &["m"])], vec![]).await).await;
        let small = serve_proxy(
            hive_with(vec![ep("a", port, &["m"])], vec![])
                .await
                .with_max_body(Some(1024 * 1024)),
        )
        .await;
        let c = reqwest::Client::new();
        let post = |url: &str| c.post(format!("{url}/v1/chat/completions")).json(&big_chat(3 << 20)).send();
        // 3 MB: over axum's 2 MB default, well under ours.
        assert_eq!(post(&default).await.unwrap().status(), 200);
        assert_eq!(post(&small).await.unwrap().status(), 413);
        assert_eq!(hits(&backend_hits), 1);
    }

    #[tokio::test]
    async fn upstream_headers_pass_through_minus_cors_and_hop_by_hop() {
        // A streaming request answered with a JSON error: the client must
        // see JSON (not text/event-stream), the upstream's request id, and
        // neither the upstream's CORS policy nor its hive id.
        let mock = Router::new().route(
            "/v1/chat/completions",
            post(|| async {
                (
                    StatusCode::BAD_REQUEST,
                    [
                        ("content-type", "application/json"),
                        ("x-request-id", "req-42"),
                        ("access-control-allow-origin", "*"),
                        (HIVE_ID_HEADER, "someone-else"),
                    ],
                    r#"{"error":{"message":"context too long"}}"#,
                )
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move { axum::serve(listener, mock).await.unwrap() });
        let hive = hive_with(vec![ep("a", port, &["m"])], vec![]).await;
        let me = hive.own_id();
        let url = serve_proxy(hive).await;
        let resp = chat(&url, true).await;
        assert_eq!(resp.status(), 400);
        let h = resp.headers();
        assert_eq!(h["content-type"], "application/json");
        assert_eq!(h["x-request-id"], "req-42");
        assert!(h.get("access-control-allow-origin").is_none());
        assert_eq!(h[HIVE_ID_HEADER], me.as_str());
        assert_eq!(resp.json::<Value>().await.unwrap()["error"]["message"], "context too long");
    }

    #[tokio::test]
    async fn successful_streams_keep_sse_headers() {
        let (port, _) = status_backend(200).await;
        let url = serve_proxy(hive_with(vec![ep("a", port, &["m"])], vec![]).await).await;
        let resp = chat(&url, true).await;
        assert_eq!(resp.status(), 200);
        assert_eq!(resp.headers()["content-type"], "text/event-stream");
        assert_eq!(resp.headers()["cache-control"], "no-cache");
    }

    #[tokio::test]
    async fn overlapping_scans_never_install_stale_results() {
        // First `/v1/models` answer is slow and stale ("old"), later ones
        // are fast and fresh ("new"). Unserialised, the fast second pass
        // would install "new" and then the slow first pass would overwrite
        // it with "old".
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let c = calls.clone();
        let mock = Router::new().route(
            "/v1/models",
            get(move || {
                let c = c.clone();
                async move {
                    let n = c.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    if n == 0 {
                        tokio::time::sleep(Duration::from_millis(400)).await;
                        Json(serde_json::json!({"data": [{"id": "old"}]}))
                    } else {
                        Json(serde_json::json!({"data": [{"id": "new"}]}))
                    }
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move { axum::serve(listener, mock).await.unwrap() });

        let hive = Hive::new(0);
        let pass = |h: Hive, u: String| async move {
            let id = h.own_id();
            h.scan_and_install(probe_static(&h.client, &[u], &id, &Credentials::default())).await;
        };
        let first = tokio::spawn(pass(hive.clone(), url.clone()));
        tokio::time::sleep(Duration::from_millis(50)).await;
        let second = tokio::spawn(pass(hive.clone(), url.clone()));
        first.await.unwrap();
        second.await.unwrap();
        assert_eq!(hive.snapshot().await[0].models, vec!["new".to_string()]);
    }

    fn urls_of(eps: &[DiscoveredEndpoint]) -> Vec<String> {
        eps.iter().map(|e| e.base_url.clone()).collect()
    }

    #[tokio::test]
    async fn members_survive_a_few_missed_scans() {
        let hive = Hive::new(0).with_drop_after(3);
        let a = ep("a", 9001, &["m"]);
        let b = ep("b", 9002, &["m"]);
        hive.set_endpoints(vec![a.clone(), b.clone()]).await;
        // `a` misses two scans: still a member, last known models intact.
        for _ in 0..2 {
            hive.set_endpoints(vec![b.clone()]).await;
            assert_eq!(urls_of(&hive.snapshot().await), urls(&[9001, 9002]));
        }
        assert_eq!(hive.candidates("m").await.len(), 2);
        // Third miss in a row: gone.
        hive.set_endpoints(vec![b.clone()]).await;
        assert_eq!(urls_of(&hive.snapshot().await), urls(&[9002]));
    }

    #[tokio::test]
    async fn answering_resets_the_miss_count_and_updates_models() {
        let hive = Hive::new(0).with_drop_after(2);
        hive.set_endpoints(vec![ep("a", 9001, &["m"])]).await;
        hive.set_endpoints(vec![]).await; // miss 1
        // Answers again, now without "m": taken as-is, count reset.
        hive.set_endpoints(vec![ep("a", 9001, &["other"])]).await;
        assert_eq!(hive.snapshot().await[0].models, vec!["other".to_string()]);
        hive.set_endpoints(vec![]).await; // miss 1 again, not 2
        assert_eq!(hive.snapshot().await.len(), 1);
        hive.set_endpoints(vec![]).await;
        assert!(hive.snapshot().await.is_empty());
    }

    #[tokio::test]
    async fn docker_members_and_drop_after_one_leave_at_once() {
        let mut container = ep("c", 9001, &["m"]);
        container.source = "docker".into();
        let hive = Hive::new(0);
        hive.set_endpoints(vec![container]).await;
        hive.set_endpoints(vec![]).await;
        assert!(hive.snapshot().await.is_empty());

        let strict = Hive::new(0).with_drop_after(1);
        strict.set_endpoints(vec![ep("a", 9001, &["m"])]).await;
        strict.set_endpoints(vec![]).await;
        assert!(strict.snapshot().await.is_empty());
    }

    /// Mock serving `/metrics` (vLLM gauges) and `/load`, counting hits.
    async fn counting_load_backend(
        hits: Arc<StdMutex<Vec<String>>>,
    ) -> String {
        let (h1, h2) = (hits.clone(), hits.clone());
        let mock = Router::new()
            .route(
                "/metrics",
                get(move || {
                    let h = h1.clone();
                    async move {
                        lock(&h).push("metrics".into());
                        "vllm:num_requests_running{engine=\"0\"} 3.0\n"
                    }
                }),
            )
            .route(
                "/load",
                get(move |Query(q): Query<HashMap<String, String>>| {
                    let h = h2.clone();
                    async move {
                        let model = q.get("model").cloned().unwrap_or_default();
                        lock(&h).push(format!("load?model={model}"));
                        Json(serde_json::json!({"server_load": 7, "hivllm": {"exact": true}}))
                    }
                }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move { axum::serve(listener, mock).await.unwrap() });
        url
    }

    #[tokio::test]
    async fn load_probes_run_once_per_server_and_per_model_only_for_hives() {
        let plain_hits = Arc::new(StdMutex::new(Vec::new()));
        let hive_hits = Arc::new(StdMutex::new(Vec::new()));
        let plain_url = counting_load_backend(plain_hits.clone()).await;
        let hive_url = counting_load_backend(hive_hits.clone()).await;
        let mut plain = ep("plain", 1, &["m1", "m2", "m3"]);
        plain.base_url = plain_url;
        let mut peer = ep("peer", 1, &["m1", "m2"]);
        peer.base_url = hive_url;
        peer.hive_id = Some("peer-hive".into());
        let hive = Hive::new(0);
        *hive.endpoints.write().await = vec![plain, peer];
        hive.refresh_load().await;

        // Three models, one server: one /metrics call, shared by all.
        assert_eq!(*lock(&plain_hits), vec!["metrics".to_string()]);
        // A hive peer: its per-model aggregate only, no /metrics.
        let mut asked = lock(&hive_hits).clone();
        asked.sort();
        assert_eq!(asked, vec!["load?model=m1".to_string(), "load?model=m2".to_string()]);
        let loads = hive.loads.read().await;
        for m in ["m1", "m2", "m3"] {
            assert_eq!(loads[&("plain".to_string(), m.to_string())], Load::Measured(3));
        }
        assert_eq!(loads[&("peer".to_string(), "m2".to_string())], Load::Measured(7));
    }

    /// Sink keeping entries in memory.
    #[derive(Default)]
    struct Capture(StdMutex<Vec<crate::logging::LogEntry>>);

    impl crate::logging::LogSink for Capture {
        fn name(&self) -> &'static str {
            "capture"
        }
        fn emit<'a>(&'a self, entry: crate::logging::LogEntry) -> crate::logging::BoxFuture<'a, ()> {
            lock(&self.0).push(entry);
            Box::pin(async {})
        }
    }

    #[tokio::test]
    async fn broken_streams_are_logged_with_their_error() {
        // Backend sends one chunk, then goes silent past the read timeout.
        let mock = Router::new().route(
            "/v1/chat/completions",
            post(|| async {
                let chunks = futures::stream::iter(vec![Ok::<_, std::io::Error>(Bytes::from_static(
                    b"data: {\"choices\":[{\"delta\":{\"content\":\"half\"}}]}\n\n",
                ))])
                .chain(futures::stream::pending());
                ([("content-type", "text/event-stream")], Body::from_stream(chunks))
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move { axum::serve(listener, mock).await.unwrap() });

        let capture = Arc::new(Capture::default());
        let hive = hive_with(vec![ep("a", port, &["m"])], vec![])
            .await
            .with_timeouts(Duration::from_secs(1), Some(Duration::from_millis(300)))
            .with_logger(RequestLogger::new().with_sink(capture.clone()));
        let url = serve_proxy(hive).await;
        let body = chat(&url, true).await.bytes().await;
        assert!(body.is_err(), "the client sees the stream break too");
        let mut entry = None;
        for _ in 0..50 {
            if let Some(e) = lock(&capture.0).first().cloned() {
                entry = Some(e);
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        let entry = entry.expect("stream logged");
        assert!(entry.error.as_deref().unwrap_or("").starts_with("stream broken"), "{:?}", entry.error);
        assert_eq!(entry.response.unwrap().content.as_deref(), Some("half"));
    }

    /// Mock recording (path, content-type, body) of every request.
    async fn recording_backend() -> (String, Arc<StdMutex<Vec<(String, String, Bytes)>>>) {
        let seen = Arc::new(StdMutex::new(Vec::new()));
        let s2 = seen.clone();
        let mock = Router::new()
            .fallback(move |uri: axum::http::Uri, headers: HeaderMap, body: Bytes| {
                let s2 = s2.clone();
                async move {
                    let ct = headers
                        .get("content-type")
                        .map(|v| v.to_str().unwrap().to_string())
                        .unwrap_or_default();
                    lock(&s2).push((uri.path().to_string(), ct, body));
                    Json(serde_json::json!({"ok": true}))
                }
            })
            .layer(axum::extract::DefaultBodyLimit::disable());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move { axum::serve(listener, mock).await.unwrap() });
        (url, seen)
    }

    #[tokio::test]
    async fn extra_routes_reach_the_right_upstream_paths() {
        let (url, seen) = recording_backend().await;
        let mut gw = ep("gw", 1, &["m"]);
        gw.base_url = format!("{url}/gw/v1"); // path-routing gateway
        let hive = hive_with(vec![gw], vec![]).await;
        let proxy = serve_proxy(hive).await;
        let c = reqwest::Client::new();
        for (path, expected) in [
            ("/v1/responses", "/gw/v1/responses"),
            ("/v1/rerank", "/gw/v1/rerank"),
            ("/v2/rerank", "/gw/v2/rerank"),
            ("/tokenize", "/gw/tokenize"),
            ("/v1/audio/speech", "/gw/v1/audio/speech"),
        ] {
            let resp = c
                .post(format!("{proxy}{path}"))
                .json(&serde_json::json!({"model": "m", "input": "x"}))
                .send()
                .await
                .unwrap();
            assert_eq!(resp.status(), 200, "{path}");
            assert_eq!(lock(&seen).last().unwrap().0, expected);
        }
    }

    #[tokio::test]
    async fn audio_uploads_route_by_form_field_and_pass_through_intact() {
        let (url, seen) = recording_backend().await;
        let mut whisper = ep("w", 1, &["whisper"]);
        whisper.base_url = url;
        let capture = Arc::new(Capture::default());
        let hive = hive_with(vec![whisper], vec![])
            .await
            .with_logger(RequestLogger::new().with_sink(capture.clone()));
        let proxy = serve_proxy(hive).await;
        let body: &[u8] = b"--B0\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nwhisper\r\n\
--B0\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n\
Content-Type: audio/wav\r\n\r\nRIFF\x00\xff\r\n--B0--\r\n";
        let resp = reqwest::Client::new()
            .post(format!("{proxy}/v1/audio/transcriptions"))
            .header("content-type", "multipart/form-data; boundary=B0")
            .body(body.to_vec())
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let (path, ct, got) = lock(&seen)[0].clone();
        assert_eq!(path, "/v1/audio/transcriptions");
        assert_eq!(ct, "multipart/form-data; boundary=B0");
        assert_eq!(&got[..], body);
        let entry = lock(&capture.0)[0].clone();
        assert_eq!(entry.route, "transcriptions");
        assert_eq!(entry.model, "whisper");
        assert_eq!(entry.request["form"]["model"], "whisper");
        assert_eq!(entry.request["files"][0]["filename"], "a.wav");
        assert_eq!(entry.request["files"][0]["bytes"], 6);
    }

    #[tokio::test]
    async fn model_list_keeps_upstream_metadata() {
        let mut e = ep("vllm", 9001, &["m"]);
        e.model_meta.insert(
            "m".into(),
            serde_json::json!({"id": "m", "object": "model", "owned_by": "vllm", "max_model_len": 8192, "created": 42}),
        );
        let url = serve_proxy(hive_with(vec![e], vec![]).await).await;
        let list: Value = reqwest::get(format!("{url}/v1/models")).await.unwrap().json().await.unwrap();
        let m = &list["data"][0];
        assert_eq!(m["max_model_len"], 8192);
        assert_eq!(m["created"], 42);
        assert_eq!(m["owned_by"], "test"); // the member's name
        assert!(m["hivllm"]["paths"].is_array());
        let one: Value = reqwest::get(format!("{url}/v1/models/m")).await.unwrap().json().await.unwrap();
        assert_eq!(one["max_model_len"], 8192);
        assert_eq!(reqwest::get(format!("{url}/v1/models/nope")).await.unwrap().status(), 404);
    }

    #[tokio::test]
    async fn round_robin_state_is_per_model_and_pruned() {
        let hive = Hive::new(0);
        hive.set_endpoints(vec![ep("a", 9001, &["m", "old"]), ep("b", 9002, &["m"])]).await;
        hive.candidates("m").await;
        hive.candidates("old").await;
        assert_eq!(hive.rr.lock().await.len(), 2);
        // "old" is no longer served anywhere: its counter goes.
        hive.set_endpoints(vec![ep("a", 9001, &["m"]), ep("b", 9002, &["m"])]).await;
        let rr = hive.rr.lock().await;
        assert_eq!(rr.keys().collect::<Vec<_>>(), vec!["m"]);
    }

    #[tokio::test]
    async fn backend_keys_are_used_for_probes_load_and_queries() {
        // Backend protected like `vllm --api-key backend-key`.
        let seen = Arc::new(StdMutex::new(Vec::<(String, Option<String>)>::new()));
        let s2 = seen.clone();
        let mock = Router::new().fallback(move |uri: axum::http::Uri, headers: HeaderMap| {
            let s2 = s2.clone();
            async move {
                let auth = headers.get("authorization").map(|v| v.to_str().unwrap().to_string());
                lock(&s2).push((uri.path().to_string(), auth.clone()));
                if auth.as_deref() != Some("Bearer backend-key") {
                    return (StatusCode::UNAUTHORIZED, "no").into_response();
                }
                match uri.path() {
                    "/v1/models" => Json(serde_json::json!({"data": [{"id": "m"}]})).into_response(),
                    "/metrics" => "vllm:num_requests_running 2\n".into_response(),
                    _ => Json(serde_json::json!({"id": "ok", "choices": []})).into_response(),
                }
            }
        });
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        tokio::spawn(async move { axum::serve(listener, mock).await.unwrap() });

        let creds = Credentials::new([(url.clone(), "backend-key".to_string())]);
        let hive = Hive::new(0).with_credentials(creds.clone()).with_forward_auth(true);
        let found = probe_static(&hive.client, std::slice::from_ref(&url), &hive.own_id(), &creds).await;
        assert_eq!(found.len(), 1, "key-protected backend discovered");
        hive.set_endpoints(found).await;
        hive.refresh_load().await;
        let id = hive.snapshot().await[0].id.clone();
        assert_eq!(hive.loads.read().await[&(id, "m".to_string())], Load::Measured(2));

        let proxy = serve_proxy(hive).await;
        let resp = reqwest::Client::new()
            .post(format!("{proxy}/v1/chat/completions"))
            .bearer_auth("client-token") // forwarded only to key-less backends
            .json(&serde_json::json!({"model": "m", "messages": []}))
            .send()
            .await
            .unwrap();
        assert_eq!(resp.status(), 200);
        let seen = lock(&seen);
        assert!(seen.iter().all(|(_, a)| a.as_deref() == Some("Bearer backend-key")), "{seen:?}");
        assert!(seen.iter().any(|(p, _)| p == "/v1/chat/completions"));
    }
}
