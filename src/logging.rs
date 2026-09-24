//! Query logging.
//!
//! Every inference request proxied by the hive is recorded as a [`LogEntry`]
//! and fanned out to all configured [`LogSink`]s.
//!
//! Sink model (deliberately open-ended):
//! - [`JsonLinesSink`] (implemented): appends one JSON object per line.
//!   `tail -f`, `jq` and every log shipper understand JSONL.
//! - Future file sinks (e.g. YAML): implement [`LogSink`] and serialize the
//!   entry followed by a document separator (`---\n`) — YAML documents
//!   append just as cleanly as JSON lines.
//! - Future API sinks (Langfuse, Logfire, ...): implement [`LogSink::emit`]
//!   as an HTTP POST of the entry; batching / retries can live inside
//!   the sink without touching the proxy code.

use chrono::{DateTime, Utc};
use serde::Serialize;
use serde_json::Value;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use tokio::sync::{oneshot, Mutex};

use bytes::Bytes;
use clap::ValueEnum;
use futures::Stream;

pub type BoxFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

/// One proxied query (or failed routing attempt).
#[derive(Debug, Clone, Serialize)]
pub struct LogEntry {
    pub ts: DateTime<Utc>,
    /// "chat" | "completions" | "embeddings"
    pub route: &'static str,
    pub model: String,
    /// Upstream that served it, if routing succeeded.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub upstream: Option<String>,
    pub stream: bool,
    pub status: u16,
    pub latency_ms: u64,
    /// Token usage reported by upstream (non-streaming responses only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// Full request body sent by the client.
    pub request: Value,
    /// Upstream response: extracted text fields + full raw payload.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub response: Option<LoggedResponse>,
}

/// Upstream response, logged in full by default.
///
/// `content` and `reasoning` are kept in separate fields on purpose
/// (reasoning models put thinking in `message.reasoning` /
/// `delta.reasoning[_content]`, not in `content`). Assistant `tool_calls`
/// are merged into one array; tool *results* (role `"tool"`) travel in
/// later client requests, which are logged in full under `request`.
#[derive(Debug, Clone, Default, Serialize)]
pub struct LoggedResponse {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub finish_reason: Option<String>,
    /// SSE chunks seen (streaming responses only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunks: Option<u64>,
    /// Complete upstream payload (non-streaming responses only).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw: Option<Value>,
}

impl LoggedResponse {
    pub fn truncated(mut self, strategy: Truncate, max_chars: usize) -> Self {
        if strategy == Truncate::None {
            return self;
        }
        if let Some(c) = self.content.take() {
            self.content = Some(truncate_text(&c, max_chars));
        }
        if let Some(r) = self.reasoning.take() {
            self.reasoning = Some(truncate_text(&r, max_chars));
        }
        for tc in &mut self.tool_calls {
            truncate_strings(tc, max_chars);
        }
        if let Some(raw) = self.raw.take() {
            self.raw = Some(truncate_value(raw, strategy, max_chars));
        }
        self
    }
}

/// Extract a [`LoggedResponse`] from a complete (non-streaming) upstream
/// JSON payload. Handles chat completions
/// (`choices[].message.{content,reasoning,tool_calls}`) and legacy text
/// completions (`choices[].text`). Embeddings keep their vector in `raw`.
pub fn extract_response(body: &Value) -> LoggedResponse {
    let mut out = LoggedResponse {
        raw: Some(body.clone()),
        ..Default::default()
    };
    let first_choice = body
        .get("choices")
        .and_then(|c| c.as_array())
        .and_then(|a| a.first());
    if let Some(choice) = first_choice {
        if let Some(msg) = choice.get("message") {
            out.content = msg
                .get("content")
                .and_then(|v| v.as_str())
                .map(str::to_string);
            out.reasoning = msg
                .get("reasoning")
                .and_then(|v| v.as_str())
                .or_else(|| {
                    msg.get("reasoning_content")
                        .and_then(|v| v.as_str())
                })
                .map(str::to_string);
            out.tool_calls = msg
                .get("tool_calls")
                .and_then(|v| v.as_array())
                .cloned()
                .unwrap_or_default();
        }
        if out.content.is_none() {
            out.content = choice
                .get("text")
                .and_then(|v| v.as_str())
                .map(str::to_string);
        }
        out.finish_reason = choice
            .get("finish_reason")
            .and_then(|v| v.as_str())
            .map(str::to_string);
    }
    out
}

/// Truncation strategy for long text in query logs. Applies to logged
/// requests and responses alike. Default [`Truncate::None`] keeps
/// everything; [`Truncate::Chars`] cuts long strings to `max_chars`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, ValueEnum)]
pub enum Truncate {
    #[default]
    None,
    Chars,
}

pub fn truncate_text(s: &str, max_chars: usize) -> String {
    let total = s.chars().count();
    if total <= max_chars {
        return s.to_string();
    }
    let kept: String = s.chars().take(max_chars).collect();
    format!("{kept}…[truncated, {total} chars total]")
}

fn truncate_strings(v: &mut Value, max_chars: usize) {
    match v {
        Value::String(s) => {
            if s.chars().count() > max_chars {
                *s = truncate_text(s, max_chars);
            }
        }
        Value::Array(a) => a.iter_mut().for_each(|x| truncate_strings(x, max_chars)),
        Value::Object(o) => o.values_mut().for_each(|x| truncate_strings(x, max_chars)),
        _ => {}
    }
}

pub fn truncate_value(mut v: Value, strategy: Truncate, max_chars: usize) -> Value {
    if strategy == Truncate::None {
        return v;
    }
    truncate_strings(&mut v, max_chars);
    v
}

// ---------- streaming accumulation ----------

/// Incrementally rebuilt response while SSE chunks flow through.
#[derive(Debug, Default)]
pub struct StreamAcc {
    buf: Vec<u8>,
    content: String,
    reasoning: String,
    tool_calls: HashMap<u32, Value>,
    finish_reason: Option<String>,
    chunks: u64,
}

#[derive(Debug, Default)]
pub struct StreamSummary {
    pub content: String,
    pub reasoning: String,
    pub tool_calls: Vec<Value>,
    pub finish_reason: Option<String>,
    pub chunks: u64,
}

fn some_if_nonempty(s: String) -> Option<String> {
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

impl StreamAcc {
    pub fn summary(&self) -> StreamSummary {
        let mut idx: Vec<u32> = self.tool_calls.keys().copied().collect();
        idx.sort_unstable();
        StreamSummary {
            content: self.content.clone(),
            reasoning: self.reasoning.clone(),
            tool_calls: idx
                .iter()
                .filter_map(|i| self.tool_calls.get(i).cloned())
                .collect(),
            finish_reason: self.finish_reason.clone(),
            chunks: self.chunks,
        }
    }
}

impl StreamSummary {
    pub fn into_response(self) -> LoggedResponse {
        LoggedResponse {
            content: some_if_nonempty(self.content),
            reasoning: some_if_nonempty(self.reasoning),
            tool_calls: self.tool_calls,
            finish_reason: self.finish_reason,
            chunks: Some(self.chunks),
            raw: None,
        }
    }
}

fn merge_tool_call(slot: &mut Value, d: &Value) {
    let Some(obj) = slot.as_object_mut() else {
        return;
    };
    for key in ["id", "type"] {
        if !obj.contains_key(key) {
            if let Some(v) = d.get(key) {
                obj.insert(key.to_string(), v.clone());
            }
        }
    }
    let Some(dfunc) = d.get("function") else {
        return;
    };
    let func = obj
        .entry("function")
        .or_insert_with(|| Value::Object(Default::default()));
    let Some(fobj) = func.as_object_mut() else {
        return;
    };
    if !fobj.contains_key("name") {
        if let Some(n) = dfunc.get("name") {
            fobj.insert("name".to_string(), n.clone());
        }
    }
    // `arguments` stream in pieces — concatenate them.
    let mut args = fobj
        .get("arguments")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    args.push_str(dfunc.get("arguments").and_then(|v| v.as_str()).unwrap_or(""));
    fobj.insert("arguments".to_string(), Value::String(args));
}

fn merge_tool_calls(slots: &mut HashMap<u32, Value>, deltas: Option<&Value>) {
    let Some(items) = deltas.and_then(|v| v.as_array()) else {
        return;
    };
    for d in items {
        let idx = d.get("index").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
        let slot = slots
            .entry(idx)
            .or_insert_with(|| Value::Object(Default::default()));
        merge_tool_call(slot, d);
    }
}

/// Feed raw SSE bytes; complete `data:` lines update the accumulator.
pub fn feed_sse(acc: &mut StreamAcc, bytes: &[u8]) {
    acc.buf.extend_from_slice(bytes);
    while let Some(pos) = acc.buf.iter().position(|&b| b == b'\n') {
        let line: Vec<u8> = acc.buf.drain(..=pos).collect();
        let payload = String::from_utf8_lossy(&line)
            .trim()
            .strip_prefix("data:")
            .map(|s| s.trim().to_string());
        let Some(payload) = payload else { continue };
        if payload.is_empty() || payload == "[DONE]" {
            continue;
        }
        let Ok(v) = serde_json::from_str::<Value>(&payload) else {
            continue;
        };
        acc.chunks += 1;
        let choices = v
            .get("choices")
            .and_then(|c| c.as_array())
            .cloned()
            .unwrap_or_default();
        for choice in &choices {
            if let Some(delta) = choice.get("delta") {
                if let Some(s) = delta.get("content").and_then(|v| v.as_str()) {
                    acc.content.push_str(s);
                }
                if let Some(s) = delta
                    .get("reasoning")
                    .and_then(|v| v.as_str())
                    .or_else(|| delta.get("reasoning_content").and_then(|v| v.as_str()))
                {
                    acc.reasoning.push_str(s);
                }
                merge_tool_calls(&mut acc.tool_calls, delta.get("tool_calls"));
            }
            if let Some(fr) = choice.get("finish_reason").and_then(|v| v.as_str()) {
                acc.finish_reason = Some(fr.to_string());
            }
        }
    }
}

/// A [`Stream`] wrapper that forwards every byte untouched while feeding a
/// shared [`StreamAcc`], then reports the [`StreamSummary`] through a
/// oneshot channel when the stream ends. Lets the hive log the *assembled*
/// streamed response (and true end-to-end latency) after serving it.
pub struct TeeStream<S> {
    inner: Pin<Box<S>>,
    acc: Arc<std::sync::Mutex<StreamAcc>>,
    done: Option<oneshot::Sender<StreamSummary>>,
}

impl<S> TeeStream<S> {
    pub fn new(inner: S, acc: Arc<std::sync::Mutex<StreamAcc>>, done: oneshot::Sender<StreamSummary>) -> Self {
        Self {
            inner: Box::pin(inner),
            acc,
            done: Some(done),
        }
    }
}

impl<S, E> Stream for TeeStream<S>
where
    S: Stream<Item = Result<Bytes, E>>,
{
    type Item = Result<Bytes, E>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        // Safe: every field is Unpin, so TeeStream<S> is Unpin for all S.
        let this = self.get_mut();
        match this.inner.as_mut().poll_next(cx) {
            Poll::Ready(None) => {
                if let Some(tx) = this.done.take() {
                    if let Ok(acc) = this.acc.lock() {
                        let _ = tx.send(acc.summary());
                    }
                }
                Poll::Ready(None)
            }
            Poll::Ready(Some(Ok(chunk))) => {
                if let Ok(mut acc) = this.acc.lock() {
                    feed_sse(&mut acc, &chunk);
                }
                Poll::Ready(Some(Ok(chunk)))
            }
            other => other,
        }
    }
}

/// Anything that can receive log entries: files (JSONL, YAML, ...) or
/// HTTP APIs (Langfuse, Logfire, ...).
pub trait LogSink: Send + Sync {
    fn name(&self) -> &'static str;
    fn emit<'a>(&'a self, entry: LogEntry) -> BoxFuture<'a, ()>;
    /// Local file this sink appends to, if any (used by `/api/hive/queries`).
    fn file_path(&self) -> Option<&str> {
        None
    }
}

/// Fan-out logger: every entry goes to every sink. Cheap to clone,
/// no-op when no sinks are configured.
///
/// With [`RequestLogger::spawn_writer`], entries are queued and written
/// by a background task, so serialising and writing a full payload never
/// delays a response. Nothing is dropped: a full queue makes `log` wait.
/// Without it, `log` writes inline.
#[derive(Clone, Default)]
pub struct RequestLogger {
    sinks: Vec<Arc<dyn LogSink>>,
    queue: Option<tokio::sync::mpsc::Sender<LogMsg>>,
}

enum LogMsg {
    Entry(Box<LogEntry>),
    /// Answered once every entry queued before it is written.
    Flush(oneshot::Sender<()>),
}

/// Entries buffered before `log` starts waiting for the writer.
const LOG_QUEUE: usize = 4096;

impl RequestLogger {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_sink(mut self, sink: Arc<dyn LogSink>) -> Self {
        self.sinks.push(sink);
        self
    }

    /// Move writing to a background task (call after adding sinks).
    pub fn spawn_writer(mut self) -> Self {
        if self.sinks.is_empty() {
            return self;
        }
        let (tx, mut rx) = tokio::sync::mpsc::channel::<LogMsg>(LOG_QUEUE);
        let sinks = self.sinks.clone();
        tokio::spawn(async move {
            while let Some(msg) = rx.recv().await {
                match msg {
                    LogMsg::Entry(entry) => {
                        for sink in &sinks {
                            sink.emit((*entry).clone()).await;
                        }
                    }
                    LogMsg::Flush(done) => {
                        let _ = done.send(());
                    }
                }
            }
        });
        self.queue = Some(tx);
        self
    }

    pub async fn log(&self, entry: LogEntry) {
        match &self.queue {
            Some(tx) => {
                if tx.send(LogMsg::Entry(Box::new(entry))).await.is_err() {
                    tracing::warn!("query log writer gone, entry lost");
                }
            }
            None => {
                for sink in &self.sinks {
                    sink.emit(entry.clone()).await;
                }
            }
        }
    }

    /// Wait until every entry logged so far is written.
    pub async fn flush(&self) {
        if let Some(tx) = &self.queue {
            let (done, wait) = oneshot::channel();
            if tx.send(LogMsg::Flush(done)).await.is_ok() {
                let _ = wait.await;
            }
        }
    }

    /// Path of the first file-backed sink, if any.
    pub fn file_sink_path(&self) -> Option<String> {
        self.sinks
            .iter()
            .find_map(|s| s.file_path().map(str::to_string))
    }
}

/// Size-based rolling for [`JsonLinesSink`]. When the live file would grow
/// past `max_bytes`, it is renamed to a timestamped archive next to it
/// (`hivllm-queries.jsonl` → `hivllm-queries.20260924T101500.123Z.jsonl`),
/// gzipped in the background (`….jsonl.gz`) and the oldest archives beyond
/// `keep` are deleted. Entries are never truncated by rolling.
#[derive(Debug, Clone, Copy)]
pub struct Rotation {
    /// Roll when the live file would exceed this size. 0 = never roll.
    pub max_bytes: u64,
    /// Archives kept after rolling. 0 = keep all.
    pub keep: usize,
    /// Gzip archives.
    pub compress: bool,
}

/// Appends one JSON object per line: `<entry as JSON>\n`, rolling the file
/// per [`Rotation`].
pub struct JsonLinesSink {
    path: String,
    rotation: Rotation,
    live: Mutex<LiveFile>,
}

struct LiveFile {
    file: tokio::fs::File,
    size: u64,
}

async fn open_append(path: &str) -> std::io::Result<LiveFile> {
    let file = tokio::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .await?;
    let size = file.metadata().await?.len();
    Ok(LiveFile { file, size })
}

impl JsonLinesSink {
    /// Never rolls.
    #[cfg(test)]
    pub async fn open(path: &str) -> std::io::Result<Self> {
        let never = Rotation { max_bytes: 0, keep: 0, compress: false };
        Self::open_with(path, never).await
    }

    pub async fn open_with(path: &str, rotation: Rotation) -> std::io::Result<Self> {
        Ok(Self {
            path: path.to_string(),
            rotation,
            live: Mutex::new(open_append(path).await?),
        })
    }

    /// Move the live file to a fresh archive and reopen it empty.
    /// Compression and pruning run off the logging path.
    async fn roll(&self, live: &mut LiveFile) -> std::io::Result<()> {
        use tokio::io::AsyncWriteExt;
        live.file.flush().await?;
        let archive = ArchiveName::new(&self.path).fresh_path();
        tokio::fs::rename(&self.path, &archive).await?;
        *live = open_append(&self.path).await?;
        let (path, rotation) = (self.path.clone(), self.rotation);
        tokio::task::spawn_blocking(move || {
            if rotation.compress {
                if let Err(e) = gzip_in_place(&archive) {
                    tracing::warn!(error = %e, archive = %archive.display(), "query log compression failed");
                }
            }
            if rotation.keep > 0 {
                prune_archives(&path, rotation.keep);
            }
        });
        Ok(())
    }
}

impl LogSink for JsonLinesSink {
    fn name(&self) -> &'static str {
        "jsonl"
    }

    fn file_path(&self) -> Option<&str> {
        Some(&self.path)
    }

    fn emit<'a>(&'a self, entry: LogEntry) -> BoxFuture<'a, ()> {
        Box::pin(async move {
            use tokio::io::AsyncWriteExt;
            let mut line = serde_json::to_vec(&entry).unwrap_or_default();
            line.push(b'\n');
            let mut live = self.live.lock().await;
            let max = self.rotation.max_bytes;
            if max > 0 && live.size > 0 && live.size + line.len() as u64 > max {
                if let Err(e) = self.roll(&mut live).await {
                    tracing::warn!(error = %e, "query log rolling failed, still appending");
                }
            }
            // tokio's File buffers writes in a background task: without the
            // flush, entries (and write errors) surface late or never.
            let written = async {
                live.file.write_all(&line).await?;
                live.file.flush().await
            };
            match written.await {
                Ok(()) => live.size += line.len() as u64,
                Err(e) => tracing::warn!(error = %e, "query log write failed"),
            }
        })
    }
}

/// Naming scheme shared by rolling, pruning and reading back:
/// `{dir}/{stem}.{timestamp}{.ext}{.gz}`.
struct ArchiveName {
    dir: std::path::PathBuf,
    live_name: String,
    stem: String,
    /// `.jsonl` (with the dot), or empty when the live file has no extension.
    ext: String,
}

impl ArchiveName {
    fn new(live: &str) -> Self {
        let p = std::path::Path::new(live);
        let dir = match p.parent() {
            Some(d) if !d.as_os_str().is_empty() => d.to_path_buf(),
            _ => std::path::PathBuf::from("."),
        };
        let str_of = |o: Option<&std::ffi::OsStr>| {
            o.map(|s| s.to_string_lossy().into_owned()).unwrap_or_default()
        };
        let ext = str_of(p.extension());
        Self {
            dir,
            live_name: str_of(p.file_name()),
            stem: str_of(p.file_stem()),
            ext: if ext.is_empty() { ext } else { format!(".{ext}") },
        }
    }

    /// Unused archive path for "now" (millisecond timestamps, suffixed on
    /// the rare collision).
    fn fresh_path(&self) -> std::path::PathBuf {
        let ts = Utc::now().format("%Y%m%dT%H%M%S%.3fZ").to_string();
        let mut n = 0;
        loop {
            let tag = if n == 0 { ts.clone() } else { format!("{ts}-{n}") };
            let plain = self.dir.join(format!("{}.{tag}{}", self.stem, self.ext));
            let gz = self.dir.join(format!("{}.{tag}{}.gz", self.stem, self.ext));
            if !plain.exists() && !gz.exists() {
                return plain;
            }
            n += 1;
        }
    }

    /// Timestamp tag of an archive file name, if it is one of ours.
    fn tag_of<'n>(&self, name: &'n str) -> Option<&'n str> {
        if name == self.live_name {
            return None;
        }
        let rest = name.strip_prefix(&self.stem)?.strip_prefix('.')?;
        let rest = rest.strip_suffix(".gz").unwrap_or(rest);
        let tag = rest.strip_suffix(self.ext.as_str())?;
        tag.starts_with(|c: char| c.is_ascii_digit()).then_some(tag)
    }

    /// Archives newest first, one entry per tag: `(tag, files)`. A tag can
    /// briefly have both the plain file and its `.gz` while compressing;
    /// the plain one is listed first (the `.gz` may not be final yet).
    fn list(&self) -> Vec<(String, Vec<std::path::PathBuf>)> {
        let mut by_tag: std::collections::BTreeMap<String, Vec<std::path::PathBuf>> =
            Default::default();
        let Ok(entries) = std::fs::read_dir(&self.dir) else {
            return Vec::new();
        };
        for entry in entries.flatten() {
            let name = entry.file_name().to_string_lossy().into_owned();
            if let Some(tag) = self.tag_of(&name) {
                by_tag.entry(tag.to_string()).or_default().push(entry.path());
            }
        }
        let mut out: Vec<_> = by_tag.into_iter().rev().collect();
        for (_, files) in &mut out {
            files.sort_by_key(|f| f.extension().is_some_and(|e| e == "gz"));
        }
        out
    }
}

/// `archive` → `archive.gz` (written as `.gz.tmp`, then renamed, so a
/// `.gz` on disk is always complete), then the plain file is removed.
fn gzip_in_place(archive: &std::path::Path) -> std::io::Result<()> {
    let name = archive.file_name().unwrap_or_default().to_string_lossy();
    let gz = archive.with_file_name(format!("{name}.gz"));
    let tmp = archive.with_file_name(format!("{name}.gz.tmp"));
    let mut input = std::fs::File::open(archive)?;
    let mut enc = flate2::write::GzEncoder::new(
        std::fs::File::create(&tmp)?,
        flate2::Compression::default(),
    );
    std::io::copy(&mut input, &mut enc)?;
    enc.finish()?.sync_all()?;
    std::fs::rename(&tmp, &gz)?;
    std::fs::remove_file(archive)
}

/// Delete archives beyond the newest `keep`.
fn prune_archives(live: &str, keep: usize) {
    for (_, files) in ArchiveName::new(live).list().into_iter().skip(keep) {
        for f in files {
            if let Err(e) = std::fs::remove_file(&f) {
                tracing::warn!(error = %e, file = %f.display(), "query log pruning failed");
            }
        }
    }
}

/// Last `limit` entries across the live file and its archives, newest
/// first. Lines that don't parse (e.g. one being written) are skipped.
pub async fn read_recent(live: &str, limit: usize) -> Vec<Value> {
    let live = live.to_string();
    tokio::task::spawn_blocking(move || {
        let mut out = Vec::new();
        let take_from = |content: &str, out: &mut Vec<Value>| {
            for line in content.lines().rev() {
                if out.len() >= limit {
                    return;
                }
                if let Ok(v) = serde_json::from_str(line) {
                    out.push(v);
                }
            }
        };
        take_from(&std::fs::read_to_string(&live).unwrap_or_default(), &mut out);
        for (_, files) in ArchiveName::new(&live).list() {
            if out.len() >= limit {
                break;
            }
            if let Some(content) = files.iter().find_map(|f| read_archive(f).ok()) {
                take_from(&content, &mut out);
            }
        }
        out
    })
    .await
    .unwrap_or_default()
}

fn read_archive(path: &std::path::Path) -> std::io::Result<String> {
    use std::io::Read;
    let file = std::fs::File::open(path)?;
    let mut content = String::new();
    if path.extension().is_some_and(|e| e == "gz") {
        flate2::read::GzDecoder::new(file).read_to_string(&mut content)?;
    } else {
        std::io::BufReader::new(file).read_to_string(&mut content)?;
    }
    Ok(content)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entry_serializes_with_expected_fields() {
        let entry = LogEntry {
            ts: Utc::now(),
            route: "chat",
            model: "m".into(),
            upstream: Some("http://127.0.0.1:9037".into()),
            stream: false,
            status: 200,
            latency_ms: 12,
            usage: Some(serde_json::json!({"total_tokens": 6})),
            error: None,
            request: serde_json::json!({"model": "m"}),
            response: None,
        };
        let v: Value = serde_json::to_value(&entry).unwrap();
        assert_eq!(v["route"], "chat");
        assert_eq!(v["model"], "m");
        assert_eq!(v["status"], 200);
        assert!(v.get("error").is_none());
        assert_eq!(v["usage"]["total_tokens"], 6);
    }

    #[tokio::test]
    async fn jsonl_sink_appends_one_line_per_entry() {
        let path = std::env::temp_dir().join("hivllm-test-queries.jsonl");
        let _ = std::fs::remove_file(&path);
        let sink = JsonLinesSink::open(path.to_str().unwrap()).await.unwrap();
        let entry = LogEntry {
            ts: Utc::now(),
            route: "embeddings",
            model: "m".into(),
            upstream: None,
            stream: false,
            status: 404,
            latency_ms: 1,
            usage: None,
            error: Some("nope".into()),
            request: serde_json::json!({}),
            response: None,
        };
        sink.emit(entry).await;
        drop(sink);
        // Flushed on emit: visible to readers right away.
        let content = std::fs::read_to_string(&path).unwrap_or_default();
        assert_eq!(content.lines().count(), 1);
        let v: Value = serde_json::from_str(content.trim()).unwrap();
        assert_eq!(v["route"], "embeddings");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn extract_keeps_content_reasoning_and_tool_calls_separate() {
        let body = serde_json::json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "I'll check that.",
                    "reasoning": "user wants X, so I need tool Y",
                    "tool_calls": [{
                        "id": "call_1", "type": "function",
                        "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}
                    }]
                },
                "finish_reason": "tool_calls"
            }]
        });
        let r = extract_response(&body);
        assert_eq!(r.content.as_deref(), Some("I'll check that."));
        assert_eq!(r.reasoning.as_deref(), Some("user wants X, so I need tool Y"));
        assert_eq!(r.tool_calls.len(), 1);
        assert_eq!(r.tool_calls[0]["function"]["name"], "get_weather");
        assert_eq!(r.finish_reason.as_deref(), Some("tool_calls"));
        assert!(r.raw.is_some());
    }

    #[test]
    fn extract_handles_legacy_text_completions() {
        let body = serde_json::json!({"choices": [{"text": "hello there", "finish_reason": "length"}]});
        let r = extract_response(&body);
        assert_eq!(r.content.as_deref(), Some("hello there"));
        assert!(r.reasoning.is_none());
    }

    #[test]
    fn stream_accumulator_merges_deltas() {
        let mut acc = StreamAcc::default();
        // chunk split mid-line + reasoning + split tool_calls arguments
        feed_sse(&mut acc, b"data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"Hi\"}}]}\n\n");
        feed_sse(&mut acc, b"data: {\"choices\":[{\"delta\":{\"reasoning\":\"think");
        feed_sse(&mut acc, b"ing\"}}]}\n");
        feed_sse(&mut acc, b"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c1\",\"function\":{\"name\":\"f\",\"arguments\":\"{\\\"a\\\"\"}}]}}]}\n");
        feed_sse(&mut acc, b"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\":1}\"}}]}},{\"finish_reason\":null}]}\n");
        feed_sse(&mut acc, b"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n");
        feed_sse(&mut acc, b"data: [DONE]\n");
        let s = acc.summary();
        assert_eq!(s.content, "Hi");
        assert_eq!(s.reasoning, "thinking");
        assert_eq!(s.tool_calls.len(), 1);
        assert_eq!(s.tool_calls[0]["id"], "c1");
        assert_eq!(s.tool_calls[0]["function"]["arguments"], "{\"a\":1}");
        assert_eq!(s.finish_reason.as_deref(), Some("tool_calls"));
        assert_eq!(s.chunks, 5);
        let r = s.into_response();
        assert_eq!(r.chunks, Some(5));
        assert!(r.raw.is_none());
    }

    #[test]
    fn truncate_chars_cuts_long_strings_only() {
        let v = serde_json::json!({"short": "abc", "long": "x".repeat(100)});
        let t = truncate_value(v, Truncate::Chars, 10);
        assert_eq!(t["short"], "abc");
        assert!(t["long"].as_str().unwrap().starts_with("xxxxxxxxxx…"));
        let kept = truncate_value(serde_json::json!({"a": 1}), Truncate::None, 10);
        assert_eq!(kept["a"], 1);
    }

    fn entry(model: &str) -> LogEntry {
        LogEntry {
            ts: Utc::now(),
            route: "chat",
            model: model.into(),
            upstream: None,
            stream: false,
            status: 200,
            latency_ms: 1,
            usage: None,
            error: None,
            request: serde_json::json!({"pad": "x".repeat(200)}),
            response: None,
        }
    }

    fn scratch_dir(name: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "hivllm-{name}-{}-{}",
            std::process::id(),
            Utc::now().timestamp_nanos_opt().unwrap_or_default()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn names(dir: &std::path::Path) -> Vec<String> {
        let mut v: Vec<String> = std::fs::read_dir(dir)
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        v.sort();
        v
    }

    #[tokio::test]
    async fn rolls_compresses_prunes_and_reads_back_across_archives() {
        let dir = scratch_dir("roll");
        let live = dir.join("q.jsonl");
        let live_s = live.to_str().unwrap();
        // Every entry (~300 bytes) overflows 300 bytes: one roll per entry.
        let rotation = Rotation { max_bytes: 300, keep: 2, compress: true };
        let sink = JsonLinesSink::open_with(live_s, rotation).await.unwrap();
        for i in 0..6 {
            sink.emit(entry(&format!("m{i}"))).await;
        }
        // Compression + pruning run in the background: wait for them.
        let settled = |n: &[String]| {
            n.iter().filter(|f| f.ends_with(".jsonl.gz")).count() == 2
                && n.iter().filter(|f| f.starts_with("q.2")).all(|f| f.ends_with(".gz"))
        };
        for _ in 0..200 {
            if settled(&names(&dir)) {
                break;
            }
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        }
        let files = names(&dir);
        assert!(settled(&files), "{files:?}");
        assert!(files.contains(&"q.jsonl".to_string()), "{files:?}");
        assert_eq!(files.len(), 3, "{files:?}");

        // Newest first: live file (m5), then the two kept archives.
        let recent = read_recent(live_s, 10).await;
        let models: Vec<&str> = recent.iter().map(|v| v["model"].as_str().unwrap()).collect();
        assert_eq!(models, vec!["m5", "m4", "m3"]);
        assert_eq!(read_recent(live_s, 2).await.len(), 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn rolling_without_compression_keeps_plain_archives() {
        let dir = scratch_dir("roll-plain");
        let live = dir.join("q.jsonl");
        let rotation = Rotation { max_bytes: 300, keep: 0, compress: false };
        let sink = JsonLinesSink::open_with(live.to_str().unwrap(), rotation).await.unwrap();
        for i in 0..3 {
            sink.emit(entry(&format!("m{i}"))).await;
        }
        let files = names(&dir);
        assert_eq!(files.len(), 3, "{files:?}");
        assert!(files.iter().all(|f| f.ends_with(".jsonl")), "{files:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn archive_names_only_match_our_scheme() {
        let a = ArchiveName::new("/logs/q.jsonl");
        assert_eq!(a.tag_of("q.20260924T101500.123Z.jsonl"), Some("20260924T101500.123Z"));
        assert_eq!(a.tag_of("q.20260924T101500.123Z-1.jsonl.gz"), Some("20260924T101500.123Z-1"));
        assert_eq!(a.tag_of("q.jsonl"), None); // the live file
        assert_eq!(a.tag_of("q.20260924T101500.123Z.jsonl.gz.tmp"), None);
        assert_eq!(a.tag_of("q.backup.jsonl"), None);
        assert_eq!(a.tag_of("other.20260924T101500.123Z.jsonl"), None);
        let bare = ArchiveName::new("queries");
        assert_eq!(bare.tag_of("queries.20260924T101500.123Z.gz"), Some("20260924T101500.123Z"));
    }

    /// Sink that takes its time, recording what it got.
    struct SlowSink(std::time::Duration, Arc<std::sync::Mutex<Vec<String>>>);

    impl LogSink for SlowSink {
        fn name(&self) -> &'static str {
            "slow"
        }
        fn emit<'a>(&'a self, entry: LogEntry) -> BoxFuture<'a, ()> {
            Box::pin(async move {
                tokio::time::sleep(self.0).await;
                self.1.lock().unwrap().push(entry.model);
            })
        }
    }

    #[tokio::test]
    async fn background_writer_keeps_logging_off_the_request_path() {
        let got = Arc::new(std::sync::Mutex::new(Vec::new()));
        let slow = SlowSink(std::time::Duration::from_millis(300), got.clone());
        let logger = RequestLogger::new().with_sink(Arc::new(slow)).spawn_writer();
        let start = std::time::Instant::now();
        logger.log(entry("a")).await;
        logger.log(entry("b")).await;
        assert!(start.elapsed() < std::time::Duration::from_millis(200));
        assert!(got.lock().unwrap().is_empty());
        // Flush waits for everything queued before it, in order.
        logger.flush().await;
        assert_eq!(*got.lock().unwrap(), vec!["a".to_string(), "b".to_string()]);
    }
}
