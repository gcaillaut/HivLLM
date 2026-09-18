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
}

/// Fan-out logger: every entry goes to every sink. Cheap to clone,
/// no-op when no sinks are configured.
#[derive(Clone, Default)]
pub struct RequestLogger {
    sinks: Vec<Arc<dyn LogSink>>,
}

impl RequestLogger {
    pub fn new() -> Self {
        Self { sinks: Vec::new() }
    }

    pub fn with_sink(mut self, sink: Arc<dyn LogSink>) -> Self {
        self.sinks.push(sink);
        self
    }

    pub async fn log(&self, entry: LogEntry) {
        for sink in &self.sinks {
            sink.emit(entry.clone()).await;
        }
    }
}

/// Appends one JSON object per line: `<entry as JSON>\n`.
pub struct JsonLinesSink {
    file: Mutex<tokio::fs::File>,
}

impl JsonLinesSink {
    pub async fn open(path: &str) -> std::io::Result<Self> {
        let file = tokio::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .await?;
        Ok(Self {
            file: Mutex::new(file),
        })
    }
}

impl LogSink for JsonLinesSink {
    fn name(&self) -> &'static str {
        "jsonl"
    }

    fn emit<'a>(&'a self, entry: LogEntry) -> BoxFuture<'a, ()> {
        Box::pin(async move {
            let mut line = serde_json::to_vec(&entry).unwrap_or_default();
            line.push(b'\n');
            let mut file = self.file.lock().await;
            use tokio::io::AsyncWriteExt;
            if let Err(e) = file.write_all(&line).await {
                tracing::warn!(error = %e, "query log write failed");
            }
        })
    }
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
        // Read with a deadline: some filesystems/sandboxes make appends
        // visible to subsequent readers with a tiny delay.
        let mut content = String::new();
        for _ in 0..200 {
            content = std::fs::read_to_string(&path).unwrap_or_default();
            if content.lines().count() == 1 {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
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
}
