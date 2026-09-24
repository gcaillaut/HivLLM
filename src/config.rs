//! YAML configuration (`--config hivllm.yaml`).
//!
//! Every command-line flag has a key here; precedence is
//! flag / environment variable > config file > built-in default.
//! [`Config::default`] holds the defaults, and `config/hivllm.example.yaml`
//! spells them out (a test keeps the two in sync).
//!
//! Secrets can live in the file (`api_key`) or, preferably, be read from
//! the environment (`api_key_env`) so the file can be committed.

use serde::{Deserialize, Serialize};
use std::net::{IpAddr, Ipv4Addr};

use crate::discovery::Credentials;
use crate::logging::Truncate;

/// Query-log output format. Only `jsonl` is implemented today —
/// `yaml` (and Langfuse/Logfire API sinks) plug into the same
/// `LogSink` trait (see `src/logging.rs`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LogFormat {
    #[default]
    Jsonl,
}

#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub server: Server,
    pub discovery: Discovery,
    pub routing: Routing,
    pub log: Log,
    /// API keys for backends that aren't listed under
    /// `discovery.static_backends` (discovered ports, containers, or a
    /// whole host by URL prefix).
    pub credentials: Vec<Credential>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Server {
    pub port: u16,
    pub bind: IpAddr,
    /// Instance id; random per process when unset.
    pub hive_id: Option<String>,
    /// Key clients must send (`Authorization: Bearer …`); unset = open.
    pub api_key: Option<String>,
    /// Environment variable holding that key (instead of `api_key`).
    pub api_key_env: Option<String>,
    /// "local", "*", a comma-separated origin list, or "" (disabled).
    pub cors_origin: String,
    pub max_body_mb: usize,
}

impl Default for Server {
    fn default() -> Self {
        Self {
            port: 8335,
            bind: IpAddr::V4(Ipv4Addr::LOCALHOST),
            hive_id: None,
            api_key: None,
            api_key_env: None,
            cors_origin: "local".into(),
            max_body_mb: 64,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Discovery {
    pub extra_ports: Vec<u16>,
    pub static_backends: Vec<StaticBackend>,
    /// Docker Engine socket; "" = container discovery disabled.
    pub docker_socket: String,
    pub scan_interval: u64,
    pub drop_after: u32,
}

impl Default for Discovery {
    fn default() -> Self {
        Self {
            extra_ports: Vec::new(),
            static_backends: Vec::new(),
            docker_socket: String::new(),
            scan_interval: 30,
            drop_after: crate::hive::DEFAULT_DROP_AFTER,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Routing {
    pub load_interval: u64,
    pub connect_timeout: u64,
    /// 0 = no read timeout.
    pub read_timeout: u64,
    pub forward_auth: bool,
}

impl Default for Routing {
    fn default() -> Self {
        Self {
            load_interval: 5,
            connect_timeout: 5,
            read_timeout: 600,
            forward_auth: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Log {
    /// "" = no query logging.
    pub file: String,
    pub format: LogFormat,
    pub truncate: Truncate,
    pub max_chars: usize,
    pub max_mb: u64,
    pub keep: usize,
    pub compress: bool,
}

impl Default for Log {
    fn default() -> Self {
        Self {
            file: "hivllm-queries.jsonl".into(),
            format: LogFormat::Jsonl,
            truncate: Truncate::None,
            max_chars: 2000,
            max_mb: 100,
            keep: 20,
            compress: true,
        }
    }
}

/// A static backend: a bare URL, or a URL with its API key.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(untagged)]
pub enum StaticBackend {
    Url(String),
    WithKey(Credential),
}

impl<'de> Deserialize<'de> for StaticBackend {
    // By hand rather than `untagged`, so a typo inside an entry is
    // reported as such, not as "did not match any variant".
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        match serde_yaml_ng::Value::deserialize(d)? {
            serde_yaml_ng::Value::String(url) => Ok(StaticBackend::Url(url)),
            other => serde_yaml_ng::from_value(other)
                .map(StaticBackend::WithKey)
                .map_err(serde::de::Error::custom),
        }
    }
}

impl StaticBackend {
    pub fn url(&self) -> &str {
        match self {
            StaticBackend::Url(u) => u,
            StaticBackend::WithKey(c) => &c.url,
        }
    }
}

/// A backend URL (or URL prefix) and the key to send it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Credential {
    pub url: String,
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default)]
    pub api_key_env: Option<String>,
}

/// `api_key` or `api_key_env` (exactly one, or neither), resolved.
fn secret(what: &str, inline: &Option<String>, env: &Option<String>) -> Result<Option<String>, String> {
    match (inline, env) {
        (Some(_), Some(_)) => Err(format!("{what}: set `api_key` or `api_key_env`, not both")),
        (Some(k), None) => Ok(Some(k.clone()).filter(|k| !k.is_empty())),
        (None, Some(var)) => match std::env::var(var) {
            Ok(k) if !k.is_empty() => Ok(Some(k)),
            _ => Err(format!("{what}: environment variable `{var}` is not set")),
        },
        (None, None) => Ok(None),
    }
}

impl Config {
    pub fn from_yaml(text: &str) -> Result<Self, String> {
        serde_yaml_ng::from_str(text).map_err(|e| e.to_string())
    }

    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| format!("cannot read config {}: {e}", path.display()))?;
        Self::from_yaml(&text).map_err(|e| format!("invalid config {}: {e}", path.display()))
    }

    /// The hive's own API key, if any.
    pub fn server_api_key(&self) -> Result<Option<String>, String> {
        secret("server", &self.server.api_key, &self.server.api_key_env)
    }

    /// Per-backend keys from `discovery.static_backends` and `credentials`.
    pub fn credentials(&self) -> Result<Credentials, String> {
        let mut entries = Vec::new();
        let keyed = self.discovery.static_backends.iter().filter_map(|b| match b {
            StaticBackend::WithKey(c) => Some(c),
            StaticBackend::Url(_) => None,
        });
        for c in keyed.chain(&self.credentials) {
            match secret(&format!("backend {}", c.url), &c.api_key, &c.api_key_env)? {
                Some(key) => entries.push((c.url.clone(), key)),
                None => return Err(format!("backend {}: no `api_key` / `api_key_env`", c.url)),
            }
        }
        Ok(Credentials::new(entries))
    }

    /// Static backend URLs, keys stripped.
    pub fn static_urls(&self) -> Vec<String> {
        self.discovery.static_backends.iter().map(|b| b.url().to_string()).collect()
    }

    /// YAML of the effective configuration, inline secrets masked.
    pub fn to_yaml_redacted(&self) -> String {
        let mut c = self.clone();
        let mask = |k: &mut Option<String>| {
            if k.is_some() {
                *k = Some("***".into());
            }
        };
        mask(&mut c.server.api_key);
        for b in &mut c.discovery.static_backends {
            if let StaticBackend::WithKey(cred) = b {
                mask(&mut cred.api_key);
            }
        }
        for cred in &mut c.credentials {
            mask(&mut cred.api_key);
        }
        serde_yaml_ng::to_string(&c).unwrap_or_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_file_is_all_defaults() {
        assert_eq!(Config::from_yaml("{}").unwrap(), Config::default());
        assert_eq!(Config::from_yaml("server: {}").unwrap(), Config::default());
    }

    #[test]
    fn example_file_spells_out_the_defaults() {
        let text = include_str!("../config/hivllm.example.yaml");
        assert_eq!(Config::from_yaml(text).unwrap(), Config::default());
    }

    #[test]
    fn typos_inside_static_backends_are_named() {
        let err = Config::from_yaml("discovery:\n  static_backends:\n    - url: http://x\n      api_kee: y\n")
            .unwrap_err();
        assert!(err.contains("api_kee"), "{err}");
    }

    #[test]
    fn typos_are_rejected() {
        let err = Config::from_yaml("server:\n  prot: 9000\n").unwrap_err();
        assert!(err.contains("prot"), "{err}");
        assert!(Config::from_yaml("routng: {}").is_err());
    }

    #[test]
    fn static_backends_take_bare_urls_or_keys() {
        std::env::set_var("HIVLLM_TEST_GPU_KEY", "from-env");
        let c = Config::from_yaml(
            "discovery:\n  static_backends:\n    - http://llamacpp:8080\n    - url: http://gpu:8000\n      api_key_env: HIVLLM_TEST_GPU_KEY\n\
credentials:\n  - url: http://127.0.0.1:9000\n    api_key: inline\n",
        )
        .unwrap();
        assert_eq!(c.static_urls(), vec!["http://llamacpp:8080", "http://gpu:8000"]);
        let creds = c.credentials().unwrap();
        assert_eq!(creds.for_url("http://gpu:8000"), Some("from-env"));
        assert_eq!(creds.for_url("http://127.0.0.1:9000"), Some("inline"));
        assert_eq!(creds.for_url("http://llamacpp:8080"), None);
        let shown = c.to_yaml_redacted();
        assert!(!shown.contains("inline") && shown.contains("***"), "{shown}");
    }

    #[test]
    fn broken_secrets_fail_loudly() {
        let both = Config::from_yaml("server:\n  api_key: a\n  api_key_env: B\n").unwrap();
        assert!(both.server_api_key().is_err());
        let missing = Config::from_yaml("server:\n  api_key_env: HIVLLM_TEST_UNSET_VAR\n").unwrap();
        assert!(missing.server_api_key().unwrap_err().contains("HIVLLM_TEST_UNSET_VAR"));
        let keyless = Config::from_yaml("credentials:\n  - url: http://x\n").unwrap();
        assert!(keyless.credentials().is_err());
    }
}
