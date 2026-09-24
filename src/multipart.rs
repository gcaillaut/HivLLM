//! Just enough `multipart/form-data` reading to route audio uploads
//! (`/v1/audio/transcriptions`, `/translations`): find the `model` field
//! and describe the form for the query log. The body itself is forwarded
//! untouched.

use serde_json::{Map, Value};

/// One form part: a text field, or a file (content not kept).
#[derive(Debug, Clone, PartialEq)]
pub enum Part {
    Field { name: String, value: String },
    File {
        name: String,
        filename: String,
        content_type: Option<String>,
        bytes: usize,
    },
}

/// Boundary from a `multipart/form-data; boundary=...` content type.
pub fn boundary(content_type: &str) -> Option<String> {
    let (mime, params) = content_type.split_once(';')?;
    if !mime.trim().eq_ignore_ascii_case("multipart/form-data") {
        return None;
    }
    params.split(';').find_map(|p| {
        let (k, v) = p.trim().split_once('=')?;
        k.trim()
            .eq_ignore_ascii_case("boundary")
            .then(|| v.trim().trim_matches('"').to_string())
            .filter(|b| !b.is_empty())
    })
}

fn find(haystack: &[u8], needle: &[u8], from: usize) -> Option<usize> {
    if needle.is_empty() || from >= haystack.len() {
        return None;
    }
    haystack[from..]
        .windows(needle.len())
        .position(|w| w == needle)
        .map(|i| i + from)
}

/// `name="…"`-style parameter of a Content-Disposition header.
fn disposition_param(disposition: &str, key: &str) -> Option<String> {
    disposition.split(';').find_map(|p| {
        let (k, v) = p.trim().split_once('=')?;
        (k.trim().eq_ignore_ascii_case(key)).then(|| v.trim().trim_matches('"').to_string())
    })
}

/// Parts of a multipart body, in order. Malformed parts are skipped.
pub fn parse(body: &[u8], boundary: &str) -> Vec<Part> {
    let delim = format!("--{boundary}").into_bytes();
    let mut parts = Vec::new();
    let Some(mut pos) = find(body, &delim, 0) else {
        return parts;
    };
    loop {
        let start = pos + delim.len();
        // `--` right after a delimiter closes the body.
        if body.get(start..start + 2) == Some(b"--") {
            break;
        }
        let Some(next) = find(body, &delim, start) else {
            break;
        };
        let raw = &body[start..next];
        let raw = raw.strip_prefix(b"\r\n").unwrap_or(raw);
        let raw = raw.strip_suffix(b"\r\n").unwrap_or(raw);
        if let Some(split) = find(raw, b"\r\n\r\n", 0) {
            let head = String::from_utf8_lossy(&raw[..split]);
            let content = &raw[split + 4..];
            let header = |name: &str| {
                head.lines().find_map(|l| {
                    let (k, v) = l.split_once(':')?;
                    k.trim().eq_ignore_ascii_case(name).then(|| v.trim().to_string())
                })
            };
            if let Some(disp) = header("content-disposition") {
                if let Some(name) = disposition_param(&disp, "name") {
                    parts.push(match disposition_param(&disp, "filename") {
                        Some(filename) => Part::File {
                            name,
                            filename,
                            content_type: header("content-type"),
                            bytes: content.len(),
                        },
                        None => Part::Field {
                            name,
                            value: String::from_utf8_lossy(content).into_owned(),
                        },
                    });
                }
            }
        }
        pos = next;
    }
    parts
}

/// Value of the first text field called `name`.
pub fn field<'p>(parts: &'p [Part], wanted: &str) -> Option<&'p str> {
    parts.iter().find_map(|p| match p {
        Part::Field { name, value } if name == wanted => Some(value.as_str()),
        _ => None,
    })
}

/// Query-log view of a form: text fields as-is, files described.
pub fn describe(parts: &[Part]) -> Value {
    let mut form = Map::new();
    let mut files = Vec::new();
    for p in parts {
        match p {
            Part::Field { name, value } => {
                form.insert(name.clone(), Value::String(value.clone()));
            }
            Part::File {
                name,
                filename,
                content_type,
                bytes,
            } => files.push(serde_json::json!({
                "field": name,
                "filename": filename,
                "content_type": content_type,
                "bytes": bytes,
            })),
        }
    }
    serde_json::json!({ "form": form, "files": files })
}

#[cfg(test)]
mod tests {
    use super::*;

    const BODY: &[u8] = b"--XyZ\r\n\
Content-Disposition: form-data; name=\"model\"\r\n\r\n\
whisper-large\r\n\
--XyZ\r\n\
Content-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n\
Content-Type: audio/wav\r\n\r\n\
RIFF\x00\x01\x02--not-a-boundary\r\n\
--XyZ\r\n\
Content-Disposition: form-data; name=\"stream\"\r\n\r\n\
true\r\n\
--XyZ--\r\n";

    #[test]
    fn reads_boundary_fields_and_files() {
        let b = boundary("multipart/form-data; boundary=XyZ").unwrap();
        assert_eq!(boundary("multipart/form-data; boundary=\"XyZ\"").unwrap(), b);
        assert!(boundary("application/json").is_none());
        let parts = parse(BODY, &b);
        assert_eq!(field(&parts, "model"), Some("whisper-large"));
        assert_eq!(field(&parts, "stream"), Some("true"));
        assert_eq!(
            parts[1],
            Part::File {
                name: "file".into(),
                filename: "a.wav".into(),
                content_type: Some("audio/wav".into()),
                bytes: 23,
            }
        );
        let d = describe(&parts);
        assert_eq!(d["form"]["model"], "whisper-large");
        assert_eq!(d["files"][0]["bytes"], 23);
    }

    #[test]
    fn garbage_yields_no_parts() {
        assert!(parse(b"no boundary here", "XyZ").is_empty());
        assert!(parse(b"--XyZ\r\nbroken", "XyZ").is_empty());
    }
}
