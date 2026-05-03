use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Trace {
    UserMessage { text: String },
    AssistantMessage { text: String },
    ToolUse { name: String, input: serde_json::Value },
    ToolResult { name: String, output: String },
    Pair { prompt: String, completion: String },
    Other { raw: serde_json::Value },
}

impl Trace {
    pub fn from_ccsniff_event(v: serde_json::Value) -> Self {
        let role = v.get("role").and_then(|r| r.as_str()).unwrap_or("");
        let typ = v.get("type").and_then(|r| r.as_str()).unwrap_or("");
        let tool = v.get("tool").and_then(|r| r.as_str()).map(String::from);
        let text_field = v
            .get("text")
            .and_then(|r| r.as_str())
            .map(String::from)
            .unwrap_or_default();
        let content_text = if text_field.is_empty() {
            extract_text(v.get("content"))
        } else {
            text_field
        };

        match (typ, role, tool) {
            ("tool_use", _, Some(name)) => Trace::ToolUse {
                name,
                input: v.get("input").cloned().unwrap_or(serde_json::Value::Null),
            },
            ("tool_result", _, Some(name)) => Trace::ToolResult {
                name,
                output: content_text,
            },
            (_, "user", _) if !content_text.is_empty() => Trace::UserMessage { text: content_text },
            (_, "assistant", _) if !content_text.is_empty() => {
                Trace::AssistantMessage { text: content_text }
            }
            _ => Trace::Other { raw: v },
        }
    }

    pub fn corpus_text(&self) -> String {
        match self {
            Trace::UserMessage { text } => text.clone(),
            Trace::AssistantMessage { text } => text.clone(),
            Trace::ToolUse { name, input } => {
                format!("<tool_use name={name}>{}</tool_use>", input)
            }
            Trace::ToolResult { name, output } => {
                format!("<tool_result name={name}>{output}</tool_result>")
            }
            Trace::Pair { prompt, completion } => {
                format!("<human>{prompt}</human>\n<bot>{completion}</bot>")
            }
            Trace::Other { raw } => raw.to_string(),
        }
    }

    pub fn is_prompt_role(v: &serde_json::Value) -> bool {
        let role = v.get("role").and_then(|r| r.as_str()).unwrap_or("");
        let typ = v.get("type").and_then(|r| r.as_str()).unwrap_or("");
        let is_meta = v.get("isMeta").and_then(|x| x.as_bool()).unwrap_or(false);
        if is_meta { return false; }
        matches!((role, typ), ("user", "text") | ("assistant", "tool_use"))
    }

    pub fn is_completion_for(prompt: &serde_json::Value, candidate: &serde_json::Value) -> bool {
        let p_role = prompt.get("role").and_then(|r| r.as_str()).unwrap_or("");
        let p_typ = prompt.get("type").and_then(|r| r.as_str()).unwrap_or("");
        let c_role = candidate.get("role").and_then(|r| r.as_str()).unwrap_or("");
        let c_typ = candidate.get("type").and_then(|r| r.as_str()).unwrap_or("");
        let c_meta = candidate.get("isMeta").and_then(|x| x.as_bool()).unwrap_or(false);
        if c_meta { return false; }
        match (p_role, p_typ, c_role, c_typ) {
            ("user", "text", "assistant", "text") => true,
            ("assistant", "tool_use", "tool_result", "tool_result") => true,
            _ => false,
        }
    }
}

fn extract_text(content: Option<&serde_json::Value>) -> String {
    let Some(c) = content else { return String::new() };
    if let Some(s) = c.as_str() {
        return s.to_string();
    }
    if let Some(arr) = c.as_array() {
        let mut out = String::new();
        for item in arr {
            if let Some(t) = item.get("text").and_then(|x| x.as_str()) {
                out.push_str(t);
                out.push('\n');
            }
        }
        return out.trim_end().to_string();
    }
    String::new()
}
