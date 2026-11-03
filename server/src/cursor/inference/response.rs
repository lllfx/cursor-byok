//! Encodes normalized provider events as Cursor inference responses.
use crate::{cursor::protocol::proto::inference as pb, provider::ModelEvent, Error, Result};
use std::collections::BTreeMap;

#[derive(Default)]
pub struct Output {
    text: String,
    thinking: String,
    reasoning: Vec<pb::InferenceReasoningPart>,
    tools: BTreeMap<usize, Tool>,
}

#[derive(Default)]
struct Tool {
    id: String,
    name: String,
    arguments: String,
    complete: bool,
    args: Option<prost_types::Struct>,
}

impl Output {
    pub fn event(&mut self, event: ModelEvent) -> Result<Vec<pb::InferenceStreamResponse>> {
        use pb::inference_stream_response::Response as R;
        let responses = match event {
            ModelEvent::TextDelta(text) => {
                self.text.push_str(&text);
                vec![R::TextPart(pb::InferenceTextStreamPart {
                    text,
                    is_final: false,
                })]
            }
            ModelEvent::TextEnd => vec![R::TextPart(pb::InferenceTextStreamPart {
                text: String::new(),
                is_final: true,
            })],
            ModelEvent::ThinkingDelta(text) => {
                self.thinking.push_str(&text);
                vec![R::ThinkingPart(pb::InferenceThinkingStreamPart {
                    text,
                    ..Default::default()
                })]
            }
            ModelEvent::ThinkingEnd => vec![R::ThinkingPart(pb::InferenceThinkingStreamPart {
                is_final: true,
                ..Default::default()
            })],
            ModelEvent::ToolCallStart {
                index,
                call_id,
                name,
            } => {
                if self
                    .tools
                    .insert(
                        index,
                        Tool {
                            id: call_id,
                            name,
                            ..Default::default()
                        },
                    )
                    .is_some()
                {
                    return Err(Error::Provider("duplicate inference tool index".into()));
                }
                Vec::new()
            }
            ModelEvent::ToolCallArgumentsDelta { index, delta } => {
                let tool = self.tools.get_mut(&index).ok_or_else(|| {
                    Error::Provider("inference tool arguments precede tool start".into())
                })?;
                tool.arguments.push_str(&delta);
                Vec::new()
            }
            ModelEvent::ToolCallEnd { index } => {
                let tool = self.tools.get_mut(&index).ok_or_else(|| {
                    Error::Provider("inference tool end precedes tool start".into())
                })?;
                let args: serde_json::Value = serde_json::from_str(if tool.arguments.is_empty() {
                    "{}"
                } else {
                    &tool.arguments
                })?;
                let args = args.as_object().ok_or_else(|| {
                    Error::Provider("inference tool arguments must be a JSON object".into())
                })?;
                tool.args = Some(prost_types::Struct {
                    fields: crate::cursor::tools::codec::json_object_to_prost(args)
                        .into_iter()
                        .collect(),
                });
                tool.complete = true;
                vec![R::ToolCallPart(pb::InferenceToolCallStreamPart {
                    tool_call_id: tool.id.clone(),
                    tool_name: tool.name.clone(),
                    args: if tool.arguments.is_empty() {
                        "{}".into()
                    } else {
                        tool.arguments.clone()
                    },
                    is_complete: true,
                    tool_index: Some(index as i32),
                    ..Default::default()
                })]
            }
            ModelEvent::Usage(usage) => {
                let count = |v: Option<u64>| v.unwrap_or(0).min(i32::MAX as u64) as i32;
                vec![
                    R::Usage(pb::InferenceUsageInfo {
                        prompt_tokens: count(usage.context_input_tokens.or(usage.input_tokens)),
                        completion_tokens: count(usage.output_tokens),
                        total_tokens: usage.total_tokens.map(|v| count(Some(v))),
                    }),
                    R::ExtendedUsage(pb::InferenceExtendedUsageInfo {
                        input_tokens: count(usage.input_tokens),
                        output_tokens: count(usage.output_tokens),
                        cache_read_tokens: count(usage.cache_read_tokens),
                        cache_write_tokens: count(usage.cache_write_tokens),
                        reasoning_tokens: usage.reasoning_tokens.map(|v| count(Some(v))),
                        ..Default::default()
                    }),
                ]
            }
            ModelEvent::ProviderReplayState(state) if state.provider_kind == "anthropic" => {
                self.reasoning = state
                    .value
                    .get("blocks")
                    .and_then(|v| v.as_array())
                    .into_iter()
                    .flatten()
                    .map(|block| pb::InferenceReasoningPart {
                        is_redacted: block["type"] == "redacted_thinking",
                        text: block["thinking"].as_str().unwrap_or_default().into(),
                        signature: block["signature"].as_str().map(str::to_owned),
                        redacted_data: block["data"].as_str().map(str::to_owned),
                        ..Default::default()
                    })
                    .collect();
                Vec::new()
            }
            ModelEvent::ProviderReplayState(state)
                if matches!(
                    state.provider_kind.as_str(),
                    "openai_responses" | "openai_chat"
                ) =>
            {
                self.reasoning = vec![pb::InferenceReasoningPart {
                    text: self.thinking.clone(),
                    signature: Some(state.value.to_string()),
                    model_name: Some(state.provider_kind),
                    ..Default::default()
                }];
                Vec::new()
            }
            _ => Vec::new(),
        };
        Ok(responses
            .into_iter()
            .map(|response| pb::InferenceStreamResponse {
                response: Some(response),
                ..Default::default()
            })
            .collect())
    }

    pub fn finish(&self, id: &str, model: &str) -> Result<pb::InferenceStreamResponse> {
        if self.tools.values().any(|tool| !tool.complete) {
            return Err(Error::Provider(
                "inference ended with an incomplete tool call".into(),
            ));
        }
        let reasoning_parts = if self.reasoning.is_empty() && !self.thinking.is_empty() {
            vec![pb::InferenceReasoningPart {
                text: self.thinking.clone(),
                ..Default::default()
            }]
        } else {
            self.reasoning.clone()
        };
        let tool_calls = self
            .tools
            .values()
            .map(|tool| pb::InferenceToolCall {
                tool_call_id: tool.id.clone(),
                tool_name: tool.name.clone(),
                args: tool.args.clone(),
                raw_tool_call_args: Some(if tool.arguments.is_empty() {
                    "{}".into()
                } else {
                    tool.arguments.clone()
                }),
            })
            .collect();
        Ok(pb::InferenceStreamResponse {
            response: Some(pb::inference_stream_response::Response::ResponseInfo(
                pb::InferenceResponseInfo {
                    id: id.into(),
                    model: model.into(),
                    created_at: chrono::Utc::now().timestamp(),
                    messages: vec![pb::InferenceResponseMessage {
                        id: id.into(),
                        role: pb::InferenceMessageRole::Assistant as i32,
                        content: Some(self.text.clone()),
                        reasoning_parts,
                        tool_calls,
                        ..Default::default()
                    }],
                    ..Default::default()
                },
            )),
            ..Default::default()
        })
    }
}

pub fn wrap(id: &str, response: pb::InferenceStreamResponse) -> pb::RunInferenceServerMessage {
    pb::RunInferenceServerMessage {
        message: Some(
            pb::run_inference_server_message::Message::InvocationResponse(
                pb::RunInferenceInvocationResponse {
                    invocation_id: id.into(),
                    response: Some(response),
                },
            ),
        ),
    }
}

pub fn end(id: &str, error: Option<&Error>) -> pb::RunInferenceServerMessage {
    pb::RunInferenceServerMessage {
        message: Some(pb::run_inference_server_message::Message::InvocationEnd(
            pb::RunInferenceInvocationEnd {
                invocation_id: id.into(),
                error: error.map(|error| pb::RunInferenceInvocationError {
                    code: if matches!(error, Error::Cancelled) {
                        1
                    } else {
                        13
                    },
                    message: error.to_string(),
                    ..Default::default()
                }),
                ..Default::default()
            },
        )),
    }
}

pub(super) fn replay(
    message: &pb::InferenceCoreMessage,
) -> Option<crate::model::ProviderReplayState> {
    let part = message.reasoning_parts.iter().find(|part| {
        matches!(
            part.model_name.as_deref(),
            Some("openai_responses" | "openai_chat")
        )
    })?;
    let value = serde_json::from_str(part.signature.as_ref()?).ok()?;
    Some(crate::model::ProviderReplayState {
        provider_kind: part.model_name.clone()?,
        value,
    })
}
