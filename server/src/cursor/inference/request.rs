//! Maps client-owned inference history without changing earlier messages.
use base64::{engine::general_purpose::STANDARD, Engine};
use serde_json::{json, Value};

use crate::{
    cursor::{
        compile,
        protocol::proto::{agent::v1 as agent, inference as pb},
        tools::tool_call_result::prost_json,
    },
    model::{
        ContentPart, ModelRequest, ModelSpec, ProjectedContent, ProjectedMessage, PromptSpec,
        ProviderReplayState, Role, ToolCallContent, ToolDefinition, ToolResultContent,
    },
    Error, Result,
};

pub fn model(requested: &pb::InferenceRequestedModel) -> Result<ModelSpec> {
    compile::requested_model(&agent::AgentRunRequest {
        requested_model: Some(agent::RequestedModel {
            model_id: requested.model_id.clone(),
            max_mode: requested.max_mode,
            parameters: requested
                .parameters
                .iter()
                .map(|p| agent::requested_model::ModelParameterValue {
                    id: p.id.clone(),
                    value: p.value.clone(),
                })
                .collect(),
            ..Default::default()
        }),
        ..Default::default()
    })
}

pub fn compile(
    request: &pb::InferenceStreamRequest,
    selected: &pb::InferenceRequestedModel,
) -> Result<ModelRequest> {
    if !request.provider_defined_tools.is_empty() {
        return Err(Error::Protocol(
            "provider-defined inference tools are not supported by the configured BYOK provider"
                .into(),
        ));
    }
    let mut spec = model(selected)?;
    if let Some(config) = &request.model_config {
        spec.max_output_tokens = config
            .max_tokens
            .and_then(|v| u64::try_from(v).ok())
            .filter(|v| *v > 0)
            .or(spec.max_output_tokens);
    }
    let mut prompt = PromptSpec {
        instructions: String::new(),
        tools: Vec::new(),
    };
    for tool in &request.tools {
        if tool.custom_tool_format.is_some() {
            return Err(Error::Protocol(format!(
                "custom inference tool format is unsupported: {}",
                tool.name
            )));
        }
        prompt.tools.push(ToolDefinition {
            name: tool.name.clone(),
            description: tool.description.clone(),
            parameters: tool
                .parameters
                .as_ref()
                .map(struct_json)
                .unwrap_or(json!({"type":"object","properties":{}})),
        });
    }
    let mut history = Vec::new();
    let mut leading_system = true;
    for (index, message) in request.messages.iter().enumerate() {
        let role = match pb::InferenceMessageRole::try_from(message.role).ok() {
            Some(pb::InferenceMessageRole::User) => Role::User,
            Some(pb::InferenceMessageRole::Assistant) => Role::Assistant,
            Some(pb::InferenceMessageRole::Tool) => Role::Tool,
            Some(pb::InferenceMessageRole::System) => Role::System,
            _ => {
                return Err(Error::Protocol(
                    "inference message has an invalid role".into(),
                ))
            }
        };
        use pb::inference_core_message::Content;
        if let Some(Content::ToolContent(results)) = &message.content {
            leading_system = false;
            for (part, result) in results.parts.iter().enumerate() {
                let value = result
                    .result
                    .as_ref()
                    .map(prost_json)
                    .unwrap_or(Value::Null);
                let content = value
                    .as_str()
                    .map(str::to_owned)
                    .unwrap_or_else(|| value.to_string());
                history.push(ProjectedMessage {
                    message_id: format!("inference:{index}:{part}"),
                    role: Role::Tool,
                    content: ProjectedContent::ToolResult(ToolResultContent {
                        call_id: result.tool_call_id.clone(),
                        name: result.tool_name.clone(),
                        content,
                        is_error: result.is_error || result.cursor_tool_call_is_error == Some(true),
                        image: None,
                        provider_parts: parts(&result.experimental_content)?,
                    }),
                });
            }
            continue;
        }
        let content = match &message.content {
            Some(Content::Text(text)) => vec![ContentPart::Text { text: text.clone() }],
            Some(Content::Parts(value)) => parts(&value.parts)?,
            None => Vec::new(),
            Some(Content::ToolContent(_)) => unreachable!(),
        };
        if role == Role::System && leading_system {
            for part in content {
                let ContentPart::Text { text } = part else {
                    return Err(Error::Protocol("system images are unsupported".into()));
                };
                if !prompt.instructions.is_empty() {
                    prompt.instructions.push_str("\n\n");
                }
                prompt.instructions.push_str(&text);
            }
            continue;
        }
        leading_system = false;
        let content = if role == Role::Assistant {
            let text = content
                .into_iter()
                .map(|p| match p {
                    ContentPart::Text { text } => Ok(text),
                    _ => Err(Error::Protocol("assistant images are unsupported".into())),
                })
                .collect::<Result<Vec<_>>>()?
                .concat();
            let calls = message
                .tool_calls
                .iter()
                .enumerate()
                .map(|(index, call)| {
                    Ok(ToolCallContent {
                        index,
                        call_id: call.tool_call_id.clone(),
                        name: call.tool_name.clone(),
                        arguments: match &call.raw_tool_call_args {
                            Some(raw) => serde_json::from_str(raw)?,
                            None => call.args.as_ref().map(struct_json).unwrap_or(json!({})),
                        },
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let thinking = message
                .reasoning_parts
                .iter()
                .map(|p| p.text.as_str())
                .collect::<String>();
            let blocks = message.reasoning_parts.iter().filter_map(|p| {
                if p.is_redacted { p.redacted_data.as_ref().map(|data| json!({"type":"redacted_thinking","data":data})) }
                else { p.signature.as_ref().map(|signature| json!({"type":"thinking","thinking":p.text,"signature":signature})) }
            }).collect::<Vec<_>>();
            let replay_state = super::response::replay(message)
                .or_else(|| {
                    (!blocks.is_empty()).then(|| ProviderReplayState {
                        provider_kind: "anthropic".into(),
                        value: json!({"blocks":blocks}),
                    })
                })
                .or_else(|| {
                    (!thinking.is_empty()).then(|| ProviderReplayState {
                        provider_kind: "openai_chat".into(),
                        value: json!({"reasoning_content": thinking}),
                    })
                });
            ProjectedContent::Assistant {
                text,
                thinking,
                calls,
                replay_state,
            }
        } else {
            ProjectedContent::Parts(content)
        };
        history.push(ProjectedMessage {
            message_id: format!("inference:{index}"),
            role,
            content,
        });
    }
    Ok(ModelRequest {
        prompt,
        model: spec,
        history,
    })
}

pub(super) fn struct_json(value: &prost_types::Struct) -> Value {
    Value::Object(
        value
            .fields
            .iter()
            .map(|(k, v)| (k.clone(), prost_json(v)))
            .collect(),
    )
}

fn parts(values: &[pb::InferenceContentPart]) -> Result<Vec<ContentPart>> {
    values
        .iter()
        .map(|part| match part.part.as_ref() {
            Some(pb::inference_content_part::Part::Text(value)) => Ok(ContentPart::Text {
                text: value.text.clone(),
            }),
            Some(pb::inference_content_part::Part::Image(value)) => {
                let (mime, data) = if let Some(data) = value.data.strip_prefix("data:") {
                    data.split_once(";base64,").ok_or_else(|| {
                        Error::Protocol("inference image must be base64 encoded".into())
                    })?
                } else {
                    (
                        value.mime_type.as_deref().unwrap_or("image/png"),
                        value.data.as_str(),
                    )
                };
                Ok(ContentPart::Image {
                    mime_type: mime.into(),
                    data: STANDARD
                        .decode(data)
                        .map_err(|_| Error::Protocol("invalid inference image base64".into()))?,
                })
            }
            Some(pb::inference_content_part::Part::File(_)) => Err(Error::Protocol(
                "inference file parts are unsupported; send extracted text or images".into(),
            )),
            None => Err(Error::Protocol("empty inference content part".into())),
        })
        .collect()
}
