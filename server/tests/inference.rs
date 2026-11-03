//! Exercises the Cursor 3.27 inference protocol through the desktop HTTP boundary.
#[path = "support/fake_provider.rs"]
mod fake_provider;
#[path = "support/fixtures.rs"]
mod fixtures;

use axum::{
    body::Body,
    http::{Request, StatusCode},
};
use bytes::Bytes;
use cursor_server::{
    api::cursor,
    cursor::{
        prompting::{PromptAssets, PromptCompiler},
        protocol::proto::aiserver::v1 as ai,
        transport::TransportRegistry,
    },
    network::NetworkClients,
};
use cursor_server::{
    cursor::{
        inference::Session,
        protocol::{connect, proto::inference as pb},
    },
    provider::{FinishReason, ModelEvent},
};
use prost::Message;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::mpsc::UnboundedReceiver;
use tower::ServiceExt;

struct Harness {
    _directory: tempfile::TempDir,
    router: axum::Router,
    registry: TransportRegistry,
    provider: fake_provider::FakeProvider,
    model: String,
}

impl Harness {
    async fn new() -> Self {
        let (directory, store) = fixtures::temp_store().await;
        let model = store
            .create_model(
                &serde_json::from_value(serde_json::json!({
                    "display_name":"Test", "type":"openai", "base_url":"https://example.com",
                    "api_key":"test", "tooltip_data":"Test model", "model_id":"gpt-test"
                }))
                .unwrap(),
            )
            .await
            .unwrap()
            .model_hash;
        let provider = fake_provider::FakeProvider::default();
        let assets = PromptAssets::load(
            &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("prompt/cursor"),
        )
        .unwrap();
        let registry = TransportRegistry::new(
            store.clone(),
            Arc::new(provider.clone()),
            PromptCompiler::new(assets),
        );
        let router = cursor::router(registry.clone(), NetworkClients::new(store)).unwrap();
        Self {
            _directory: directory,
            router,
            registry,
            provider,
            model,
        }
    }

    fn run(&self) -> pb::run_inference_client_message::Message {
        pb::run_inference_client_message::Message::RunRequest(pb::RunInferenceRunRequest {
            conversation_id: "100d734a-afcd-4460-b001-a28998347595".into(),
            requested_model: Some(pb::InferenceRequestedModel {
                model_id: self.model.clone(),
                ..Default::default()
            }),
            agent_module: Some(pb::RunInferenceAgentModule {
                prompt_versions: vec!["v1".into()],
                ..Default::default()
            }),
            ..Default::default()
        })
    }

    async fn append(&self, seqno: i64, message: pb::run_inference_client_message::Message) {
        let payload = pb::RunInferenceClientMessage {
            message: Some(message),
        }
        .encode_to_vec();
        let append = ai::BidiAppendRequest {
            request_id: Some(ai::BidiRequestId {
                request_id: "session".into(),
            }),
            append_seqno: seqno,
            // Exercise both representations supported by current Cursor clients.
            data: if seqno % 2 == 0 {
                hex::encode(&payload)
            } else {
                String::new()
            },
            data_binary: if seqno % 2 == 1 { payload } else { Vec::new() },
        };
        let response = self
            .router
            .clone()
            .oneshot(
                Request::post("/aiserver.v1.BidiService/BidiAppend")
                    .header("content-type", "application/proto")
                    .body(Body::from(append.encode_to_vec()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let status = response.status();
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
    }

    async fn start(&self) -> (Session, UnboundedReceiver<Bytes>) {
        let session = self.registry.inference().session("session").await.unwrap();
        self.append(0, self.run()).await;
        let mut receiver = session.subscribe();
        loop {
            if let Some(pb::run_inference_server_message::Message::RunReady(ready)) =
                next(&mut receiver).await.message
            {
                assert_eq!(ready.resolved_model.unwrap().model_id, self.model);
                assert_eq!(ready.prompt_model_metadata.unwrap().prompt_version, "v1");
                assert!(!ready.accepts_delta_invocations);
                assert!(ready.request_compression.is_none());
                break;
            }
        }
        (session, receiver)
    }
}

async fn next(receiver: &mut UnboundedReceiver<Bytes>) -> pb::RunInferenceServerMessage {
    let bytes = tokio::time::timeout(Duration::from_secs(3), receiver.recv())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(bytes[0], 0);
    pb::RunInferenceServerMessage::decode(&bytes[5..]).unwrap()
}

async fn completed(
    receiver: &mut UnboundedReceiver<Bytes>,
    id: &str,
) -> (
    Vec<pb::InferenceStreamResponse>,
    pb::RunInferenceInvocationEnd,
) {
    let mut responses = Vec::new();
    loop {
        match next(receiver).await.message.unwrap() {
            pb::run_inference_server_message::Message::InvocationResponse(response) => {
                assert_eq!(response.invocation_id, id);
                responses.push(response.response.unwrap());
            }
            pb::run_inference_server_message::Message::InvocationEnd(end) => {
                assert_eq!(end.invocation_id, id);
                return (responses, end);
            }
            _ => {}
        }
    }
}

fn text(role: pb::InferenceMessageRole, text: &str) -> pb::InferenceCoreMessage {
    pb::InferenceCoreMessage {
        role: role as i32,
        content: Some(pb::inference_core_message::Content::Text(text.into())),
        ..Default::default()
    }
}

fn invoke(
    id: &str,
    request: pb::InferenceStreamRequest,
) -> pb::run_inference_client_message::Message {
    pb::run_inference_client_message::Message::InvokeModel(pb::RunInferenceInvokeModel {
        invocation_id: id.into(),
        request: Some(request),
        ..Default::default()
    })
}

#[tokio::test]
async fn streams_text_and_tools_and_preserves_history_prefix_across_invocations() {
    let h = Harness::new().await;
    let (session, mut rx) = h.start().await;
    h.provider.push(vec![
        ModelEvent::ThinkingDelta("think".into()),
        ModelEvent::ThinkingEnd,
        ModelEvent::TextDelta("Reading".into()),
        ModelEvent::TextEnd,
        ModelEvent::ToolCallStart {
            index: 0,
            call_id: "call-1".into(),
            name: "read_file".into(),
        },
        ModelEvent::ToolCallArgumentsDelta {
            index: 0,
            delta: "{\"path\":".into(),
        },
        ModelEvent::ToolCallArgumentsDelta {
            index: 0,
            delta: "\"main.rs\"}".into(),
        },
        ModelEvent::ToolCallEnd { index: 0 },
        ModelEvent::Done(FinishReason::ToolUse),
    ]);
    let mut request = pb::InferenceStreamRequest {
        messages: vec![
            text(pb::InferenceMessageRole::System, "Stable instructions"),
            text(pb::InferenceMessageRole::User, "Read main.rs"),
        ],
        tools: vec![pb::InferenceAgentTool {
            name: "read_file".into(),
            description: "Read file".into(),
            ..Default::default()
        }],
        ..Default::default()
    };
    h.append(1, invoke("first", request.clone())).await;
    let (responses, end) = completed(&mut rx, "first").await;
    assert!(end.error.is_none());
    assert!(responses.iter().any(|r| matches!(&r.response, Some(pb::inference_stream_response::Response::TextPart(p)) if p.text == "Reading")));
    let tool = responses
        .iter()
        .find_map(|r| match &r.response {
            Some(pb::inference_stream_response::Response::ToolCallPart(tool)) => Some(tool),
            _ => None,
        })
        .unwrap();
    assert!(tool.is_complete);
    assert_eq!(
        serde_json::from_str::<serde_json::Value>(&tool.args).unwrap(),
        serde_json::json!({"path":"main.rs"})
    );
    let info = responses
        .into_iter()
        .find_map(|r| match r.response {
            Some(pb::inference_stream_response::Response::ResponseInfo(info)) => Some(info),
            _ => None,
        })
        .unwrap();
    let assistant = &info.messages[0];
    assert!(assistant.tool_calls[0].args.is_some());
    request.messages.push(pb::InferenceCoreMessage {
        role: pb::InferenceMessageRole::Assistant as i32,
        content: assistant
            .content
            .clone()
            .map(pb::inference_core_message::Content::Text),
        tool_calls: assistant.tool_calls.clone(),
        reasoning_parts: assistant.reasoning_parts.clone(),
        ..Default::default()
    });
    request.messages.push(pb::InferenceCoreMessage {
        role: pb::InferenceMessageRole::Tool as i32,
        content: Some(pb::inference_core_message::Content::ToolContent(
            pb::InferenceToolResultContent {
                parts: vec![pb::InferenceToolResultPart {
                    tool_call_id: "call-1".into(),
                    tool_name: "read_file".into(),
                    result: Some(prost_types::Value {
                        kind: Some(prost_types::value::Kind::StringValue("fn main() {}".into())),
                    }),
                    ..Default::default()
                }],
            },
        )),
        ..Default::default()
    });
    h.provider.push(vec![
        ModelEvent::TextDelta("Done".into()),
        ModelEvent::Done(FinishReason::Stop),
    ]);
    h.append(2, invoke("second", request)).await;
    assert!(completed(&mut rx, "second").await.1.error.is_none());
    let requests = h.provider.requests();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].prompt, requests[1].prompt);
    assert_eq!(
        requests[0].history,
        requests[1].history[..requests[0].history.len()]
    );
    assert_eq!(requests[1].history.len(), 3);
    session.close();
}

#[tokio::test]
async fn out_of_order_and_duplicate_appends_do_not_repeat_provider_calls() {
    let h = Harness::new().await;
    let (session, mut rx) = h.start().await;
    h.provider.push(vec![ModelEvent::Done(FinishReason::Stop)]);
    h.append(2, invoke("once", Default::default())).await;
    assert!(h.provider.requests().is_empty());
    h.append(
        1,
        pb::run_inference_client_message::Message::ClientHeartbeat(Default::default()),
    )
    .await;
    assert!(completed(&mut rx, "once").await.1.error.is_none());
    h.append(2, invoke("once", Default::default())).await;
    assert_eq!(h.provider.requests().len(), 1);
    session.close();
}

#[tokio::test]
async fn cancellation_ends_only_the_invocation_and_run_can_continue() {
    let h = Harness::new().await;
    let (session, mut rx) = h.start().await;
    h.provider.push_pending();
    h.append(1, invoke("cancelled", Default::default())).await;
    h.append(
        2,
        pb::run_inference_client_message::Message::CancelInvocation(
            pb::RunInferenceCancelInvocation {
                invocation_id: "cancelled".into(),
            },
        ),
    )
    .await;
    assert_eq!(
        completed(&mut rx, "cancelled").await.1.error.unwrap().code,
        1
    );
    h.provider.push(vec![ModelEvent::Done(FinishReason::Stop)]);
    h.append(3, invoke("next", Default::default())).await;
    assert!(completed(&mut rx, "next").await.1.error.is_none());
    session.close();
}

#[tokio::test]
async fn provider_errors_eof_and_incomplete_tools_terminate_invocations() {
    let h = Harness::new().await;
    let (session, mut rx) = h.start().await;
    h.provider.push_error(cursor_server::Error::Provider(
        "upstream unavailable".into(),
    ));
    h.append(1, invoke("error", Default::default())).await;
    assert!(completed(&mut rx, "error")
        .await
        .1
        .error
        .unwrap()
        .message
        .contains("upstream unavailable"));
    h.provider
        .push(vec![ModelEvent::TextDelta("partial".into())]);
    h.append(2, invoke("eof", Default::default())).await;
    assert!(completed(&mut rx, "eof")
        .await
        .1
        .error
        .unwrap()
        .message
        .contains("without completion"));
    h.provider.push(vec![
        ModelEvent::ToolCallStart {
            index: 0,
            call_id: "tool".into(),
            name: "read_file".into(),
        },
        ModelEvent::Done(FinishReason::ToolUse),
    ]);
    h.append(3, invoke("incomplete", Default::default())).await;
    assert!(completed(&mut rx, "incomplete")
        .await
        .1
        .error
        .unwrap()
        .message
        .contains("incomplete tool"));
    session.close();
}

#[tokio::test]
async fn sse_waits_for_run_and_replays_completed_lifecycle() {
    let h = Harness::new().await;
    let router = h.router.clone();
    let stream = tokio::spawn(async move {
        router
            .oneshot(
                Request::post("/aiserver.v1.InferenceService/RunInferenceSSE")
                    .header("content-type", "application/proto")
                    .body(Body::from(
                        pb::BidiRequestId {
                            request_id: "session".into(),
                        }
                        .encode_to_vec(),
                    ))
                    .unwrap(),
            )
            .await
            .unwrap()
    });
    h.append(0, h.run()).await;
    let response = tokio::time::timeout(Duration::from_secs(3), stream)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()["content-type"], "text/event-stream");
    h.append(
        1,
        pb::run_inference_client_message::Message::FinishRun(Default::default()),
    )
    .await;
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let frames = connect::decode_frames(&bytes).unwrap();
    assert_eq!(frames.last().unwrap().0, connect::END_STREAM_FLAG);
    assert!(frames
        .iter()
        .filter(|(flag, _)| *flag == 0)
        .any(|(_, body)| matches!(
            pb::RunInferenceServerMessage::decode(body.clone())
                .unwrap()
                .message,
            Some(pb::run_inference_server_message::Message::RunReady(_))
        )));
    let replay = h
        .router
        .clone()
        .oneshot(
            Request::post("/aiserver.v1.InferenceService/RunInferenceSSE")
                .body(Body::from(
                    pb::BidiRequestId {
                        request_id: "session".into(),
                    }
                    .encode_to_vec(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(
        bytes,
        axum::body::to_bytes(replay.into_body(), usize::MAX)
            .await
            .unwrap()
    );
}

#[tokio::test]
async fn poll_encodes_sequenced_messages_and_terminal_eof() {
    let h = Harness::new().await;
    h.append(0, h.run()).await;
    h.append(
        1,
        pb::run_inference_client_message::Message::FinishRun(Default::default()),
    )
    .await;
    let response = h
        .router
        .clone()
        .oneshot(
            Request::post("/aiserver.v1.InferenceService/RunInferencePoll")
                .body(Body::from(
                    pb::BidiPollRequest {
                        request_id: Some(pb::BidiRequestId {
                            request_id: "session".into(),
                        }),
                        start_request: Some(true),
                    }
                    .encode_to_vec(),
                ))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let frames = connect::decode_frames(&bytes).unwrap();
    assert_eq!(frames.last().unwrap().0, connect::END_STREAM_FLAG);
    let messages = frames[..frames.len() - 1]
        .iter()
        .enumerate()
        .map(|(index, (flag, body))| {
            assert_eq!(*flag, 0);
            let message = pb::BidiPollResponse::decode(body.clone()).unwrap();
            assert_eq!(message.seqno, index as i64);
            message
        })
        .collect::<Vec<_>>();
    assert_eq!(messages.last().unwrap().eof, Some(true));
    assert!(messages
        .iter()
        .filter(|m| m.eof != Some(true))
        .any(|m| matches!(
            pb::RunInferenceServerMessage::decode(hex::decode(&m.data).unwrap().as_slice())
                .unwrap()
                .message,
            Some(pb::run_inference_server_message::Message::RunReady(_))
        )));
}

#[tokio::test]
async fn unsolicited_delta_is_rejected_without_calling_provider() {
    let h = Harness::new().await;
    let (session, mut rx) = h.start().await;
    h.append(
        1,
        pb::run_inference_client_message::Message::InvokeModel(pb::RunInferenceInvokeModel {
            invocation_id: "delta".into(),
            delta_base: Some(Default::default()),
            ..Default::default()
        }),
    )
    .await;
    assert!(completed(&mut rx, "delta")
        .await
        .1
        .error
        .unwrap()
        .message
        .contains("full uncompressed request"));
    assert!(h.provider.requests().is_empty());
    session.close();
}

#[test]
fn reasoning_replay_survives_cursor_response_and_request_round_trip() {
    use cursor_server::{
        cursor::inference::{request, response},
        model::{ProjectedContent, ProviderReplayState},
    };
    for (provider_kind, value) in [
        (
            "anthropic",
            serde_json::json!({"blocks":[{"type":"thinking","thinking":"reason","signature":"signed"},{"type":"redacted_thinking","data":"opaque"}]}),
        ),
        (
            "openai_responses",
            serde_json::json!({"items":[{"type":"reasoning","id":"rs_1","encrypted_content":"opaque"}]}),
        ),
        (
            "openai_chat",
            serde_json::json!({"reasoning_content":"reason"}),
        ),
    ] {
        let state = ProviderReplayState {
            provider_kind: provider_kind.into(),
            value,
        };
        let mut output = response::Output::default();
        output
            .event(ModelEvent::ThinkingDelta("reason".into()))
            .unwrap();
        output
            .event(ModelEvent::ProviderReplayState(state.clone()))
            .unwrap();
        let Some(pb::inference_stream_response::Response::ResponseInfo(info)) =
            output.finish("invoke", "model").unwrap().response
        else {
            panic!("response info")
        };
        let message = &info.messages[0];
        let request = pb::InferenceStreamRequest {
            messages: vec![pb::InferenceCoreMessage {
                role: pb::InferenceMessageRole::Assistant as i32,
                content: message
                    .content
                    .clone()
                    .map(pb::inference_core_message::Content::Text),
                reasoning_parts: message.reasoning_parts.clone(),
                ..Default::default()
            }],
            ..Default::default()
        };
        let request = request::compile(
            &request,
            &pb::InferenceRequestedModel {
                model_id: "model".into(),
                ..Default::default()
            },
        )
        .unwrap();
        let ProjectedContent::Assistant { replay_state, .. } = &request.history[0].content else {
            panic!("assistant history")
        };
        assert_eq!(replay_state.as_ref(), Some(&state), "{provider_kind}");
    }
}

#[test]
fn images_tool_errors_and_changed_context_preserve_input_order() {
    use cursor_server::{
        cursor::inference::request,
        model::{ContentPart, ProjectedContent},
    };
    let selected = pb::InferenceRequestedModel {
        model_id: "model".into(),
        ..Default::default()
    };
    let mut input = pb::InferenceStreamRequest {
        messages: vec![
            text(pb::InferenceMessageRole::System, "system"),
            text(pb::InferenceMessageRole::User, "context A"),
        ],
        ..Default::default()
    };
    let first = request::compile(&input, &selected).unwrap();
    input
        .messages
        .push(text(pb::InferenceMessageRole::System, "context B"));
    input.messages.push(pb::InferenceCoreMessage {
        role: pb::InferenceMessageRole::User as i32,
        content: Some(pb::inference_core_message::Content::Parts(
            pb::InferenceContentParts {
                parts: vec![pb::InferenceContentPart {
                    part: Some(pb::inference_content_part::Part::Image(
                        pb::InferenceImagePart {
                            data: "data:image/png;base64,AQID".into(),
                            ..Default::default()
                        },
                    )),
                }],
            },
        )),
        ..Default::default()
    });
    let second = request::compile(&input, &selected).unwrap();
    assert_eq!(first.prompt, second.prompt);
    assert_eq!(first.history, second.history[..first.history.len()]);
    let ProjectedContent::Parts(parts) = &second.history[2].content else {
        panic!("image parts")
    };
    assert_eq!(
        parts,
        &vec![ContentPart::Image {
            mime_type: "image/png".into(),
            data: vec![1, 2, 3]
        }]
    );
    input
        .messages
        .push(text(pb::InferenceMessageRole::System, "context A"));
    let third = request::compile(&input, &selected).unwrap();
    assert_eq!(second.history, third.history[..second.history.len()]);
    assert_eq!(third.history.len(), second.history.len() + 1);
}

#[derive(Clone, PartialEq, Message)]
struct RequestedModel {
    #[prost(string, tag = "1")]
    model_id: String,
}
#[derive(Clone, PartialEq, Message)]
struct InferenceRun {
    #[prost(string, tag = "1")]
    conversation_id: String,
    #[prost(message, optional, tag = "3")]
    requested_model: Option<RequestedModel>,
}
#[derive(Clone, PartialEq, Message)]
struct InferenceClient {
    #[prost(message, optional, tag = "1")]
    run_request: Option<InferenceRun>,
}

#[tokio::test]
async fn inference_run_request_is_not_decoded_as_agent_state() {
    let (_directory, store) = fixtures::temp_store().await;
    let input = serde_json::from_value(serde_json::json!({
        "display_name":"Test", "type":"openai", "base_url":"https://example.com",
        "api_key":"test", "tooltip_data":"Test model", "model_id":"gpt-test"
    }))
    .unwrap();
    let model = store.create_model(&input).await.unwrap();
    let assets =
        PromptAssets::load(&std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("prompt/cursor"))
            .unwrap();
    let registry = TransportRegistry::new(
        store.clone(),
        Arc::new(fake_provider::FakeProvider::default()),
        PromptCompiler::new(assets),
    );
    let router = cursor::router(registry, NetworkClients::new(store)).unwrap();
    let payload = InferenceClient {
        run_request: Some(InferenceRun {
            conversation_id: "100d734a-afcd-4460-b001-a28998347595".into(),
            requested_model: Some(RequestedModel {
                model_id: model.model_hash,
            }),
        }),
    }
    .encode_to_vec();
    let append = ai::BidiAppendRequest {
        request_id: Some(ai::BidiRequestId {
            request_id: "inference-test".into(),
        }),
        data: hex::encode(payload),
        ..Default::default()
    };
    let response = router
        .oneshot(
            Request::post("/aiserver.v1.BidiService/BidiAppend")
                .header("content-type", "application/proto")
                .body(Body::from(append.encode_to_vec()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
}
