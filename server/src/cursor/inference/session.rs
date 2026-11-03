//! Owns one inference lifecycle; model invocations share its ordered input and output.
use super::{request, response};
use crate::{
    cursor::{
        protocol::{connect, proto::inference as pb},
        transport::{OrderedInbox, OutputHub},
    },
    model::{ModelInvocation, ModelType},
    provider::{ModelEvent, Provider},
    store::Store,
    Error, Result,
};
use futures_util::StreamExt;
use std::{
    collections::{HashMap, HashSet},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
use tokio::sync::{watch, Mutex};
use tokio_util::sync::CancellationToken;

#[derive(Clone)]
pub struct Session {
    inner: Arc<Inner>,
}
struct Inner {
    id: String,
    store: Store,
    provider: Arc<dyn Provider>,
    state: Mutex<State>,
    route: watch::Sender<Option<bool>>,
    output: OutputHub,
    closed: CancellationToken,
    invocations: parking_lot::Mutex<HashMap<String, CancellationToken>>,
    touched: parking_lot::Mutex<Instant>,
    subscribers: AtomicUsize,
}
struct State {
    inbox: OrderedInbox<pb::RunInferenceClientMessage>,
    next_seqno: i64,
    run: Option<pb::RunInferenceRunRequest>,
    seen: HashSet<String>,
    call_index: u64,
}

impl Session {
    pub(super) fn new(id: String, store: Store, provider: Arc<dyn Provider>) -> Self {
        let (route, _) = watch::channel(None);
        Self {
            inner: Arc::new(Inner {
                id,
                store,
                provider,
                route,
                output: OutputHub::default(),
                state: Mutex::new(State {
                    inbox: OrderedInbox::starting_at(0),
                    next_seqno: 0,
                    run: None,
                    seen: HashSet::new(),
                    call_index: 0,
                }),
                closed: CancellationToken::new(),
                invocations: parking_lot::Mutex::new(HashMap::new()),
                touched: parking_lot::Mutex::new(Instant::now()),
                subscribers: AtomicUsize::new(0),
            }),
        }
    }

    pub async fn local(&self) -> Result<bool> {
        let mut route = self.inner.route.subscribe();
        tokio::time::timeout(Duration::from_secs(30), async {
            loop {
                if let Some(local) = *route.borrow_and_update() {
                    return Ok(local);
                }
                route.changed().await.map_err(|_| Error::Cancelled)?;
            }
        })
        .await
        .map_err(|_| Error::Protocol("inference stream timed out waiting for run_request".into()))?
    }

    pub fn subscribe(&self) -> tokio::sync::mpsc::UnboundedReceiver<bytes::Bytes> {
        self.inner.subscribers.fetch_add(1, Ordering::Relaxed);
        self.inner.output.subscribe()
    }

    pub fn disconnect(&self) {
        if self.inner.subscribers.fetch_sub(1, Ordering::Relaxed) != 1 {
            return;
        }
        let session = self.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_secs(10)).await;
            if session.inner.subscribers.load(Ordering::Relaxed) == 0 {
                session.close();
            }
        });
    }

    pub fn close(&self) {
        self.inner.closed.cancel();
        for token in self.inner.invocations.lock().values() {
            token.cancel();
        }
        self.inner.output.emit(connect::encode_end_stream());
        self.inner.output.close();
    }

    pub(super) async fn maintain(&self) {
        let mut heartbeat = tokio::time::interval(Duration::from_secs(15));
        loop {
            tokio::select! {
                _ = self.inner.closed.cancelled() => return,
                _ = heartbeat.tick() => {
                    if self.inner.touched.lock().elapsed() > Duration::from_secs(120) { self.close(); return; }
                    if *self.inner.route.borrow() == Some(true) {
                        self.emit(pb::run_inference_server_message::Message::Heartbeat(pb::RunInferenceHeartbeat::default()));
                    }
                }
            }
        }
    }

    pub async fn append(&self, seqno: i64, message: pb::RunInferenceClientMessage) -> Result<bool> {
        *self.inner.touched.lock() = Instant::now();
        let mut state = self.inner.state.lock().await;
        if seqno < 0 || seqno > state.next_seqno.saturating_add(128) {
            return Err(Error::Protocol(
                "inference append sequence is outside the admission window".into(),
            ));
        }
        if self.inner.closed.is_cancelled() {
            return if seqno < state.next_seqno {
                Ok(true)
            } else {
                Err(Error::Cancelled)
            };
        }
        // Official inference messages remain opaque, including unknown future fields.
        if *self.inner.route.borrow() == Some(false) {
            return Ok(false);
        }
        for (seqno, message) in state.inbox.push(seqno, message) {
            state.next_seqno = seqno.saturating_add(1);
            use pb::run_inference_client_message::Message as M;
            match message
                .message
                .ok_or_else(|| Error::Protocol("empty inference client message".into()))?
            {
                M::RunRequest(run) => {
                    if state.run.is_some() {
                        return Err(Error::Protocol(
                            "inference lifecycle already started".into(),
                        ));
                    }
                    let selected = run
                        .requested_model
                        .as_ref()
                        .filter(|v| !v.model_id.is_empty())
                        .ok_or_else(|| {
                            Error::Protocol("inference run must select a model".into())
                        })?;
                    let configured = self.inner.store.model(&selected.model_id).await?;
                    let local = configured.is_some()
                        || selected
                            .model_id
                            .starts_with(crate::plugin::ADAPTER_ID_PREFIX);
                    self.inner.route.send_replace(Some(local));
                    if !local {
                        return Ok(false);
                    }
                    let model_id = configured
                        .as_ref()
                        .map(|m| m.model_id.as_str())
                        .unwrap_or(&selected.model_id);
                    let vendor = match configured.as_ref().map(|m| m.model_type) {
                        Some(ModelType::Anthropic) => "anthropic",
                        _ => "openai",
                    };
                    let metadata = pb::RunInferencePromptModelMetadata {
                        vendor: vendor.into(),
                        prompt_version: run
                            .agent_module
                            .as_ref()
                            .and_then(|m| m.prompt_versions.first())
                            .cloned()
                            .unwrap_or_default(),
                        is_gpt5_family: model_id.contains("gpt-5"),
                        is_codex_family: model_id.contains("codex"),
                        is_claude_4x: vendor == "anthropic",
                        agent_token_limit: configured
                            .as_ref()
                            .and_then(|m| m.context_window_tokens)
                            .map(|v| v.min(i32::MAX as u64) as i32),
                        ..Default::default()
                    };
                    tracing::info!(
                        request_id = self.inner.id,
                        conversation_id = run.conversation_id,
                        model_id = selected.model_id,
                        vendor,
                        "routing Cursor RunInference to BYOK provider"
                    );
                    self.emit(pb::run_inference_server_message::Message::RunReady(
                        pb::RunInferenceRunReady {
                            resolved_model: Some(selected.clone()),
                            prompt_model_metadata: Some(metadata),
                            routed_model_display_name: configured
                                .as_ref()
                                .map(|m| m.display_name.clone()),
                            accepts_client_heartbeat: true,
                            accepts_aborted_finish: true,
                            accepts_subagent_model_resolution: true,
                            // Full requests preserve client history; do not negotiate deltas or compression.
                            accepts_delta_invocations: false,
                            summarization: Some(pb::RunInferenceSummarizationConfig {
                                model: Some(selected.clone()),
                                is_fallback_to_main_model: true,
                                ..Default::default()
                            }),
                            ..Default::default()
                        },
                    ));
                    state.run = Some(run);
                }
                M::InvokeModel(invoke) => {
                    let run = state
                        .run
                        .as_ref()
                        .ok_or_else(|| {
                            Error::Protocol("inference invocation precedes run_request".into())
                        })?
                        .clone();
                    if invoke.invocation_id.is_empty()
                        || !state.seen.insert(invoke.invocation_id.clone())
                    {
                        return Err(Error::Protocol(
                            "inference invocation_id must be nonempty and unique".into(),
                        ));
                    }
                    state.call_index += 1;
                    self.invoke(run, invoke, state.call_index).await;
                }
                M::CancelInvocation(cancel) => {
                    if let Some(token) = self.inner.invocations.lock().get(&cancel.invocation_id) {
                        token.cancel();
                    }
                }
                M::FinishRun(_) => {
                    self.close();
                    break;
                }
                M::ResolveSubagentModels(resolve) => {
                    let main = state
                        .run
                        .as_ref()
                        .and_then(|r| r.requested_model.as_ref())
                        .map(|m| m.model_id.as_str());
                    let mut models = Vec::new();
                    for requested in resolve.requested_models {
                        let resolved_slug = if requested == "default" {
                            main.map(str::to_owned)
                        } else if requested.starts_with(crate::plugin::ADAPTER_ID_PREFIX)
                            || self.inner.store.model(&requested).await?.is_some()
                        {
                            Some(requested.clone())
                        } else {
                            None
                        };
                        models.push(pb::RunInferenceResolvedSubagentModel {
                            requested,
                            resolved_slug,
                        });
                    }
                    self.emit(
                        pb::run_inference_server_message::Message::SubagentModelsResolved(
                            pb::RunInferenceSubagentModelsResolved {
                                resolution_id: resolve.resolution_id,
                                models,
                            },
                        ),
                    );
                }
                M::ClientHeartbeat(_)
                | M::AgentEvent(_)
                | M::RunRequestLog(_)
                | M::ClientFirstCallTimings(_) => {}
            }
        }
        Ok(true)
    }

    async fn invoke(
        &self,
        run: pb::RunInferenceRunRequest,
        invoke: pb::RunInferenceInvokeModel,
        index: u64,
    ) {
        let id = invoke.invocation_id;
        let prepared = async {
            if invoke.delta_base.is_some() || invoke.compressed_request.is_some() {
                return Err(Error::Protocol(
                    "RunInference requires a full uncompressed request as negotiated in RunReady"
                        .into(),
                ));
            }
            let request = invoke.request.ok_or_else(|| {
                Error::Protocol("inference invocation contains no request".into())
            })?;
            let mut selected = request
                .requested_model
                .clone()
                .or(run.requested_model)
                .ok_or_else(|| Error::Protocol("inference invocation has no model".into()))?;
            if let Some(model_id) = request.model_id.as_ref().filter(|id| !id.is_empty()) {
                selected.model_id = model_id.clone();
            }
            if !selected
                .model_id
                .starts_with(crate::plugin::ADAPTER_ID_PREFIX)
                && self.inner.store.model(&selected.model_id).await?.is_none()
            {
                return Err(Error::Protocol(
                    "inference invocation must select a configured BYOK model".into(),
                ));
            }
            Ok(ModelInvocation {
                call_id: uuid::Uuid::new_v4().to_string(),
                run_id: self.inner.id.clone(),
                conversation_id: run.conversation_id,
                provider_call_index: index,
                request: request::compile(&request, &selected)?,
            })
        }
        .await;
        let invocation = match prepared {
            Ok(invocation) => invocation,
            Err(error) => {
                self.send(response::end(&id, Some(&error)));
                return;
            }
        };
        let token = self.inner.closed.child_token();
        self.inner
            .invocations
            .lock()
            .insert(id.clone(), token.clone());
        let session = self.clone();
        tokio::spawn(async move {
            let started = Instant::now();
            let model = invocation.request.model.model_id.clone();
            tracing::info!(
                request_id = session.inner.id,
                invocation_id = id,
                model_id = model,
                message_count = invocation.request.history.len(),
                tool_count = invocation.request.prompt.tools.len(),
                "starting Cursor inference invocation"
            );
            let mut stream = session.inner.provider.stream(invocation, token.clone());
            let mut output = response::Output::default();
            let result = async {
                loop {
                    let event = tokio::select! { _ = token.cancelled() => return Err(Error::Cancelled), event = stream.next() => event };
                    let event = event.ok_or_else(|| Error::Provider("inference provider stream ended without completion".into()))??;
                    let done = matches!(event, ModelEvent::Done(_));
                    for response in output.event(event)? { session.send(response::wrap(&id, response)); }
                    if done { session.send(response::wrap(&id, output.finish(&id, &model)?)); return Ok(()); }
                }
            }.await;
            if let Err(error) = &result {
                tracing::warn!(request_id = session.inner.id, invocation_id = id, %error, "Cursor inference invocation failed");
            }
            tracing::info!(
                request_id = session.inner.id,
                invocation_id = id,
                elapsed_ms = started.elapsed().as_millis() as u64,
                success = result.is_ok(),
                "Cursor inference invocation completed"
            );
            session.send(response::end(&id, result.as_ref().err()));
            session.inner.invocations.lock().remove(&id);
        });
    }

    fn emit(&self, message: pb::run_inference_server_message::Message) {
        self.send(pb::RunInferenceServerMessage {
            message: Some(message),
        });
    }
    fn send(&self, message: pb::RunInferenceServerMessage) {
        match connect::encode_message(&message) {
            Ok(frame) => {
                self.inner.output.emit(frame);
            }
            Err(error) => {
                tracing::error!(%error, "cannot encode inference response");
                self.close();
            }
        }
    }
}
