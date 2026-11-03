//! Runs client-owned Cursor inference sessions.
pub mod request;
pub mod response;
mod session;

use crate::{
    cursor::protocol::proto::{aiserver::v1 as ai, inference as pb},
    provider::Provider,
    store::Store,
    Error, Result,
};
use prost::Message;
pub use session::Session;
use std::{collections::HashMap, sync::Arc, time::Duration};
use tokio::sync::Mutex;

#[derive(Clone)]
pub struct InferenceRegistry {
    sessions: Arc<Mutex<HashMap<String, Session>>>,
    store: Store,
    provider: Arc<dyn Provider>,
}

impl InferenceRegistry {
    pub fn new(store: Store, provider: Arc<dyn Provider>) -> Self {
        Self {
            sessions: Arc::default(),
            store,
            provider,
        }
    }

    pub async fn contains(&self, id: &str) -> bool {
        self.sessions.lock().await.contains_key(id)
    }

    pub async fn shutdown(&self) {
        for session in std::mem::take(&mut *self.sessions.lock().await).into_values() {
            session.close();
        }
    }

    pub async fn session(&self, id: &str) -> Result<Session> {
        if id.is_empty() {
            return Err(Error::Protocol("inference request_id is required".into()));
        }
        let mut sessions = self.sessions.lock().await;
        if let Some(session) = sessions.get(id) {
            return Ok(session.clone());
        }
        let session = Session::new(id.to_owned(), self.store.clone(), self.provider.clone());
        sessions.insert(id.into(), session.clone());
        let handle = session.clone();
        let registry = Arc::downgrade(&self.sessions);
        let id = id.to_owned();
        tokio::spawn(async move {
            handle.maintain().await;
            // Keep completed output briefly for an SSE reconnect, then release all request state.
            tokio::time::sleep(Duration::from_secs(30)).await;
            if let Some(registry) = registry.upgrade() {
                registry.lock().await.remove(&id);
            }
        });
        Ok(session)
    }

    /// BidiAppend is shared by both services. Only recognize a complete, valid
    /// inference run header; its UUID string distinguishes it from Agent state.
    pub fn is_run(payload: &[u8]) -> bool {
        matches!(pb::RunInferenceClientMessage::decode(payload), Ok(pb::RunInferenceClientMessage {
            message: Some(pb::run_inference_client_message::Message::RunRequest(run)),
        }) if uuid::Uuid::parse_str(&run.conversation_id).is_ok()
            && run.requested_model.as_ref().is_some_and(|model| !model.model_id.is_empty()))
    }

    pub async fn append(&self, request: &ai::BidiAppendRequest, payload: &[u8]) -> Result<bool> {
        let id = request
            .request_id
            .as_ref()
            .map(|v| v.request_id.as_str())
            .unwrap_or_default();
        let session = self.session(id).await?;
        session
            .append(
                request.append_seqno,
                pb::RunInferenceClientMessage::decode(payload)?,
            )
            .await
    }
}
