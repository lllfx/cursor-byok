//! Bridges RunInference SSE/Poll subscriptions to request-scoped inference sessions.
use super::proxy::{self, CursorProxy};
use crate::{
    cursor::{
        inference::Session,
        protocol::{connect, proto::inference as pb},
        transport::TransportRegistry,
    },
    Result,
};
use axum::{
    body::Body,
    extract::{Extension, State},
    http::{header, Request, Response},
};
use bytes::Bytes;
use std::convert::Infallible;

pub async fn sse(
    State(registry): State<TransportRegistry>,
    Extension(proxy): Extension<CursorProxy>,
    request: Request<Body>,
) -> Result<Response<Body>> {
    subscribe(registry, proxy, request, false).await
}

pub async fn poll(
    State(registry): State<TransportRegistry>,
    Extension(proxy): Extension<CursorProxy>,
    request: Request<Body>,
) -> Result<Response<Body>> {
    subscribe(registry, proxy, request, true).await
}

async fn subscribe(
    registry: TransportRegistry,
    proxy: CursorProxy,
    request: Request<Body>,
    poll: bool,
) -> Result<Response<Body>> {
    let (parts, body) = super::handlers::buffered(request).await?;
    let id = if poll {
        connect::decode_unary::<pb::BidiPollRequest>(&body)?
            .request_id
            .map(|v| v.request_id)
            .unwrap_or_default()
    } else {
        connect::decode_unary::<pb::BidiRequestId>(&body)?.request_id
    };
    let session = registry.inference().session(&id).await?;
    let local = session.local().await?;
    if !local {
        return proxy::forward(
            Extension(proxy),
            Request::from_parts(parts, Body::from(body)),
        )
        .await;
    }
    let mut receiver = session.subscribe();
    let subscription = Subscription(session);
    let stream = async_stream::stream! {
        let _guard = subscription;
        let mut seqno = 0_i64;
        while let Some(frame) = receiver.recv().await {
            let terminal = frame.first().is_some_and(|flag| flag & connect::END_STREAM_FLAG != 0);
            let bytes = if poll {
                let message = pb::BidiPollResponse { seqno, data: if terminal { String::new() } else { hex::encode(&frame[5..]) }, eof: terminal.then_some(true) };
                match connect::encode_message(&message) { Ok(frame) => frame, Err(_) => break }
            } else { frame };
            yield Ok::<Bytes, Infallible>(bytes);
            seqno += 1;
            if terminal {
                if poll { yield Ok(connect::encode_end_stream()); }
                break;
            }
        }
    };
    Ok(Response::builder()
        .header(header::CONTENT_TYPE, "text/event-stream")
        .header(header::CACHE_CONTROL, "no-cache")
        .header("connect-protocol-version", "1")
        .body(Body::from_stream(stream))
        .expect("static inference response headers"))
}

struct Subscription(Session);
impl Drop for Subscription {
    fn drop(&mut self) {
        self.0.disconnect();
    }
}
