//! Verifies negotiated upstream protocols and streaming connection reuse over TLS.
#[path = "network_transport/benchmark.rs"]
mod benchmark;
#[path = "support/fixtures.rs"]
mod fixtures;

use std::{
    convert::Infallible,
    net::SocketAddr,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};

use axum::body::Body;
use cursor_server::network::client_builder;
use futures_util::StreamExt;
use hudsucker::{
    hyper::{self, service::service_fn, Request, Response},
    hyper_util::rt::{TokioExecutor, TokioIo},
    rustls,
};
use tokio::{net::TcpListener, sync::Notify};
use tokio_rustls::TlsAcceptor;
use tokio_util::sync::CancellationToken;

struct Upstream {
    address: SocketAddr,
    certificate: reqwest::Certificate,
    connections: Arc<AtomicUsize>,
    finish_stream: Arc<Notify>,
    stop: CancellationToken,
}

impl Drop for Upstream {
    fn drop(&mut self) {
        self.stop.cancel();
    }
}

impl Upstream {
    async fn start(protocols: &[&[u8]]) -> Self {
        let rcgen::CertifiedKey { cert, signing_key } =
            rcgen::generate_simple_self_signed(vec!["localhost".into()]).unwrap();
        let certificate = reqwest::Certificate::from_der(cert.der().as_ref()).unwrap();
        let mut config = rustls::ServerConfig::builder_with_provider(Arc::new(
            rustls::crypto::aws_lc_rs::default_provider(),
        ))
        .with_safe_default_protocol_versions()
        .unwrap()
        .with_no_client_auth()
        .with_single_cert(
            vec![cert.der().clone()],
            rustls::pki_types::PrivatePkcs8KeyDer::from(signing_key.serialize_der()).into(),
        )
        .unwrap();
        config.alpn_protocols = protocols.iter().map(|p| p.to_vec()).collect();
        let acceptor = TlsAcceptor::from(Arc::new(config));
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let connections = Arc::new(AtomicUsize::new(0));
        let finish_stream = Arc::new(Notify::new());
        let stop = CancellationToken::new();
        let count = connections.clone();
        let gate = finish_stream.clone();
        let cancelled = stop.clone();
        tokio::spawn(async move {
            loop {
                let accepted = tokio::select! { _ = cancelled.cancelled() => break, result = listener.accept() => result };
                let Ok((socket, _)) = accepted else { break };
                socket.set_nodelay(true).unwrap();
                count.fetch_add(1, Ordering::Relaxed);
                let acceptor = acceptor.clone();
                let gate = gate.clone();
                let cancelled = cancelled.clone();
                tokio::spawn(async move {
                    let tls = tokio::select! { _ = cancelled.cancelled() => return, result = acceptor.accept(socket) => result };
                    let Ok(tls) = tls else { return };
                    let h2 = tls.get_ref().1.alpn_protocol() == Some(b"h2".as_slice());
                    let service = service_fn(move |request: Request<hyper::body::Incoming>| {
                        let gate = gate.clone();
                        async move {
                            let body = if request.uri().path() == "/stream" {
                                Body::from_stream(async_stream::stream! {
                                    yield Ok::<_, Infallible>("first");
                                    gate.notified().await;
                                    yield Ok::<_, Infallible>("last");
                                })
                            } else if request.uri().path() == "/benchmark" {
                                Body::from_stream(async_stream::stream! {
                                    for _ in 0..8 {
                                        tokio::time::sleep(Duration::from_millis(2)).await;
                                        yield Ok::<_, Infallible>(bytes::Bytes::from_static(b"data: {\"delta\":\"token\"}\n\n"));
                                    }
                                })
                            } else {
                                Body::from("ok")
                            };
                            Ok::<_, Infallible>(Response::new(body))
                        }
                    });
                    let serve = async move {
                        if h2 {
                            let _ = hyper::server::conn::http2::Builder::new(TokioExecutor::new())
                                .serve_connection(TokioIo::new(tls), service)
                                .await;
                        } else {
                            let _ = hyper::server::conn::http1::Builder::new()
                                .serve_connection(TokioIo::new(tls), service)
                                .await;
                        }
                    };
                    tokio::select! { _ = cancelled.cancelled() => {}, _ = serve => {} }
                });
            }
        });
        Self {
            address,
            certificate,
            connections,
            finish_stream,
            stop,
        }
    }

    fn url(&self, path: &str) -> String {
        format!("https://localhost:{}{path}", self.address.port())
    }
}

#[tokio::test]
async fn native_tls_negotiates_h2_and_multiplexes_while_a_stream_is_open() {
    let (_directory, store) = fixtures::temp_store().await;
    let upstream = Upstream::start(&[b"h2", b"http/1.1"]).await;
    let client = client_builder(&store)
        .await
        .unwrap()
        .no_proxy()
        .add_root_certificate(upstream.certificate.clone())
        .build()
        .unwrap();
    let response = client.get(upstream.url("/warmup")).send().await.unwrap();
    assert_eq!(response.version(), reqwest::Version::HTTP_2);
    assert_eq!(response.text().await.unwrap(), "ok");
    let response = client.get(upstream.url("/stream")).send().await.unwrap();
    let mut stream = response.bytes_stream();
    let first = tokio::time::timeout(Duration::from_secs(3), stream.next())
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(&first[..], b"first");
    let other = tokio::time::timeout(
        Duration::from_secs(3),
        client.get(upstream.url("/other")).send(),
    )
    .await
    .unwrap()
    .unwrap();
    assert_eq!(other.version(), reqwest::Version::HTTP_2);
    assert_eq!(other.text().await.unwrap(), "ok");
    assert_eq!(
        upstream.connections.load(Ordering::Relaxed),
        1,
        "parallel requests must share the existing TLS connection"
    );
    upstream.finish_stream.notify_one();
    let mut rest = Vec::new();
    while let Some(chunk) = tokio::time::timeout(Duration::from_secs(3), stream.next())
        .await
        .unwrap()
    {
        rest.extend_from_slice(&chunk.unwrap());
    }
    assert_eq!(rest, b"last");
}

#[tokio::test]
async fn upstream_can_select_http11_and_reuse_its_connection() {
    let (_directory, store) = fixtures::temp_store().await;
    let upstream = Upstream::start(&[b"http/1.1"]).await;
    let client = client_builder(&store)
        .await
        .unwrap()
        .no_proxy()
        .add_root_certificate(upstream.certificate.clone())
        .build()
        .unwrap();
    for _ in 0..2 {
        let response = client.get(upstream.url("/request")).send().await.unwrap();
        assert_eq!(response.version(), reqwest::Version::HTTP_11);
        assert_eq!(response.text().await.unwrap(), "ok");
    }
    assert_eq!(upstream.connections.load(Ordering::Relaxed), 1);
}
