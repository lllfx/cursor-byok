//! Manual loopback benchmark; models controlled output, not an external model API.
use std::{
    sync::atomic::Ordering,
    time::{Duration, Instant},
};

use futures_util::{future::join_all, StreamExt};
use serde_json::json;

use super::{client_builder, fixtures, Upstream};

const TRIALS: usize = 7;
const REUSED_BATCHES: usize = 4;

#[derive(Default)]
struct Measurements {
    first_chunk_ms: Vec<f64>,
    batch_ms: Vec<f64>,
    connections: Vec<usize>,
}

impl Measurements {
    async fn batch(&mut self, client: &reqwest::Client, upstream: &Upstream, concurrency: usize) {
        let started = Instant::now();
        let requests = (0..concurrency).map(|_| {
            let url = upstream.url("/benchmark");
            async move {
                let started = Instant::now();
                let response = client.get(url).send().await.unwrap();
                let mut stream = response.bytes_stream();
                let first = stream.next().await.unwrap().unwrap();
                assert!(!first.is_empty());
                let first_chunk_ms = started.elapsed().as_secs_f64() * 1000.0;
                let mut bytes = first.len();
                while let Some(chunk) = stream.next().await {
                    bytes += chunk.unwrap().len();
                }
                assert_eq!(bytes, 8 * b"data: {\"delta\":\"token\"}\n\n".len());
                first_chunk_ms
            }
        });
        self.first_chunk_ms.extend(join_all(requests).await);
        self.batch_ms.push(started.elapsed().as_secs_f64() * 1000.0);
    }

    fn print(&mut self, protocol: &str, scenario: &str, concurrency: usize) {
        self.first_chunk_ms.sort_by(f64::total_cmp);
        self.batch_ms.sort_by(f64::total_cmp);
        self.connections.sort_unstable();
        println!(
            "LOCAL_TLS_BENCHMARK {}",
            json!({
                "protocol":protocol, "scenario":scenario, "concurrency":concurrency,
                "samples":self.first_chunk_ms.len(),
                "first_chunk_p50_ms":percentile(&self.first_chunk_ms, 50),
                "first_chunk_p95_ms":percentile(&self.first_chunk_ms, 95),
                "batch_p50_ms":percentile(&self.batch_ms, 50),
                "tls_connections_p50":self.connections[(self.connections.len()-1)/2],
            })
        );
    }
}

fn percentile(sorted: &[f64], percentage: usize) -> f64 {
    sorted[(sorted.len() * percentage).div_ceil(100).saturating_sub(1)]
}

#[tokio::test]
#[ignore = "manual controlled TLS comparison; timings are not CI assertions"]
async fn compare_http11_and_http2() {
    let (_directory, store) = fixtures::temp_store().await;
    let upstream = Upstream::start(&[b"h2", b"http/1.1"]).await;
    for concurrency in [1, 8, 32] {
        let mut h1_burst = Measurements::default();
        let mut h1_reused = Measurements::default();
        let mut h2_burst = Measurements::default();
        let mut h2_reused = Measurements::default();
        for trial in 0..TRIALS {
            // Alternate order so one protocol does not always get the cooler CPU.
            let order = if trial % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            };
            for h2 in order {
                let before = upstream.connections.load(Ordering::Relaxed);
                let builder = client_builder(&store)
                    .await
                    .unwrap()
                    .no_proxy()
                    .add_root_certificate(upstream.certificate.clone())
                    .timeout(Duration::from_secs(10));
                let client = if h2 { builder } else { builder.http1_only() }
                    .build()
                    .unwrap();
                let warmup = client.get(upstream.url("/warmup")).send().await.unwrap();
                assert_eq!(
                    warmup.version(),
                    if h2 {
                        reqwest::Version::HTTP_2
                    } else {
                        reqwest::Version::HTTP_11
                    }
                );
                warmup.bytes().await.unwrap();
                let (burst, reused) = if h2 {
                    (&mut h2_burst, &mut h2_reused)
                } else {
                    (&mut h1_burst, &mut h1_reused)
                };
                burst.batch(&client, &upstream, concurrency).await;
                burst
                    .connections
                    .push(upstream.connections.load(Ordering::Relaxed) - before);
                for _ in 0..REUSED_BATCHES {
                    reused.batch(&client, &upstream, concurrency).await;
                    reused
                        .connections
                        .push(upstream.connections.load(Ordering::Relaxed) - before);
                }
            }
        }
        h1_burst.print("http1.1", "first_concurrent_burst", concurrency);
        h2_burst.print("http2", "first_concurrent_burst", concurrency);
        h1_reused.print("http1.1", "reused_pool", concurrency);
        h2_reused.print("http2", "reused_pool", concurrency);
    }
}
