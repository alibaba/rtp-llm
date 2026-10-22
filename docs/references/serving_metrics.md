# Model-server latency histograms

Set `RTP_LLM_METRICS_ENABLED=1` to expose Prometheus histograms at `GET /metrics`
on the Python frontend HTTP port. This is independent of tracing: no
`RTP_LLM_TRACE_CONFIG`, OTLP receiver, access-log reader, or KMonitor gateway is
required. Existing KMonitor gauges keep their names, units, and reporting paths.
The new histograms observe individual requests; they are not reconstructed from
aggregated gauges.

These metrics follow the types, units, and recommended buckets in the
[OpenTelemetry GenAI model-server semantic conventions](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-metrics.md#generative-ai-model-server-metrics).
Those are public, community-maintained conventions with **Development** stability,
not a finalized stable standard. The same three histogram definitions are used by
[Kubernetes SIGs model-serving-signals](https://github.com/kubernetes-sigs/model-serving-signals/blob/main/model-server-contract.md)
to normalize observations across inference engines.

| Prometheus histogram | OpenTelemetry metric | Observation |
| --- | --- | --- |
| `rtp_llm_server_request_duration_seconds` | `gen_ai.server.request.duration` | From frontend ASGI request entry to the last nonempty response body handed to the server transport. Streaming and non-streaming chat requests are included. Aborted requests use the abort time. |
| `rtp_llm_server_time_to_first_token_seconds` | `gen_ai.server.time_to_first_token` | From the same request entry to the first visible token-bearing streaming response produced by the frontend. Includes frontend processing, engine queuing and prefill. Recorded only after successful completion. |
| `rtp_llm_server_time_per_output_token_seconds` | `gen_ai.server.time_per_output_token` | One mean per successful request: `(last_token_time - first_token_time) / (output_tokens - 1)`, measured at the same frontend token boundary. |

Each histogram exposes cumulative `_bucket` samples (including `le="+Inf"`),
`_sum`, and `_count` in **seconds**. Labels include `gen_ai_operation_name="chat"`
and `gen_ai_provider_name="rtp_llm"`. Duration also has a bounded `error_type`:
empty on success, `client_error`, `server_error`, `internal_error`, or `cancelled`.
An error encoded inside an HTTP 200 SSE response is an error, not a successful
token-timing observation. Request IDs, prompts, and arbitrary request model names
are not metric labels. These are RTP-LLM-native metric names with an explicit
mapping to the common conventions, rather than a complete OTel instrumentation.

## Start and scrape

Python frontend processes share a port through `SO_REUSEPORT`. Enable the
Prometheus client's multiprocess storage **before** launching any Python process,
so any frontend's `/metrics` returns the combined histogram counts:

```sh
# Use a fresh directory for each complete server run. All frontend processes of
# this instance must inherit the same directory; separate instances must not.
export PROMETHEUS_MULTIPROC_DIR="$(mktemp -d /tmp/rtp-llm-metrics.XXXXXX)"
export RTP_LLM_METRICS_ENABLED=1
python -m rtp_llm.start_server \
  --checkpoint_path=/path/to/model --model_type=qwen_3 --start_port=8000
```

```sh
curl http://localhost:8000/metrics
```

Scrape one endpoint per shared multiprocess directory, not every process as a
separate target. Histogram data from a restarted worker remains in that directory
so cumulative counts do not drop. Start with a fresh directory on a full instance
restart; clean up old directories only after all their workers have stopped.
A pod-local volume can hold this directory. A scraping sidecar needs HTTP access
only; it does not need to mount the directory. Initialization fails explicitly
when metrics are enabled without a valid multiprocess directory. Without the
opt-in flag, the endpoint and collection are disabled.

## Coverage and timing boundaries

- Covers `POST /v1/chat/completions` and `/chat/completions`. Native inference,
  embeddings, batch chat, and DashScope gRPC are outside this initial scope.
- TTFT and TPOT are currently available for **streaming, single-choice** requests.
  Non-streaming output may already be merged by the engine, so it does not expose
  the frontend first-token boundary. Duration still covers these requests.
- Role-only, empty, usage-only, and finish-only chunks do not count as tokens.
  TPOT requires a reliable increasing cumulative token count, a first visible
  chunk containing exactly one token, and a later token delivery. Missing counts,
  rebased counts, a multi-token first chunk, or a single-token response omit TPOT.
  Missing timing observations are never replaced with zero.
- Token timings are taken when frontend output is available, before ASGI body
  serialization; duration uses actual ASGI send completion. Post-body access-log
  and cleanup work does not extend duration. These are frontend observations,
  distinct from engine-only KMonitor latency and client-side network timing.
- Metrics are independent of trace enablement, sampling, and trace export loss.
  Counts reset on a full server restart with a fresh multiprocess directory.
