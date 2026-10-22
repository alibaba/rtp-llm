# Model-server latency histograms

Set `RTP_LLM_METRICS_ENABLED=1` to collect explicit-bucket latency histograms
through the existing KMonitor Python client and its Flume reporting worker.
No Prometheus client, HTTP metrics endpoint, multiprocess directory, tracing,
or access-log reader is added. Existing KMonitor gauges keep their names,
units, and reporting paths. The new histograms observe individual requests;
they are not reconstructed from aggregated gauges.

These metrics follow the types, units, and recommended buckets in the
[OpenTelemetry GenAI model-server semantic conventions](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-metrics.md#generative-ai-model-server-metrics).
Those are public, community-maintained conventions with **Development** stability,
not a finalized stable standard. The same three histogram definitions are used by
[Kubernetes SIGs model-serving-signals](https://github.com/kubernetes-sigs/model-serving-signals/blob/main/model-server-contract.md)
to normalize observations across inference engines.

| KMonitor histogram family | OpenTelemetry metric | Observation |
| --- | --- | --- |
| `rtp_llm_server_request_duration_seconds` | `gen_ai.server.request.duration` | From frontend ASGI request entry to the last nonempty response body handed to the server transport. Streaming and non-streaming chat requests are included. Aborted requests use the abort time. |
| `rtp_llm_server_time_to_first_token_seconds` | `gen_ai.server.time_to_first_token` | From the same request entry to the first visible token-bearing streaming response produced by the frontend. Includes frontend processing, engine queuing and prefill. Recorded only after successful completion. |
| `rtp_llm_server_time_per_output_token_seconds` | `gen_ai.server.time_per_output_token` | One mean per successful request: `(last_token_time - first_token_time) / (output_tokens - 1)`, measured at the same frontend token boundary. |

Each histogram reports cumulative `_bucket` samples (including `le="+Inf"`),
`_sum`, and `_count`. Observed values and `_sum` use **seconds**; bucket counts
and `_count` count observations. Labels include `gen_ai_operation_name="chat"`
and `gen_ai_provider_name="rtp_llm"`. Duration also has a bounded `error_type`:
`none` on success, `client_error`, `server_error`, `internal_error`, or `cancelled`.
An error encoded inside an HTTP 200 SSE response is an error, not a successful
token-timing observation. Request IDs, prompts, and arbitrary request model names
are not metric labels. These are RTP-LLM-native metric names with an explicit
mapping to the common conventions, rather than a complete OTel instrumentation.

## Collection and transport

```sh
export RTP_LLM_METRICS_ENABLED=1
python -m rtp_llm.start_server \
  --checkpoint_path=/path/to/model --model_type=qwen_3 --start_port=8000
```

This flag enables observations, not a new exporter. Delivery uses the existing
KMonitor configuration: the Python report worker connects to
`HIPPO_SLAVE_IP:4141` when `HIPPO_ROLE_SHORT_NAME` or `HIPPO_ROLE` is nonempty.
Outside that configuration, the existing worker does not deliver metrics.
An operational KMonitor/Flume collector is therefore required; setting the flag
alone on a standalone deployment does not create a scrape endpoint.

`KMonitor.register_histogram_metric(name, buckets, tags)` registers a composite
metric. `report(value, tags)` places each finite, nonnegative observation into
one explicit bucket. Each periodic report emits numeric components using the
existing `py_kmonitor` Flume event format:

```text
rtp_llm_server_request_duration_seconds_bucket <timestamp> 3 le=0.64 ...
rtp_llm_server_request_duration_seconds_bucket <timestamp> 5 le=+Inf ...
rtp_llm_server_request_duration_seconds_sum <timestamp> 4.2 ...
rtp_llm_server_request_duration_seconds_count <timestamp> 5 ...
```

Buckets are cumulative both across boundaries and over the process lifetime;
`_sum` is in seconds and `_count` counts observations. Snapshots do not reset
or consume histogram state. With no new requests, later reports repeat the
same values. A downstream collector must retain the latest snapshot per series,
not sum consecutive reporting cycles. The protocol still carries scalar events;
family reconstruction and any Prometheus/OTel export belong in that collector.
This PR does not implement or claim that downstream conversion.

Every histogram component includes `process_instance_id`, generated once per
process lifetime, to distinguish multiple frontends and process restarts.
Collectors must retain this identity while calculating counter increases/rates,
then aggregate across processes. A restart starts a new series at zero; it must
not overwrite another frontend's series. Histograms have no shared filesystem
state and are initialized in each frontend process. In-flight observations since
the last successful report can be lost on an abrupt process exit, as with the
existing reporting transport; this is not durable event delivery.

KMonitor's C++ `SUMMARY` uses DDSketch, but the Python frontend uses a separate
client that does not expose it. This change extends that Python client's existing
registration, reporting and transport path with explicit buckets. It does not
rebin DDSketch or modify the C++ summary protocol. Bucket boundaries follow the
recommendations linked above, preserving exact counts at those boundaries.

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
  Cumulative counts belong to each frontend process lifetime.
