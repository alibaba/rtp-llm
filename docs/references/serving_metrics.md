# KMonitor latency histogram companions

Set `RTP_LLM_METRICS_ENABLED=1` before starting RTP-LLM to record histograms
alongside three existing latency Gauges. `MetricReporter.report()` writes the
original Gauge value in milliseconds, then records the **same sample** in a
separate Histogram after converting milliseconds to seconds. Both receive the
same caller tags and KMonitor default tags. No new timing boundaries, request
state, middleware, or API routes are introduced.

| Existing Gauge (milliseconds) | New Histogram (seconds) |
| --- | --- |
| `py_rtp_framework_rt` | `py_rtp_framework_rt_seconds` |
| `py_rtp_response_first_token_rt` | `py_rtp_response_first_token_rt_seconds` |
| `py_rtp_response_iterate_rt` | `py_rtp_response_iterate_rt_seconds` |

Existing Gauge names, units, call sites, and reporting behavior remain unchanged.
The flag is off by default. All existing callers of these three metrics get the
companion Histogram when enabled, including the HTTP frontend and DashScope gRPC
frontend/forwarder. Other Gauges and accumulated metrics are unaffected.
Nonfinite or negative observations cannot enter a latency Histogram; the original
Gauge still receives them and a Histogram reporting error is logged.

## Timing semantics and community conventions

Histograms in seconds support aggregation and latency distributions, as used by
the public, community-maintained
[OpenTelemetry GenAI model-server semantic conventions](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-metrics.md#generative-ai-model-server-metrics)
and [model-serving-signals](https://github.com/kubernetes-sigs/model-serving-signals/blob/main/model-server-contract.md).
The bucket boundaries use the corresponding duration/first-token/per-token
recommendations. The OTel conventions currently have **Development** stability.

These are distributions of **existing RTP-LLM observations**, not a claim that
the existing measurements satisfy every GenAI metric definition:

- Framework RT retains each caller's existing start/end boundaries. In the HTTP
  generation path it spans the generation call through generator exhaustion;
  the same Gauge is also used by embeddings and gRPC. It is not a newly measured
  HTTP-entry-to-last-response-byte duration. Error/cancellation coverage follows
  the existing call sites, which differ between HTTP and gRPC.
- First-token RT retains the existing first-response boundary and sampling
  behavior. The HTTP generation wrapper measures from iteration start to the
  first yielded response, including non-streaming responses; it does not newly
  filter for a first visible token. The gRPC path starts at its RPC record time.
- Iterate RT is sampled per subsequent response iteration. The HTTP path uses
  its existing `step_output_len` normalization; gRPC reports chunk intervals.
  This is **not** the per-request mean TPOT distribution required by
  `gen_ai.server.time_per_output_token`. A longer request contributes more
  iteration samples, so simply renaming this Histogram would be incorrect.

A downstream adapter must verify timing boundaries, sampling populations, and
protocol/operation distinctions before mapping any of these families to standard
metrics. Adding a Histogram preserves a distribution; it does not change what
its observations mean.

## Existing KMonitor transport

The KMonitor Python client gains
`register_histogram_metric(name, buckets, tags)`. Each finite, nonnegative
observation increments one explicit bucket. The existing report worker emits
cumulative `_bucket` components (with an `le` tag including `+Inf`), `_sum` in
seconds, and `_count` using the numeric `py_kmonitor` Flume event format.

Snapshots do not reset Histogram state. Receivers must retain the latest
snapshot per series, not add successive reporting cycles as new observations.
Each Histogram component also has `process_instance_id`, identifying a frontend
process lifetime. Retain it while calculating counter increases/rates, then
aggregate across processes. A restart starts a new series; observations since
the last successful report can be lost on abrupt exit.

Delivery uses the existing KMonitor configuration: the Python report worker
connects to `HIPPO_SLAVE_IP:4141` when `HIPPO_ROLE_SHORT_NAME` or `HIPPO_ROLE` is
nonempty. Outside that configuration, the existing worker does not deliver
metrics. An operational KMonitor/Flume collector is required. The wire format
still carries scalar components; family reconstruction and any Prometheus/OTel
export belong in that collector and are not implemented by this PR.

No Prometheus dependency, `/metrics` endpoint, shared multiprocess directory,
trace receiver, or access-log reader is introduced. The C++ DDSketch-based
SUMMARY implementation is unchanged; this extends the separate Python client's
existing registration and reporting path with explicit buckets.
