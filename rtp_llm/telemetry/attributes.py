"""Centralized OTel trace attribute schema for RTP-LLM.

Every span attribute key used by the Python frontend and its C++ mirror
(rtp_llm/cpp/telemetry/TraceAttributes.h) is registered here, so names and
units stay aligned and no string literals scatter across the frontend. Keys are
grouped in three layers:
  - OTel official candidate: standard keys every backend already understands
  - ARMS-Unitrace extension:  NOT current OTel semconv, kept for Unitrace
  - rtp_llm.* internal:       RTP-LLM own fields (mapped from AuxInfo)

The frontend HTTP SERVER span carries the request-level business summary;
per-role C++ spans carry topology, latency and their own per-hop token usage.
"""

# --- OTel official candidate ---
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
# Not in current OTel GenAI semconv (input/output only), but the platform LLM
# view aggregates "Total tokens" from it; kept for platform compatibility.
GEN_AI_USAGE_TOTAL_TOKENS = "gen_ai.usage.total_tokens"
# Legacy semconv aliases: some platform views only read these older names.
GEN_AI_USAGE_PROMPT_TOKENS = "gen_ai.usage.prompt_tokens"
GEN_AI_USAGE_COMPLETION_TOKENS = "gen_ai.usage.completion_tokens"
GEN_AI_RESPONSE_FINISH_REASONS = "gen_ai.response.finish_reasons"
# GenAI semconv classification keys: platforms commonly use these to decide
# whether a span is an LLM call at all (belt-and-braces alongside the platform
# classification trio below, since the LLM-view matching rule is not publicly
# documented).
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"  # value: "chat"
GEN_AI_SYSTEM = "gen_ai.system"  # value: "rtp_llm"

# --- OTel stable server endpoint attributes (CLIENT spans) ---
# The address/port identify the logical peer selected by RTP-LLM, allowing the
# backend to render the P->D topology edge instead of UNSET_ENDPOINT.
SERVER_ADDRESS = "server.address"
SERVER_PORT = "server.port"

# --- ARMS-Unitrace extension: NOT official OTel semconv ---
# Logical-request TTFT observed at the HTTP/Dash SERVER boundary: SERVER span
# start to the first response carrying caller-visible output tokens. Unit: ms.
# Do not rename this span attribute to gen_ai.server.time_to_first_token: the
# latter is an OTel Histogram metric in seconds, not a per-trace attribute.
GEN_AI_TIME_TO_FIRST_TOKEN = "gen_ai.response.time_to_first_token"

# Logical-request delivery interval after the first caller-visible token. It is
# written only on streaming HTTP/Dash SERVER spans. Unit: milliseconds.
RTP_LLM_FRONTEND_TIME_PER_OUTPUT_TOKEN_MS = "rtp_llm.frontend.time_per_output_token_ms"

# Per-engine-stream latency copied from AuxInfo onto generate_stream_call CLIENT
# spans. first_token_cost_time/cost_time are already milliseconds on the Python
# side (model_rpc_client divides the protobuf microseconds by 1000).
RTP_LLM_ENGINE_TIME_TO_FIRST_TOKEN_MS = "rtp_llm.engine.time_to_first_token_ms"
RTP_LLM_ENGINE_TIME_PER_OUTPUT_TOKEN_MS = "rtp_llm.engine.time_per_output_token_ms"

# Unit: NANOSECONDS. The platform expects nanoseconds for these two keys even
# though TTFT above stays in milliseconds; verified against the LLM view.
GEN_AI_LATENCY_TIME_IN_MODEL_PREFILL = "gen_ai.latency.time_in_model_prefill"
GEN_AI_LATENCY_TIME_IN_MODEL_DECODE = "gen_ai.latency.time_in_model_decode"

# --- LLM-view classification (platform compatibility) ---
# The platform LLM view only recognizes a span as an LLM call when it carries
# this trio, so the root SERVER span always sets all three together.
# NOTE: lingji_flag / acs.arms.* are platform-specific fields (not OTel
# semconv); semantics unconfirmed with the platform — revisit if the LLM view
# still ignores the span after this trio lands.
GEN_AI_SPAN_KIND = "gen_ai.span.kind"  # value: "LLM"
LINGJI_FLAG = "lingji_flag"  # value: True
ACS_ARMS_TENANT_SPAN_POLICY = "acs.arms.tenant.span.policy"  # value: "mask"

# --- rtp_llm.* internal ---
# Caller-supplied correlation id on transparent ingress/forwarding spans. Keep
# this distinct from the platform-indexed request_id key, which represents the
# generated engine request and creates a model-topology node in Unitrace.
RTP_LLM_EXTERNAL_REQUEST_ID = "rtp_llm.external_request_id"
RTP_LLM_PD_SEP = "rtp_llm.pd_sep"
RTP_LLM_CACHE_TOTAL_REUSE_LEN = "rtp_llm.cache.total_reuse_len"
RTP_LLM_CACHE_LOCAL_REUSE_LEN = "rtp_llm.cache.local_reuse_len"
RTP_LLM_CACHE_REMOTE_REUSE_LEN = "rtp_llm.cache.remote_reuse_len"
# PD node selection outcome on the master_route span (low-cardinality enum:
# Final route-source chain. A request can provide a partial role list and then
# be completed by domain routing, hence the explicit combined values.
# "master" | "domain_fallback" | "master+domain_fallback" |
# "request" | "request+domain_fallback" | "none".
RTP_LLM_ROUTE_SOURCE = "rtp_llm.route.source"
RTP_LLM_ROUTE_SOURCE_VALUES = frozenset(
    {
        "master",
        "domain_fallback",
        "master+domain_fallback",
        "request",
        "request+domain_fallback",
        "none",
    }
)

# Proactive queue rejection diagnostics on the master_route span (set only on
# the TRAFFIC_LIMIT_ERROR path): cached FlexLB queue length observed at
# rejection time and the configured threshold it exceeded.
RTP_LLM_ROUTE_QUEUE_LENGTH = "rtp_llm.route.queue_length"
RTP_LLM_ROUTE_QUEUE_REJECT_THRESHOLD = "rtp_llm.route.queue_reject_threshold"

# --- C++-side span keys (single-source registry; C++ mirror lives in
# rtp_llm/cpp/telemetry/TraceAttributes.h and must stay in sync with this
# section). host.ip is a resource attribute set in TelemetryRuntime.cc and is
# intentionally excluded from the span-key registry. ---
# Bailian Unitrace indexes spans by the unprefixed string request_id; the
# rtp_llm.* twin retains the numeric engine id for internal correlation.
REQUEST_ID = "request_id"
RTP_LLM_REQUEST_ID = "rtp_llm.request_id"
# Numeric gRPC status companion to error.type (GrpcStatusSpanGuard).
RTP_LLM_GRPC_STATUS_CODE = "rtp_llm.grpc_status_code"
# Stable application error identity on the operation that directly observed
# the failure. Values come from the bounded C++ ErrorCode enum; raw messages
# must not be written to either attribute.
RTP_LLM_ERROR_CODE = "rtp_llm.error.code"
RTP_LLM_ERROR_REASON = "rtp_llm.error.reason"
# Zero-based number of retries already performed before this physical P->D
# RemoteGenerate RPC. The initial attempt is 0, the first retry is 1.
RTP_LLM_RETRY_ATTEMPT = "rtp_llm.retry_attempt"
# P->D CLIENT span stage timings in microseconds.
RTP_LLM_ALLOCATE_RT_US = "rtp_llm.allocate_rt_us"
RTP_LLM_POLL_LOCAL_OUTPUT_RT_US = "rtp_llm.poll_local_output_rt_us"
RTP_LLM_POLL_REMOTE_OUTPUT_RT_US = "rtp_llm.poll_remote_output_rt_us"
# True only when a failed request cuts off a phase before its natural end.
RTP_LLM_PHASE_TRUNCATED = "rtp_llm.phase.truncated"

# --- HTTP semconv (root SERVER span only). Platform views read the two
# semconv generations with inconsistent priority, and the HTTP-error counter
# only reads the legacy key, so both are always written together. ---
HTTP_RESPONSE_STATUS_CODE = "http.response.status_code"
HTTP_STATUS_CODE = "http.status_code"  # legacy alias
HTTP_REQUEST_METHOD = "http.request.method"
HTTP_METHOD = "http.method"  # legacy alias

# --- error / status (low-cardinality enum values only, no raw error text on
# spans) ---
ERROR_TYPE = "error.type"

# --- RPC semconv: transport marker for real gRPC boundary spans. Only the
# C++ span factories (RpcTraceHelper.h) set it now: on the Python frontend
# CLIENT span it made the platform re-classify the span as an RPC client call,
# which broke the top-bar Total tokens aggregation (measured regression), so
# start_client_span deliberately omits it. Never on INTERNAL spans. ---
RPC_SYSTEM = "rpc.system"
RPC_SYSTEM_GRPC = "grpc"
RPC_METHOD = "rpc.method"
RPC_RESPONSE_STATUS_CODE = "rpc.response.status_code"

# --- span event names (platform UIs surface them under an Events(N) tab) ---
# first_response_chunk: fired when the first frontend response object for a
# streaming request becomes available, before SSE serialization and sending.
# The object may be a role-only chunk, so this is neither the engine first-token
# timestamp nor client-observed TTFT. Non-streaming requests do not emit it.
EVENT_FIRST_RESPONSE_CHUNK = "first_response_chunk"
