import logging
from enum import Enum
from typing import Any, Dict, Iterable, Optional, Union

from .emb_interval_metrics import (
    EmbCacheTokensMetric,
    EmbInflightMetric,
    EmbLatencyMetric,
    EmbOutcomeMetric,
    EmbSampleMetric,
)

# Auto-TPM QoS priority header conveyed by the DashScope gateway; the same
# value the engine forwards to FlexLB as ``Schedule.priority``.
QOS_PRIORITY_HEADER = "x-dashscope-inner-qos-level"


def qos_priority_tag(qos_level: Any) -> str:
    """Normalize an ``x-dashscope-inner-qos-level`` value into the kmonitor
    ``priority`` tag value.

    Returns the raw 1-100 integer as a string, or ``"0"`` when the request
    carries no (or an unparseable) priority — same convention as the FlexLB
    ``auto_tpm.*`` metric family ("0" = legacy request without a budget).
    """
    if qos_level is None:
        return "0"
    try:
        return str(int(str(qos_level).strip()))
    except (TypeError, ValueError):
        return "0"


class AccMetrics(Enum):
    CANCEL_QPS_METRIC = "py_rtp_cancal_qps_metric"
    SUCCESS_QPS_METRIC = "py_rtp_success_qps_metric"
    QPS_METRIC = "py_rtp_framework_qps"
    ERROR_QPS_METRIC = "py_rtp_framework_error_qps"
    CONFLICT_QPS_METRIC = "py_rtp_framework_concurrency_exception_qps"
    ITER_QPS_METRIC = "py_rtp_response_iterate_qps"
    UPDATE_QPS_METRIC = "py_rtp_update_qps_metric"
    ERROR_UPDATE_QPS_METRIC = "py_rtp_error_update_target_qps"

    # worker_status
    WORKER_STATUS_QPS_METRIC = "py_rtp_worker_status_qps"
    CACHE_STATUS_QPS_METRIC = "py_rtp_cache_status_qps"
    # route:
    ROUTE_QPS_METRIC = "py_rtp_route_qps"
    MASTER_ROUTE_QPS_METRIC = "py_rtp_master_route_qps"
    DOMAIN_ROUTE_QPS_METRIC = "py_rtp_domain_route_qps"
    MASTER_ROUTE_ERROR_QPS_METRIC = "py_rtp_master_route_error_qps"
    MASTER_QUEUE_REJECT_QPS_METRIC = "py_rtp_master_queue_reject_qps"
    RECENT_CACHE_KEY_HIT_COUNT_METRIC = "py_rtp_recent_cache_key_hit_count"
    RECENT_CACHE_KEY_TOTAL_COUNT_METRIC = "py_rtp_recent_cache_key_total_count"
    RECENT_CACHE_KEY_REQUEST_COUNT_METRIC = "py_rtp_recent_cache_key_request_count"
    RECENT_CACHE_KEY_EMPTY_REQUEST_COUNT_METRIC = (
        "py_rtp_recent_cache_key_empty_request_count"
    )

    # igraph
    IGRAPH_QPS_METRIC = "py_rtp_igraph_qps"
    IGRAPH_ERROR_QPS_METRIC = "py_rtp_igraph_error_qps"
    IGRAPH_EMPTY_QPS_METRIC = "py_rtp_igraph_empty_qps"

    # Logical frontend requests, including warm requests that skip either ViT phase.
    VIT_MULTIMODAL_REQUEST_QPS_METRIC = "py_rtp_vit_multimodal_request_qps"
    VIT_PHASE_QPS_METRIC = "py_rtp_vit_phase_qps"
    VIT_QPS_METRIC = "py_rtp_vit_qps"
    VIT_ERROR_QPS_METRIC = "py_rtp_vit_error_qps"
    # Count rejected async submissions, not the number of media items in them.
    VIT_QUEUE_REJECT_QPS_METRIC = "py_rtp_vit_queue_reject_qps"
    VIT_SUCCESS_QPS_METRIC = "py_rtp_vit_success_qps"
    # Accumulated embedding rows; AccMetric normalizes by the flush interval,
    # so the reported value is visual tokens/s.
    VIT_EMBEDDING_TPS_METRIC = "py_rtp_vit_embedding_tps"
    VIT_PACKED_FORWARD_WORK_QPS_METRIC = "py_rtp_vit_packed_forward_work_qps"
    VIT_BATCH_SEAL_QPS_METRIC = "py_rtp_vit_batch_seal_qps"
    VIT_URL_SOURCE_QPS_METRIC = "py_rtp_vit_url_source_qps"
    VIT_HTTP_CONNECT_RETRY_QPS_METRIC = "py_rtp_vit_http_connect_retry_qps"
    VIT_HTTP_SOURCE_RETRY_QPS_METRIC = "py_rtp_vit_http_source_retry_qps"
    # Incremented when a submission is rejected because the scheduler's waiting
    # queue is full (overload / stalled forward backpressure).
    VIT_EMBEDDING_OVERLOAD_QPS_METRIC = "py_rtp_vit_embedding_overload_qps"
    VIT_PROCESS_POOL_RESTART_QPS_METRIC = "py_rtp_vit_process_pool_restart_qps"
    VIT_RPC_CLIENT_ERROR_QPS_METRIC = "rtp_llm_vit_rpc_client_error_qps"
    VIT_RPC_SERVER_ERROR_QPS_METRIC = "rtp_llm_vit_rpc_server_error_qps"
    VIT_RPC_PROXY_ERROR_QPS_METRIC = "rtp_llm_vit_rpc_proxy_error_qps"

    GREENNET_SUCCESS_QPS_METRIC = "py_rtp_greennet_success_qps"
    GREENNET_ERROR_QPS_METRIC = "py_rtp_greennet_error_qps"

    # dash_sc DSV4 phase-2: incremented once per request that enters phase-2
    # via the terminate-token-id (token 1) abort path in the think phase.
    # One increment per request — guarded by ``phase2_triggered`` so the
    # rate matches "requests with a think-abort", not "abort tokens seen".
    DASH_SC_DSV4_PHASE2_QPS_METRIC = "py_rtp_dash_sc_dsv4_phase2_qps"

    # Repetition metrics are reported once per request by the frontend gRPC
    # access-log interceptor; tags intentionally avoid request_id.
    TOOL_CALL_LOOP_QPS_METRIC = "py_rtp_tool_call_loop_qps"
    OUTPUT_REPETITION_QPS_METRIC = "py_rtp_output_repetition_qps"
    SAME_TOKEN_RUN_QPS_METRIC = "py_rtp_same_token_run_qps"

    VIT_EMBEDDING_CACHE_HIT_QPS_METRIC = "py_rtp_vit_embedding_cache_hit_qps"
    VIT_EMBEDDING_CACHE_MISS_QPS_METRIC = "py_rtp_vit_embedding_cache_miss_qps"
    VIT_PREFILL_EMBEDDING_CACHE_MISS_QPS_METRIC = (
        "py_rtp_vit_prefill_embedding_cache_miss_qps"
    )
    VIT_EMBEDDING_CACHE_INFLIGHT_QPS_METRIC = "py_rtp_vit_embedding_cache_inflight_qps"
    VIT_EMBEDDING_CACHE_EVICTION_QPS_METRIC = "py_rtp_vit_embedding_cache_eviction_qps"

    # Multimodal download proxy: download attempts routed through the region
    # egress proxy, and connect-failure fallbacks to direct. Both carry the
    # request region tag.
    MM_DOWNLOAD_PROXY_QPS_METRIC = "py_rtp_mm_download_proxy_qps"
    MM_DOWNLOAD_PROXY_FALLBACK_QPS_METRIC = "py_rtp_mm_download_proxy_fallback_qps"

    # DashScope TPM recheck after the early first packet.
    DASH_SC_TPM_RECHECK_QPS_METRIC = "py_rtp_dash_sc_tpm_recheck_qps"
    DASH_SC_TPM_RECHECK_LIMIT_QPS_METRIC = "py_rtp_dash_sc_tpm_recheck_limit_qps"
    DASH_SC_TPM_RECHECK_ERROR_QPS_METRIC = "py_rtp_dash_sc_tpm_recheck_error_qps"

    # Remote tier only. Read status: hit/miss/admission_skip/timeout/error.
    # Misses and admission skips are not failures. Error QPS includes timeouts.
    VIT_REMOTE_EMB_READ_QPS_METRIC = "py_rtp_vit_remote_embedding_cache_read_qps"
    VIT_REMOTE_EMB_READ_ERROR_QPS_METRIC = (
        "py_rtp_vit_remote_embedding_cache_read_error_qps"
    )
    VIT_REMOTE_EMB_WRITE_QPS_METRIC = "py_rtp_vit_remote_embedding_cache_write_qps"
    VIT_REMOTE_EMB_WRITE_ERROR_QPS_METRIC = (
        "py_rtp_vit_remote_embedding_cache_write_error_qps"
    )
    # Per operation/stage: total = success + error + skipped. Miss is success.
    VIT_REMOTE_EMB_OPERATION_QPS_METRIC = (
        "py_rtp_vit_remote_embedding_cache_operation_qps"
    )
    VIT_REMOTE_EMB_SKIP_QPS_METRIC = "py_rtp_vit_remote_embedding_cache_skip_qps"
    VIT_REMOTE_EMB_GDR_ACQUIRE_FAILURE_QPS_METRIC = (
        "py_rtp_vit_remote_embedding_cache_gdr_acquire_failure_qps"
    )
    VIT_REMOTE_EMB_WRITE_IN_PROGRESS_QPS_METRIC = (
        "py_rtp_vit_remote_embedding_cache_write_in_progress_qps"
    )

    VIT_REMOTE_EMB_BUFFER_HOLD_QPS_METRIC = (
        "py_rtp_vit_remote_embedding_cache_buffer_hold_qps"
    )
    VIT_REMOTE_EMB_BUFFER_GC_QPS_METRIC = (
        "py_rtp_vit_remote_embedding_cache_buffer_gc_qps"
    )
    VIT_REMOTE_EMB_BUFFER_GC_SUCCESS_QPS_METRIC = (
        "py_rtp_vit_remote_embedding_cache_buffer_gc_success_qps"
    )
    VIT_REMOTE_EMB_BUFFER_GC_ERROR_QPS_METRIC = (
        "py_rtp_vit_remote_embedding_cache_buffer_gc_error_qps"
    )

    # Creation is distinct from per-operation acquisition of a shared client.
    VIT_REMOTE_EMB_CLIENT_CREATE_QPS_METRIC = (
        "py_rtp_vit_remote_embedding_cache_client_create_qps"
    )
    VIT_REMOTE_EMB_CLIENT_CREATE_ERROR_QPS_METRIC = (
        "py_rtp_vit_remote_embedding_cache_client_create_error_qps"
    )
    VIT_REMOTE_EMB_METADATA_QPS_METRIC = (
        "py_rtp_vit_remote_embedding_cache_metadata_qps"
    )
    # One inspection-approved ViT result, not one LLM response or remote read.
    VIT_REMOTE_EMB_RESULT_ACCEPTED_QPS_METRIC = (
        "py_rtp_vit_remote_embedding_cache_result_accepted_qps"
    )

    # Bounded operation/stage/status tags; no object IDs or exception text.
    VIT_REMOTE_EMB_STAGE_QPS_METRIC = "py_rtp_vit_remote_embedding_cache_stage_qps"
    VIT_REMOTE_EMB_STAGE_ERROR_QPS_METRIC = (
        "py_rtp_vit_remote_embedding_cache_stage_error_qps"
    )
    VIT_REMOTE_EMB_HEADER_OBSERVATION_QPS_METRIC = (
        "py_rtp_vit_remote_embedding_cache_header_observation_qps"
    )
    VIT_REMOTE_EMB_HEADER_D2H_BYTES_PER_SECOND_METRIC = (
        "py_rtp_vit_remote_embedding_cache_header_d2h_bytes_per_second"
    )
    # Accumulated bytes are exported as B/s, not as an average object size.
    VIT_REMOTE_EMB_SDK_OBJECT_BYTES_PER_SECOND_METRIC = (
        "py_rtp_vit_remote_embedding_cache_sdk_object_bytes_per_second"
    )
    VIT_REMOTE_EMB_PAYLOAD_BYTES_PER_SECOND_METRIC = (
        "py_rtp_vit_remote_embedding_cache_payload_bytes_per_second"
    )


class GaugeMetrics(Enum):
    VIT_REMOTE_EMB_INFLIGHT_TASKS_METRIC = (
        "py_rtp_vit_remote_embedding_cache_inflight_tasks"
    )
    VIT_REMOTE_EMB_INFLIGHT_BYTES_METRIC = (
        "py_rtp_vit_remote_embedding_cache_inflight_bytes"
    )
    VIT_REMOTE_EMB_GDR_ARENA_BYTES_METRIC = (
        "py_rtp_vit_remote_embedding_cache_gdr_arena_bytes"
    )
    VIT_REMOTE_EMB_BUFFER_HOLD_BYTES_METRIC = (
        "py_rtp_vit_remote_embedding_cache_buffer_hold_bytes"
    )
    VIT_REMOTE_EMB_BUFFER_RETAINED_BYTES_METRIC = (
        "py_rtp_vit_remote_embedding_cache_buffer_retained_bytes"
    )
    VIT_REMOTE_EMB_BUFFER_HOLD_AGE_METRIC = (
        "py_rtp_vit_remote_embedding_cache_buffer_hold_age_seconds"
    )
    VIT_REMOTE_EMB_BUFFER_HOLD_COUNT_METRIC = (
        "py_rtp_vit_remote_embedding_cache_buffer_hold_count"
    )
    RESPONSE_FIRST_TOKEN_RT_METRIC = "py_rtp_response_first_token_rt"
    RESPONSE_ITER_RT_METRIC = "py_rtp_response_iterate_rt"
    RESPONSE_ITERATE_COUNT = "py_rtp_response_iterate_count"
    LANTENCY_METRIC = "py_rtp_framework_rt"

    FT_ITERATE_COUNT_METRIC = "ft_iterate_count"
    INPUT_TOKEN_SIZE_METRIC = "ft_input_token_length"
    OUTPUT_TOKEN_SIZE_METRIC = "ft_output_token_length"
    PRE_PIPELINE_RT_METRIC = "ft_pre_pipeline_rt"
    POST_PIPELINE_RT_METRIC = "ft_post_pipeline_rt"
    NUM_BEAMS_METRIC = "ft_num_beams"

    UPDATE_LANTENCY_METRIC = "py_rtp_update_framework_rt"

    # worker_status
    WORKER_STATUS_QPS_LANTENCY_METRIC = "py_rtp_worker_status_rt"
    CACHE_STATUS_QPS_LATENCY_METRIC = "py_rtp_cache_status_rt"

    # route:
    ROUTE_RT_METRIC = "py_rtp_route_rt"
    MASTER_ROUTE_RT_METRIC = "py_rtp_master_route_rt"
    DOMAIN_ROUTE_RT_METRIC = "py_rtp_domain_route_rt"
    MASTER_QUEUE_LENGTH_METRIC = "py_rtp_master_queue_length"
    MASTER_HOST_METRIC = "py_rtp_master_host"
    RECENT_CACHE_KEY_HIT_RATIO_METRIC = "py_rtp_recent_cache_key_hit_ratio"

    # igraph
    IGRAPH_RT_METRIC = "py_rtp_igraph_rt"
    PARSE_IGRAPH_RESPONSE_RT_METRIC = "py_rtp_parse_igraph_response_rt"

    # vit preprocess
    VIT_PREPROCESS_RT_METRIC = "py_rtp_vit_preprocess_rt"
    # Milliseconds from a blocking engine request's entry to return/error,
    # including CPU/GPU work, queue/cache and GreenNet waits, but not RPC output
    # serialization. The legacy RT is for integrated ViT. Split ViT times only
    # executed phases and counts the same attempts by source and success/error.
    VIT_REQUEST_RT_METRIC = "py_rtp_vit_request_rt"
    VIT_HASH_FILL_RT_METRIC = "py_rtp_vit_hash_fill_rt"
    VIT_REMOTE_MULTIMODAL_EMBEDDING_RT_METRIC = (
        "py_rtp_vit_remote_multimodal_embedding_rt"
    )
    # Full observed cache latency in milliseconds. A read hit includes metadata,
    # queueing, SDK read, validation, and the direct GDR restore fence. A successful
    # write includes planning, admission, snapshots, queueing, packing, and SDK save.
    VIT_REMOTE_EMB_READ_RT_METRIC = "py_rtp_vit_remote_embedding_cache_read_rt"
    VIT_REMOTE_EMB_WRITE_RT_METRIC = "py_rtp_vit_remote_embedding_cache_write_rt"

    VIT_REMOTE_EMB_STAGE_RT_METRIC = "py_rtp_vit_remote_embedding_cache_stage_rt_ms"
    VIT_REMOTE_EMB_GDR_LOCK_HOLD_RT_METRIC = (
        "py_rtp_vit_remote_embedding_cache_gdr_lock_hold_ms"
    )
    # Enqueue to worker start; canceled-before-start work has no observation.
    VIT_REMOTE_EMB_QUEUE_WAIT_METRIC = "py_rtp_vit_remote_embedding_cache_queue_wait_ms"
    # Logical object bytes for completed SDK calls and bytes / SDK wall time.
    # An existing-key save may succeed without IO; reads may use local replicas.
    # These are API volume, not transferred/network bytes. PACE reports actual IO.
    VIT_REMOTE_EMB_SDK_OBJECT_BYTES_METRIC = (
        "py_rtp_vit_remote_embedding_cache_sdk_object_bytes"
    )
    VIT_REMOTE_EMB_SDK_EFFECTIVE_SPEED_METRIC = (
        "py_rtp_vit_remote_embedding_cache_sdk_effective_mib_per_second"
    )
    # Only caller-accepted reads / completed writes. A late SDK read after a
    # caller timeout contributes to transfer metrics, never to payload metrics.
    VIT_REMOTE_EMB_PAYLOAD_BYTES_METRIC = (
        "py_rtp_vit_remote_embedding_cache_payload_bytes"
    )
    VIT_REMOTE_EMB_EFFECTIVE_SPEED_METRIC = (
        "py_rtp_vit_remote_embedding_cache_effective_mib_per_second"
    )
    # Milliseconds per non-empty inspection group, including OSS/HTTP work,
    # excluding input preparation, source handoff wait and GPU embedding.
    GREENNET_RT_METRIC = "py_rtp_greennet_rt"
    # Per-request embedding latency = wait + forward, sampled once per request in
    # submit_and_wait. Preserves the historical meaning (pre-scheduler this timed
    # the forward under the embedding lock, i.e. lock-wait + forward per request)
    # so existing dashboards/alerts are unchanged.
    VIT_EMBEDDING_RT_METRIC = "py_rtp_vit_embedding_rt"
    VIT_RPC_SERVER_HANDLER_RT_US_METRIC = "rtp_llm_vit_rpc_server_handler_rt_us"
    VIT_RPC_SERVER_LIFECYCLE_RT_US_METRIC = "rtp_llm_vit_rpc_server_lifecycle_rt_us"
    VIT_RPC_PROXY_LIFECYCLE_RT_US_METRIC = "rtp_llm_vit_rpc_proxy_lifecycle_rt_us"
    VIT_RPC_PROXY_TO_WORKER_RT_US_METRIC = "rtp_llm_vit_rpc_proxy_to_worker_rt_us"
    VIT_RPC_PROXY_HEALTHY_WORKER_COUNT_METRIC = (
        "rtp_llm_vit_rpc_proxy_healthy_worker_count"
    )
    VIT_RPC_PROXY_TOTAL_WORKER_COUNT_METRIC = "rtp_llm_vit_rpc_proxy_total_worker_count"
    VIT_RPC_REQUEST_BYTES_METRIC = "rtp_llm_vit_rpc_request_bytes"
    VIT_RPC_RESPONSE_BYTES_METRIC = "rtp_llm_vit_rpc_response_bytes"
    VIT_RESPONSE_EMBEDDING_BYTES_METRIC = "rtp_llm_vit_response_embedding_bytes"
    VIT_RESPONSE_DEEPSTACK_BYTES_METRIC = "rtp_llm_vit_response_deepstack_bytes"
    VIT_RESPONSE_POS_BYTES_METRIC = "rtp_llm_vit_response_pos_bytes"
    VIT_OUTPUT_TOKEN_COUNT_METRIC = "rtp_llm_vit_output_token_count"
    VIT_INPUT_IMAGE_COUNT_METRIC = "rtp_llm_vit_input_image_count"
    VIT_IMAGE_FETCH_RT_US_METRIC = "rtp_llm_vit_image_fetch_rt_us"
    VIT_HTTP_PHASE_RT_US_METRIC = "rtp_llm_vit_http_phase_rt_us"
    VIT_HTTP_BODY_BYTES_METRIC = "rtp_llm_vit_http_body_bytes"
    VIT_PREPROCESS_RESULT_BYTES_METRIC = "rtp_llm_vit_preprocess_result_bytes"
    VIT_RGB_UPLOAD_PHASE_US_METRIC = "rtp_llm_vit_rgb_upload_phase_us"
    VIT_RGB_UPLOAD_BYTES_METRIC = "rtp_llm_vit_rgb_upload_bytes"
    VIT_IMAGE_DECODE_RT_US_METRIC = "rtp_llm_vit_image_decode_rt_us"
    VIT_IMAGE_RESIZE_RT_US_METRIC = "rtp_llm_vit_image_resize_rt_us"
    VIT_IMAGE_PROCESSOR_RT_US_METRIC = "rtp_llm_vit_image_processor_rt_us"
    VIT_RESIZED_PIXEL_COUNT_METRIC = "rtp_llm_vit_resized_pixel_count"
    # Forward-only latency of one GPU embedding forward (no wait), sampled once
    # per merged batch in _run_embedding. A NEW name so it doesn't redefine the
    # historical per-request VIT_EMBEDDING_RT.
    VIT_EMBEDDING_FORWARD_RT_METRIC = "py_rtp_vit_embedding_forward_rt"
    # Number of requests merged into one GPU forward by the MMScheduler. 1 on the
    # serial path (gpu_max_batch_size == 1); > 1 means cross-request batching kicked in.
    VIT_EMBEDDING_BATCH_SIZE_METRIC = "py_rtp_vit_embedding_batch_size"

    # DashScope TPM recheck callback latency (ms).
    DASH_SC_TPM_RECHECK_RT_METRIC = "py_rtp_dash_sc_tpm_recheck_rt"

    TOOL_CALL_LOOP_REPEAT_COUNT_METRIC = "py_rtp_tool_call_loop_repeat_count"
    TOOL_CALL_LOOP_CURRENT_SPAN_TOKENS_METRIC = (
        "py_rtp_tool_call_loop_current_span_tokens"
    )
    TOOL_CALL_LOOP_CHECK_RT_METRIC = "py_rtp_tool_call_loop_check_rt"
    OUTPUT_REPETITION_CHECK_RT_METRIC = "py_rtp_output_repetition_check_rt"
    OUTPUT_REPETITION_PERIOD_METRIC = "py_rtp_output_repetition_period"
    OUTPUT_REPETITION_REPEAT_COUNT_METRIC = "py_rtp_output_repetition_repeat_count"
    OUTPUT_REPETITION_DUPLICATE_TOKENS_METRIC = (
        "py_rtp_output_repetition_duplicate_tokens"
    )

    VIT_DOWNLOAD_RT_METRIC = "py_rtp_vit_download_rt"
    VIT_PREPROCESS_OTHER_RT_METRIC = "py_rtp_vit_preprocess_other_rt"
    VIT_PREPROCESS_QUEUE_SIZE_METRIC = "py_rtp_vit_preprocess_queue_size"
    VIT_PREPROCESS_DISPATCH_RT_METRIC = "py_rtp_vit_preprocess_dispatch_rt"
    VIT_PREPROCESS_RETURN_RT_METRIC = "py_rtp_vit_preprocess_return_rt"
    VIT_EMBEDDING_LENGTH_METRIC = "py_rtp_vit_embedding_length"
    VIT_IMAGE_COUNT_METRIC = "py_rtp_vit_image_count"
    VIT_EMBEDDING_QUEUE_SIZE_METRIC = "py_rtp_vit_embedding_queue_size"
    VIT_EMBEDDING_QUEUE_WAIT_RT_METRIC = "py_rtp_vit_embedding_queue_wait_rt"
    VIT_EMBEDDING_CACHE_HIT_RT_METRIC = "py_rtp_vit_embedding_cache_hit_rt_ms"
    VIT_EMBEDDING_CACHE_TOKENS_METRIC = "py_rtp_vit_embedding_cache_tokens"
    VIT_EMBEDDING_CACHE_BYTES_METRIC = "py_rtp_vit_embedding_cache_bytes"
    VIT_EMBEDDING_CACHE_RESIDENT_BYTES_METRIC = (
        "py_rtp_vit_embedding_cache_resident_bytes"
    )
    VIT_EMBEDDING_CACHE_RESIDENT_ENTRIES_METRIC = (
        "py_rtp_vit_embedding_cache_resident_entries"
    )
    # Configured GPU + CPU capacity / successful new admission bytes per second.
    # All shards share one activity-driven window of at least 60s. Idle periods
    # emit no samples. This is an estimated turnover time, not a per-entry TTL.
    VIT_EMBEDDING_CACHE_ESTIMATED_LIFETIME_SECONDS_METRIC = (
        "py_rtp_vit_embedding_cache_estimated_lifetime_seconds"
    )


class MetricReporter(object):
    def __init__(self, kmonitor: Any):
        self._kmon = kmonitor
        self._matic_map: Dict[str, Any] = {}
        self._inited = False
        self.embedding_cache_tokens = EmbCacheTokensMetric(
            GaugeMetrics.VIT_EMBEDDING_CACHE_TOKENS_METRIC.value, {}
        )

    def report(
        self,
        metric: Union[AccMetrics, GaugeMetrics],
        value: float = 1,
        tags: Dict[str, Any] = {},
    ):
        kmon_metric = self._matic_map.get(metric.value, None)
        if kmon_metric is None:
            logging.warning(f"no metric named {metric.name}")
            return
        kmon_metric.report(value, tags)

    def flush(self) -> None:
        self._kmon.flush()

    def init(
        self,
        *,
        acc_metrics: Optional[Iterable[AccMetrics]] = None,
        gauge_metrics: Optional[Iterable[GaugeMetrics]] = None,
        tags: Optional[Dict[str, str]] = None,
    ):
        """Register once; workers can restrict metrics and isolate their series."""
        if not self._inited:
            self._inited = True
            for metric in AccMetrics if acc_metrics is None else acc_metrics:
                if metric is AccMetrics.VIT_REMOTE_EMB_OPERATION_QPS_METRIC:
                    backend = self._kmon.register_custom_metric(
                        EmbOutcomeMetric(metric.value, tags or {})
                    )
                elif metric in (
                    AccMetrics.VIT_REMOTE_EMB_STAGE_QPS_METRIC,
                    AccMetrics.VIT_REMOTE_EMB_STAGE_ERROR_QPS_METRIC,
                ):
                    # Emitted from the same stage samples as latency.
                    continue
                else:
                    backend = self._kmon.register_acc_metric(metric.value, tags or {})
                self._matic_map[metric.value] = backend

            for metric in GaugeMetrics if gauge_metrics is None else gauge_metrics:
                if metric is GaugeMetrics.VIT_EMBEDDING_CACHE_TOKENS_METRIC:
                    self.embedding_cache_tokens.tags = dict(tags or {})
                    backend = self._kmon.register_custom_metric(
                        self.embedding_cache_tokens
                    )
                elif metric in (
                    GaugeMetrics.VIT_REMOTE_EMB_INFLIGHT_TASKS_METRIC,
                    GaugeMetrics.VIT_REMOTE_EMB_INFLIGHT_BYTES_METRIC,
                ):
                    backend = self._kmon.register_custom_metric(
                        EmbInflightMetric(metric.value, tags or {})
                    )
                elif metric in (
                    GaugeMetrics.VIT_REMOTE_EMB_STAGE_RT_METRIC,
                    GaugeMetrics.VIT_EMBEDDING_CACHE_HIT_RT_METRIC,
                    GaugeMetrics.VIT_REMOTE_EMB_GDR_LOCK_HOLD_RT_METRIC,
                    GaugeMetrics.VIT_REMOTE_EMB_QUEUE_WAIT_METRIC,
                    GaugeMetrics.VIT_REMOTE_EMB_READ_RT_METRIC,
                    GaugeMetrics.VIT_REMOTE_EMB_WRITE_RT_METRIC,
                ):
                    backend = self._kmon.register_custom_metric(
                        EmbLatencyMetric(metric.value, tags or {})
                    )
                elif metric in (
                    GaugeMetrics.VIT_REMOTE_EMB_GDR_ARENA_BYTES_METRIC,
                    GaugeMetrics.VIT_REMOTE_EMB_SDK_OBJECT_BYTES_METRIC,
                    GaugeMetrics.VIT_REMOTE_EMB_SDK_EFFECTIVE_SPEED_METRIC,
                    GaugeMetrics.VIT_REMOTE_EMB_PAYLOAD_BYTES_METRIC,
                    GaugeMetrics.VIT_REMOTE_EMB_EFFECTIVE_SPEED_METRIC,
                ):
                    backend = self._kmon.register_custom_metric(
                        EmbSampleMetric(
                            metric.value,
                            tags or {},
                            arena=(
                                metric
                                is GaugeMetrics.VIT_REMOTE_EMB_GDR_ARENA_BYTES_METRIC
                            ),
                        )
                    )
                else:
                    backend = self._kmon.register_gauge_metric(metric.value, tags or {})
                self._matic_map[metric.value] = backend
