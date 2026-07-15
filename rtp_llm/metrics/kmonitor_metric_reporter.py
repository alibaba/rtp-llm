import logging
from enum import Enum
from typing import Any, Dict, Union


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

    # igraph
    IGRAPH_QPS_METRIC = "py_rtp_igraph_qps"
    IGRAPH_ERROR_QPS_METRIC = "py_rtp_igraph_error_qps"
    IGRAPH_EMPTY_QPS_METRIC = "py_rtp_igraph_empty_qps"

    VIT_QPS_METRIC = "py_rtp_vit_qps"
    VIT_ERROR_QPS_METRIC = "py_rtp_vit_error_qps"
    VIT_SUCCESS_QPS_METRIC = "py_rtp_vit_success_qps"
    VIT_PROCESS_POOL_RESTART_QPS_METRIC = "py_rtp_vit_process_pool_restart_qps"
    VIT_RPC_CLIENT_ERROR_QPS_METRIC = "rtp_llm_vit_rpc_client_error_qps"
    VIT_RPC_SERVER_ERROR_QPS_METRIC = "rtp_llm_vit_rpc_server_error_qps"
    VIT_RPC_PROXY_ERROR_QPS_METRIC = "rtp_llm_vit_rpc_proxy_error_qps"

    # topology KV policy
    TOPOLOGY_KV_COORDINATE_FALLBACK_QPS_METRIC = (
        "py_rtp_topology_kv_coordinate_fallback_qps"
    )
    TOPOLOGY_KV_POLICY_BYPASS_QPS_METRIC = "py_rtp_topology_kv_policy_bypass_qps"


class GaugeMetrics(Enum):
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

    # igraph
    IGRAPH_RT_METRIC = "py_rtp_igraph_rt"
    PARSE_IGRAPH_RESPONSE_RT_METRIC = "py_rtp_parse_igraph_response_rt"

    # vit preprocess
    VIT_PREPROCESS_RT_METRIC = "py_rtp_vit_preprocess_rt"
    VIT_EMBEDDING_RT_METRIC = "py_rtp_vit_embedding_rt"
    VIT_RPC_SERVER_HANDLER_RT_US_METRIC = "rtp_llm_vit_rpc_server_handler_rt_us"
    VIT_RPC_SERVER_LIFECYCLE_RT_US_METRIC = "rtp_llm_vit_rpc_server_lifecycle_rt_us"
    VIT_RPC_PROXY_LIFECYCLE_RT_US_METRIC = "rtp_llm_vit_rpc_proxy_lifecycle_rt_us"
    VIT_RPC_PROXY_TO_WORKER_RT_US_METRIC = "rtp_llm_vit_rpc_proxy_to_worker_rt_us"
    VIT_RPC_REQUEST_BYTES_METRIC = "rtp_llm_vit_rpc_request_bytes"
    VIT_RPC_RESPONSE_BYTES_METRIC = "rtp_llm_vit_rpc_response_bytes"
    VIT_RESPONSE_EMBEDDING_BYTES_METRIC = "rtp_llm_vit_response_embedding_bytes"
    VIT_RESPONSE_DEEPSTACK_BYTES_METRIC = "rtp_llm_vit_response_deepstack_bytes"
    VIT_RESPONSE_POS_BYTES_METRIC = "rtp_llm_vit_response_pos_bytes"
    VIT_OUTPUT_TOKEN_COUNT_METRIC = "rtp_llm_vit_output_token_count"
    VIT_INPUT_IMAGE_COUNT_METRIC = "rtp_llm_vit_input_image_count"
    VIT_IMAGE_FETCH_RT_US_METRIC = "rtp_llm_vit_image_fetch_rt_us"
    VIT_IMAGE_DECODE_RT_US_METRIC = "rtp_llm_vit_image_decode_rt_us"
    VIT_IMAGE_RESIZE_RT_US_METRIC = "rtp_llm_vit_image_resize_rt_us"
    VIT_IMAGE_PROCESSOR_RT_US_METRIC = "rtp_llm_vit_image_processor_rt_us"
    VIT_RESIZED_PIXEL_COUNT_METRIC = "rtp_llm_vit_resized_pixel_count"

    # topology KV policy
    TOPOLOGY_KV_POLICY_SCHEDULE_MS_METRIC = "py_rtp_topology_kv_policy_schedule_ms"
    TOPOLOGY_KV_POLICY_SELECTED_TOKENS_METRIC = (
        "py_rtp_topology_kv_policy_selected_tokens"
    )
    TOPOLOGY_KV_POLICY_EVICTED_TOKENS_METRIC = (
        "py_rtp_topology_kv_policy_evicted_tokens"
    )


class MetricReporter(object):
    def __init__(self, kmonitor: Any):
        self._kmon = kmonitor
        self._matic_map: Dict[str, Any] = {}
        self._inited = False

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

    def init(self):
        if not self._inited:
            self._inited = True
            for metric in AccMetrics:
                self._matic_map[metric.value] = self._kmon.register_acc_metric(
                    metric.value
                )

            for metric in GaugeMetrics:
                self._matic_map[metric.value] = self._kmon.register_gauge_metric(
                    metric.value
                )
