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
