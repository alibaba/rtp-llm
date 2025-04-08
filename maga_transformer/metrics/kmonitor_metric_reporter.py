import logging
from enum import Enum
from typing import Dict, Any, Union
from maga_transformer.distribute.worker_info import g_parallel_info, g_frontend_server_info

class AccMetrics(Enum):
    CANCEL_QPS_METRIC = "py_rtp_cancal_qps_metric"
    SUCCESS_QPS_METRIC = "py_rtp_success_qps_metric"
    QPS_METRIC = "py_rtp_framework_qps"
    ERROR_QPS_METRIC = "py_rtp_framework_error_qps"
    CONFLICT_QPS_METRIC = "py_rtp_framework_concurrency_exception_qps"
    ITER_QPS_METRIC = "py_rtp_response_iterate_qps"
    UPDATE_QPS_METRIC = "py_rtp_update_qps_metric"
    ERROR_UPDATE_QPS_METRIC = "py_rtp_error_update_target_qps"

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

class MetricReporter(object):
    def __init__(self, kmonitor: Any):
        self._kmon = kmonitor
        self._matic_map: Dict[str, Any] = {}
        self._inited = False

    def report(self, metric: Union[AccMetrics,GaugeMetrics], value: float = 1, tags: Dict[str, Any] = {}):
        if g_parallel_info.dp_size > 1:
            tags['dp_rank'] = str(g_parallel_info.dp_rank)
        tags['frontend_server_id'] = str(g_frontend_server_info.frontend_server_id)
        kmon_metric = self._matic_map.get(metric.value, None)
        if kmon_metric is None:
            logging.warn(f"no metric named {metric.name}")
            return
        kmon_metric.report(value, tags)

    def flush(self) -> None:
        self._kmon.flush()

    def init(self):
        if not self._inited:
            self._inited = True
            for metric in AccMetrics:
                self._matic_map[metric.value] = self._kmon.register_acc_metric(metric.value)

            for metric in GaugeMetrics:
                self._matic_map[metric.value] = self._kmon.register_gauge_metric(metric.value)
