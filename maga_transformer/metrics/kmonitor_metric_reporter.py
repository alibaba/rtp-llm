import logging
from enum import Enum
from typing import Dict, Any, Union

class AccMetrics(Enum):
    CANCAL_QPS_METRIC = "py_rtp_cancal_qps_metric"
    QPS_METRIC = "py_rtp_framework_qps"
    ERROR_QPS_METRIC = "py_rtp_framework_error_qps"
    CONFLICT_QPS_METRIC = "py_rtp_framework_concurrency_exception_qps"    
    ITER_QPS_METRIC = "py_rtp_response_iterate_qps"    
    UPDATE_QPS_METRIC = "py_rtp_update_qps_metric"
    FALLBACK_QPS_METRIC = "py_rtp_fallback_qps_metric"
    ERROR_UPDATE_QPS_METRIC = "py_rtp_error_update_target_qps"

class GaugeMetrics(Enum):
    RESPONSE_FIRST_TOKEN_RT_METRIC = "py_rtp_response_first_token_rt"
    RESPONSE_ITER_RT_METRIC = "py_rtp_response_iterate_rt"
    RESPONSE_ITERATE_COUNT = "py_rtp_response_iterate_count"
    LANTENCY_METRIC = "py_rtp_framework_rt"

    FT_ITERATE_COUNT_METRIC = "ft_iterate_count"
    FT_FIRST_TOKEN_RT_METRIC = "ft_iterate_first_token_rt"
    ERROR_EXIT_METRIC = "ft_error_exit"
    QUERY_BATCH_SIZE_METRIC = "ft_query_batch_size"
    INPUT_TOKEN_SIZE_METRIC = "ft_input_token_length"
    OUTPUT_TOKEN_SIZE_METRIC = "ft_output_token_length"
    PRE_PIPELINE_RT_METRIC = "ft_pre_pipeline_rt"
    POST_PIPELINE_RT_METRIC = "ft_post_pipeline_rt"
    NUM_BEAMS_METRIC = "ft_num_beams"

    ASYNC_BATCH_SIZE_METRIC = "ft_async_batch_size"
    ASYNC_WAIT_QUERY_SIZE_METRIC = "ft_async_wait_query_size"
    ASYNC_WAIT_WAIT_TIME_METRIC = "ft_async_wait_time"
    ASYNC_ITERATE_LANTENCY = "ft_async_iterate_rt"
    KV_CACHE_MEM_USED_RATIO_METRIC = "ft_kv_cache_mem_used_ratio"
    KV_CACHE_REUSE_LENGTH_METRIC = "ft_kvcache_reuse_length"
    KV_CACHE_ITEM_NUM_METRIC = "ft_kvcache_item_num"
    UPDATE_LANTENCY_METRIC = "py_rtp_update_framework_rt"

class MetricReporter(object):
    def __init__(self, kmonitor: Any):
        self._kmon = kmonitor
        self._matic_map: Dict[str, Any] = {}

        for metric in AccMetrics:
            self._matic_map[metric.value] = self._kmon.register_acc_metric(metric.value)

        for metric in GaugeMetrics:
            self._matic_map[metric.value] = self._kmon.register_gauge_metric(metric.value)

    def report(self, metric: Union[AccMetrics,GaugeMetrics], value: float = 1, tags: Dict[str, Any] = {}):
        kmon_metric = self._matic_map.get(metric.value, None)
        if kmon_metric is None:
            logging.warn(f"no metric named {metric.name}")
            return
        kmon_metric.report(value, tags)

    def flush(self) -> None:
        self._kmon.flush()
