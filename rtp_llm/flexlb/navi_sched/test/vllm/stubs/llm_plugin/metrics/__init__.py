class AccMetrics:
    CONFLICT_QPS_METRIC = "conflict_qps"
    ERROR_QPS_METRIC = "error_qps"
    QPS_METRIC = "qps"
    SUCCESS_QPS_METRIC = "success_qps"


class GaugeMetrics:
    LANTENCY_METRIC = "latency"
    RESPONSE_FIRST_TOKEN_RT_METRIC = "response_first_token_rt"
    RESPONSE_ITER_RT_METRIC = "response_iter_rt"


class _KMonitor:
    def init(self):
        return None

    def report(self, metric, value, tags=None):
        return None


kmonitor = _KMonitor()
