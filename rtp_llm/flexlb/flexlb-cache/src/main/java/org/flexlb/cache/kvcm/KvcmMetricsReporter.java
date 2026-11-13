package org.flexlb.cache.kvcm;

import org.flexlb.util.metrics.FlexMetricTags;
import org.flexlb.util.metrics.FlexMetricType;
import org.flexlb.util.metrics.FlexMonitor;
import org.flexlb.util.metrics.FlexPriorityType;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;

import static org.flexlb.util.metrics.MetricConstant.KVCM_QUERY_FAILURE_QPS;
import static org.flexlb.util.metrics.MetricConstant.KVCM_QUERY_RETRY_QPS;

/**
 * Reports KVCM query behavior owned by the gRPC client.
 */
@Component
public class KvcmMetricsReporter {

    private final FlexMonitor monitor;

    public KvcmMetricsReporter(FlexMonitor monitor) {
        this.monitor = monitor;
    }

    @PostConstruct
    public void init() {
        monitor.register(KVCM_QUERY_RETRY_QPS, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        monitor.register(KVCM_QUERY_FAILURE_QPS, FlexMetricType.QPS, FlexPriorityType.PRECISE);
    }

    public void reportQueryRetry(int attempt) {
        monitor.report(
                KVCM_QUERY_RETRY_QPS, FlexMetricTags.of("attempt", String.valueOf(attempt)), 1.0);
    }

    /**
     * Reports a KVCM query that failed after all configured retry attempts.
     */
    public void reportQueryFailure() {
        monitor.report(KVCM_QUERY_FAILURE_QPS, FlexMetricTags.of(), 1.0);
    }
}
