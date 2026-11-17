package org.flexlb.engine.grpc.monitor;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;

import static org.flexlb.constant.MetricConstant.KVCM_QUERY_FAILURE_QPS;
import static org.flexlb.constant.MetricConstant.KVCM_QUERY_RETRY_QPS;

/**
 * Reports KVCM query behavior owned by the gRPC client.
 */
@Slf4j
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
        report(KVCM_QUERY_RETRY_QPS, FlexMetricTags.of("attempt", String.valueOf(attempt)));
    }

    /**
     * Reports a KVCM query that failed after all configured retry attempts.
     */
    public void reportQueryFailure() {
        report(KVCM_QUERY_FAILURE_QPS, FlexMetricTags.of());
    }

    private void report(String name, FlexMetricTags tags) {
        try {
            monitor.report(name, tags, 1.0);
        } catch (RuntimeException error) {
            log.debug("Failed to report KVCM query metric {}", name, error);
        }
    }
}
