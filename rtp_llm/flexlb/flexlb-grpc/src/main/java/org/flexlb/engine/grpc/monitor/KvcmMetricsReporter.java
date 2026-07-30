package org.flexlb.engine.grpc.monitor;

import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;

import static org.flexlb.constant.MetricConstant.KVCM_QUERY_RETRY_QPS;

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
    }

    public void reportQueryRetry(int attempt) {
        monitor.report(
                KVCM_QUERY_RETRY_QPS,
                FlexMetricTags.of("attempt", String.valueOf(attempt)),
                1.0);
    }
}
