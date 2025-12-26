package org.flexlb.service.monitor;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;

import static org.flexlb.constant.MetricConstant.ROUTING_QUEUE_CANCELLED_QPS;
import static org.flexlb.constant.MetricConstant.ROUTING_QUEUE_LENGTH;
import static org.flexlb.constant.MetricConstant.ROUTING_QUEUE_REJECTED_QPS;
import static org.flexlb.constant.MetricConstant.ROUTING_QUEUE_TIMEOUT_QPS;
import static org.flexlb.constant.MetricConstant.ROUTING_QUEUE_WAIT_TIME_MS;

/**
 * 排队指标监控器
 *
 * @author saichen.sm
 * @since 2025/12/22
 */
@Slf4j
@Component
public class RoutingQueueReporter {

    private final FlexMonitor monitor;
    private final FlexMetricTags tags = FlexMetricTags.of();

    @Autowired
    public RoutingQueueReporter(FlexMonitor monitor) {
        this.monitor = monitor;
    }

    /**
     * 初始化注册所有排队相关的监控指标
     */
    @PostConstruct
    public void init() {
        monitor.register(ROUTING_QUEUE_LENGTH, FlexMetricType.GAUGE);
        monitor.register(ROUTING_QUEUE_TIMEOUT_QPS, FlexMetricType.QPS);
        monitor.register(ROUTING_QUEUE_REJECTED_QPS, FlexMetricType.QPS);
        monitor.register(ROUTING_QUEUE_CANCELLED_QPS, FlexMetricType.QPS);
        monitor.register(ROUTING_QUEUE_WAIT_TIME_MS, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);

        log.info("RoutingQueueReporter initialized and registered with KMonitor");
    }

    public void reportQueueMetric(long queueLength, long waitTimeMs) {
        monitor.report(ROUTING_QUEUE_LENGTH, tags, queueLength);
        monitor.report(ROUTING_QUEUE_WAIT_TIME_MS, tags, waitTimeMs);
    }

    public void reportTimeout() {
        monitor.report(ROUTING_QUEUE_TIMEOUT_QPS, tags, 1.0);
    }

    public void reportRejected() {
        monitor.report(ROUTING_QUEUE_REJECTED_QPS, tags, 1.0);
    }

    public void reportCancelled() {
        monitor.report(ROUTING_QUEUE_CANCELLED_QPS, tags, 1.0);
    }
}
