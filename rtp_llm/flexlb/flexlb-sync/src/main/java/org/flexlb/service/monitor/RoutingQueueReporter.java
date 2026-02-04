package org.flexlb.service.monitor;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;

import static org.flexlb.constant.MetricConstant.ROUTING_FAILURE_QPS;
import static org.flexlb.constant.MetricConstant.ROUTING_QUEUE_CANCELLED_QPS;
import static org.flexlb.constant.MetricConstant.ROUTING_QUEUE_ENTRY_QPS;
import static org.flexlb.constant.MetricConstant.ROUTING_QUEUE_LENGTH;
import static org.flexlb.constant.MetricConstant.ROUTING_QUEUE_REJECTED_QPS;
import static org.flexlb.constant.MetricConstant.ROUTING_QUEUE_TIMEOUT_QPS;
import static org.flexlb.constant.MetricConstant.ROUTING_QUEUE_WAIT_TIME_MS;
import static org.flexlb.constant.MetricConstant.ROUTING_RETRY_QPS;
import static org.flexlb.constant.MetricConstant.ROUTING_ROUTE_EXECUTION_TIME_MS;
import static org.flexlb.constant.MetricConstant.ROUTING_SUCCESS_QPS;

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
        monitor.register(ROUTING_QUEUE_LENGTH, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        monitor.register(ROUTING_QUEUE_ENTRY_QPS, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        monitor.register(ROUTING_QUEUE_TIMEOUT_QPS, FlexMetricType.QPS);
        monitor.register(ROUTING_QUEUE_REJECTED_QPS, FlexMetricType.QPS);
        monitor.register(ROUTING_QUEUE_CANCELLED_QPS, FlexMetricType.QPS);
        monitor.register(ROUTING_QUEUE_WAIT_TIME_MS, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        monitor.register(ROUTING_ROUTE_EXECUTION_TIME_MS, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);

        // 路由状态监控指标
        monitor.register(ROUTING_SUCCESS_QPS, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        monitor.register(ROUTING_FAILURE_QPS, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        monitor.register(ROUTING_RETRY_QPS, FlexMetricType.QPS, FlexPriorityType.PRECISE);

        log.info("RoutingQueueReporter initialized and registered with KMonitor");
    }

    public void reportQueueSize(long queueSize) {
        monitor.report(ROUTING_QUEUE_LENGTH, FlexMetricTags.of("type", "mainQueue"), queueSize);
    }

    public void reportQueueWaitingMetric(long waitTimeMs) {
        monitor.report(ROUTING_QUEUE_WAIT_TIME_MS, tags, waitTimeMs);
    }

    public void reportRouteExecutionMetric(long routeExecutionTimeMs) {
        monitor.report(ROUTING_ROUTE_EXECUTION_TIME_MS, tags, routeExecutionTimeMs);
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

    /**
     * 上报入队列指标
     */
    public void reportQueueEntry() {
        monitor.report(ROUTING_QUEUE_ENTRY_QPS, tags, 1.0);
    }

    /**
     * 上报路由成功指标
     */
    public void reportRoutingSuccessQps(int retryTimes) {
        monitor.report(ROUTING_SUCCESS_QPS, tags, 1.0);
        monitor.report(ROUTING_RETRY_QPS, tags, retryTimes);
    }

    /**
     * 上报路由失败指标
     */
    public void reportRoutingFailureQps(int code) {
        monitor.report(ROUTING_FAILURE_QPS, FlexMetricTags.of("code", String.valueOf(code)), 1.0);
    }
}
