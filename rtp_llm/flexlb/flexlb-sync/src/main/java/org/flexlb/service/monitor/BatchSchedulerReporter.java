package org.flexlb.service.monitor;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import javax.annotation.PostConstruct;

import static org.flexlb.constant.MetricConstant.ENGINE_BALANCING_MASTER_SELECT_DETAIL;
import static org.flexlb.constant.MetricConstant.ENGINE_LOCAL_TASK_MAP_SIZE;
import static org.flexlb.constant.MetricConstant.ENGINE_RUNNING_TASK_INFO_SIZE;
import static org.flexlb.constant.MetricConstant.ROUTING_QUEUE_LENGTH;
import static org.flexlb.constant.MetricConstant.ROUTING_QUEUE_WAIT_TIME_MS;

/**
 * Batch scheduling metrics reporter for FlexLB batch dispatch path.
 *
 * <p>Reuses existing dashboard metric keys so batch-path data appears
 * on the same Grafana panels as the non-batch path:
 * queue (routing.queue.length + routing.queue.wait.time.ms),
 * dispatch reason (engine.balancing.master.select.detail),
 * inflight (health.check.local.task.map.size + health.check.running.task.info.size).
 */
@Slf4j
@Component
public class BatchSchedulerReporter {

    private final FlexMonitor monitor;

    @Autowired
    public BatchSchedulerReporter(FlexMonitor monitor) {
        this.monitor = monitor;
    }

    @PostConstruct
    public void init() {
        // Queue — same type as RoutingQueueReporter
        monitor.register(ROUTING_QUEUE_LENGTH, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        monitor.register(ROUTING_QUEUE_WAIT_TIME_MS, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);

        // Dispatch reason — same type as EngineHealthReporter
        monitor.register(ENGINE_BALANCING_MASTER_SELECT_DETAIL, FlexMetricType.QPS, FlexPriorityType.PRECISE);

        // Inflight — same type as EngineHealthReporter
        monitor.register(ENGINE_LOCAL_TASK_MAP_SIZE, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        monitor.register(ENGINE_RUNNING_TASK_INFO_SIZE, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);

        log.info("BatchSchedulerReporter initialized (5 metrics reusing existing dashboard keys)");
    }

    // ==================== Queue metrics ====================

    /**
     * Report per-worker batcher queue depth via {@code routing.queue.length}.
     */
    public void reportBatcherQueueDepth(String role, String engineIp, int depth) {
        FlexMetricTags tags = FlexMetricTags.of(
                "type", "batchQueue",
                "role", role,
                "engineIp", engineIp);
        monitor.report(ROUTING_QUEUE_LENGTH, tags, depth);
    }

    /**
     * Report batch wait time (enqueue to dispatch) via {@code routing.queue.wait.time.ms}.
     */
    public void reportBatchWaitTimeMs(String role, String engineIp, long waitMs) {
        FlexMetricTags tags = FlexMetricTags.of(
                "role", role,
                "engineIp", engineIp);
        monitor.report(ROUTING_QUEUE_WAIT_TIME_MS, tags, waitMs);
    }

    // ==================== Dispatch reason metrics ====================

    /**
     * Report batch dispatch reason via {@code engine.balancing.master.select.detail}.
     */
    public void reportDispatchReason(String role, String engineIp, String reason) {
        FlexMetricTags tags = FlexMetricTags.of(
                "role", role,
                "engineIp", engineIp,
                "reason", reason);
        monitor.report(ENGINE_BALANCING_MASTER_SELECT_DETAIL, tags, 1.0);
    }

    // ==================== Inflight metrics ====================

    /**
     * Report scheduler inflight map size via {@code health.check.local.task.map.size}.
     * Uses role=prefill + engineIp=scheduler tags to match the Grafana panel filter.
     */
    public void reportSchedulerInflightSize(int size) {
        FlexMetricTags tags = FlexMetricTags.of(
                "role", "prefill",
                "engineIp", "scheduler");
        monitor.report(ENGINE_LOCAL_TASK_MAP_SIZE, tags, size);
    }

    /**
     * Report per-worker prefilled endpoint inflight batch count via {@code health.check.running.task.info.size}.
     */
    public void reportPrefillInflightBatchCount(String role, String engineIp, int count) {
        FlexMetricTags tags = FlexMetricTags.of(
                "role", role,
                "engineIp", engineIp);
        monitor.report(ENGINE_RUNNING_TASK_INFO_SIZE, tags, count);
    }
}
