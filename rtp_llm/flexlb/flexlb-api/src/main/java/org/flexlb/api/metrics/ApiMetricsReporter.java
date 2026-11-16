package org.flexlb.api.metrics;

import org.flexlb.master.ZkMasterEvent;
import org.flexlb.util.metrics.FlexMetricTags;
import org.flexlb.util.metrics.FlexMetricType;
import org.flexlb.util.metrics.FlexMonitor;
import org.flexlb.util.metrics.FlexPriorityType;
import org.flexlb.util.metrics.FlexStatisticsType;
import org.flexlb.util.metrics.SafeFlexMonitor;
import org.springframework.stereotype.Component;

import java.util.Locale;

import static org.flexlb.util.metrics.MetricConstant.ENGINE_BALANCING_MASTER_ALL_QPS;
import static org.flexlb.util.metrics.MetricConstant.ENGINE_BALANCING_MASTER_ALL_RT;
import static org.flexlb.util.metrics.MetricConstant.FORWARD_TO_MASTER_RESULT;
import static org.flexlb.util.metrics.MetricConstant.GRACEFUL_LIFECYCLE_EVENT;
import static org.flexlb.util.metrics.MetricConstant.GRPC_SERVER_PROCESS_MS;
import static org.flexlb.util.metrics.MetricConstant.REQUEST_BLOCK_SIZE;
import static org.flexlb.util.metrics.MetricConstant.REQUEST_MESSAGE_BYTES;
import static org.flexlb.util.metrics.MetricConstant.REQUEST_NETWORK_DELAY_MS;
import static org.flexlb.util.metrics.MetricConstant.REQUEST_SEQ_LEN;
import static org.flexlb.util.metrics.MetricConstant.ZK_MASTER_EVENT;
import static org.flexlb.util.metrics.MetricConstant.ZK_MASTER_EVENT_TIME_MS;
import static org.flexlb.util.metrics.MetricConstant.ZK_MASTER_NODE;

/**
 * Reports API request, forwarding and application lifecycle metrics, including Master events.
 */
@Component
public final class ApiMetricsReporter implements org.flexlb.master.MasterMetrics {
    private final FlexMonitor monitor;

    public ApiMetricsReporter(FlexMonitor monitor) {
        this.monitor = SafeFlexMonitor.wrap(monitor);
        this.monitor.register(REQUEST_BLOCK_SIZE, FlexMetricType.GAUGE);
        this.monitor.register(ZK_MASTER_NODE, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(ZK_MASTER_EVENT, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(
                ZK_MASTER_EVENT_TIME_MS, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(
                REQUEST_NETWORK_DELAY_MS, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(
                GRPC_SERVER_PROCESS_MS, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
        this.monitor.register(REQUEST_SEQ_LEN, FlexMetricType.GAUGE, FlexStatisticsType.SUMMARY);
        this.monitor.register(
                REQUEST_MESSAGE_BYTES, FlexMetricType.GAUGE, FlexStatisticsType.SUMMARY);
        this.monitor.register(
                FORWARD_TO_MASTER_RESULT, FlexMetricType.QPS, FlexPriorityType.PRECISE);
        this.monitor.register(ENGINE_BALANCING_MASTER_ALL_QPS, FlexMetricType.QPS);
        this.monitor.register(
                ENGINE_BALANCING_MASTER_ALL_RT, FlexMetricType.TIMER, FlexPriorityType.PRECISE);
        this.monitor.register(
                GRACEFUL_LIFECYCLE_EVENT, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
    }

    public void reportBalancingService(int code, long elapsedMs) {
        FlexMetricTags metricTags = FlexMetricTags.of("code", String.valueOf(code));
        monitor.report(ENGINE_BALANCING_MASTER_ALL_QPS, metricTags, 1.0);
        monitor.report(ENGINE_BALANCING_MASTER_ALL_RT, metricTags, elapsedMs);
    }

    @Override
    public void reportMasterNode(String master) {
        monitor.report(ZK_MASTER_NODE, FlexMetricTags.of("masterNode", master), 1.0);
    }

    @Override
    public void reportPrefillBalanceMasterEvent(ZkMasterEvent event) {
        monitor.report(ZK_MASTER_EVENT, FlexMetricTags.of("event", event.name()), 1.0);
        monitor.report(
                ZK_MASTER_EVENT_TIME_MS,
                FlexMetricTags.of("event", event.name()),
                System.currentTimeMillis());
    }

    public void reportArriveDelayTime(long requestTimeMs, long grpcEntryTime, long startTimeMs) {
        if (requestTimeMs == 0) {
            return;
        }
        if (grpcEntryTime > 0) {
            long networkDelayMs = grpcEntryTime - requestTimeMs;
            long grpcProcessMs = startTimeMs - grpcEntryTime;
            monitor.report(REQUEST_NETWORK_DELAY_MS, FlexMetricTags.of(), networkDelayMs);
            monitor.report(GRPC_SERVER_PROCESS_MS, FlexMetricTags.of(), grpcProcessMs);
        } else {
            // Fallback: if grpcEntryTime not set, report total delay as network delay
            long arrivalDelayMs = startTimeMs - requestTimeMs;
            monitor.report(REQUEST_NETWORK_DELAY_MS, FlexMetricTags.of(), arrivalDelayMs);
        }
    }

    public void reportForwardToMasterResult(String type, String code) {
        monitor.report(
                FORWARD_TO_MASTER_RESULT, FlexMetricTags.of("type", type, "code", code), 1.0);
    }

    public void reportProcessOk() {
        monitor.report(GRACEFUL_LIFECYCLE_EVENT, FlexMetricTags.of("type", "process_ok"), 1);
    }

    public void reportDuration(Event event, long durationMs) {
        monitor.report(
                GRACEFUL_LIFECYCLE_EVENT,
                FlexMetricTags.of(
                        "type",
                        event.name().toLowerCase(Locale.ROOT),
                        "duration_ms",
                        String.valueOf(durationMs)),
                1);
    }

    public void reportRequestPayload(
            boolean success, long inputTokens, long blockSize, Long messageBytes) {
        var tags = FlexMetricTags.of("success", String.valueOf(success));
        if (inputTokens >= 0) monitor.report(REQUEST_SEQ_LEN, tags, inputTokens);
        if (blockSize >= 0) monitor.report(REQUEST_BLOCK_SIZE, tags, blockSize);
        if (messageBytes != null) monitor.report(REQUEST_MESSAGE_BYTES, tags, messageBytes);
    }

    public enum Event {
        HEALTH_CHECK_OFFLINE,
        ZK_NODE_OFFLINE,
        SHUTDOWN_COMPLETE,
        ZK_NODE_ONLINE,
        WARMER_COMPLETE,
        ONLINE_COMPLETE
    }
}
