package org.flexlb.service.grace;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.springframework.stereotype.Component;

import static org.flexlb.constant.MetricConstant.LIFECYCLE_EVENT_METRIC;

@Slf4j
@Component
public class GracefulLifecycleReporter {

    private static final String TYPE_TAG = "type";
    private static final String DURATION_MS_TAG = "duration_ms";

    private final FlexMonitor monitor;

    public GracefulLifecycleReporter(FlexMonitor monitor) {
        this.monitor = monitor;
        monitor.register(LIFECYCLE_EVENT_METRIC, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
    }

    public void reportHealthCheckOffline(long durationMs) {
        monitor.report(LIFECYCLE_EVENT_METRIC, FlexMetricTags.of(TYPE_TAG, "health_check_offline", DURATION_MS_TAG, String.valueOf(durationMs)), 1);
    }

    public void reportZkNodeOffline(long durationMs) {
        monitor.report(LIFECYCLE_EVENT_METRIC, FlexMetricTags.of(TYPE_TAG, "zk_node_offline", DURATION_MS_TAG, String.valueOf(durationMs)), 1);
    }

    public void reportShutdownTimeout(long durationMs) {
        monitor.report(LIFECYCLE_EVENT_METRIC, FlexMetricTags.of(TYPE_TAG, "shutdown_timeout", DURATION_MS_TAG, String.valueOf(durationMs)), 1);
    }

    public void reportShutdownComplete(long durationMs) {
        monitor.report(LIFECYCLE_EVENT_METRIC, FlexMetricTags.of(TYPE_TAG, "shutdown_complete", DURATION_MS_TAG, String.valueOf(durationMs)), 1);
    }

    public void reportProcessOk() {
        monitor.report(LIFECYCLE_EVENT_METRIC, FlexMetricTags.of(TYPE_TAG, "process_ok"), 1);
    }

    public void reportZkNodeOnline(long durationMs) {
        monitor.report(LIFECYCLE_EVENT_METRIC, FlexMetricTags.of(TYPE_TAG, "zk_node_online", DURATION_MS_TAG, String.valueOf(durationMs)), 1);
    }

    public void reportWarmerComplete(long durationMs) {
        monitor.report(LIFECYCLE_EVENT_METRIC, FlexMetricTags.of(TYPE_TAG, "warmer_complete", DURATION_MS_TAG, String.valueOf(durationMs)), 1);
    }

    public void reportOnlineComplete(long durationMs) {
        monitor.report(LIFECYCLE_EVENT_METRIC, FlexMetricTags.of(TYPE_TAG, "online_complete", DURATION_MS_TAG, String.valueOf(durationMs)), 1);
    }
}