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
        reportEvent("health_check_offline", durationMs);
    }

    public void reportZkNodeOffline(long durationMs) {
        reportEvent("zk_node_offline", durationMs);
    }

    public void reportShutdownTimeout(long durationMs) {
        reportEvent("shutdown_timeout", durationMs);
    }

    public void reportShutdownComplete(long durationMs) {
        reportEvent("shutdown_complete", durationMs);
    }

    public void reportProcessOk() {
        reportEvent("process_ok", 0);
    }

    public void reportZkNodeOnline(long durationMs) {
        reportEvent("zk_node_online", durationMs);
    }

    public void reportWarmerComplete(long durationMs) {
        reportEvent("warmer_complete", durationMs);
    }

    public void reportOnlineComplete(long durationMs) {
        reportEvent("online_complete", durationMs);
    }

    private void reportEvent(String type, long durationMs) {
        monitor.report(
                LIFECYCLE_EVENT_METRIC,
                FlexMetricTags.of(TYPE_TAG, type, DURATION_MS_TAG, String.valueOf(durationMs)),
                System.currentTimeMillis());
    }
}
