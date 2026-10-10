package org.flexlb.service.grace;

import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;
import org.springframework.stereotype.Component;

import java.util.Locale;

import static org.flexlb.constant.MetricConstant.GRACEFUL_LIFECYCLE_EVENT;

@Component
public class GracefulLifecycleReporter {

    private static final String TYPE_TAG = "type";
    private static final String DURATION_MS_TAG = "duration_ms";

    private final FlexMonitor monitor;

    public GracefulLifecycleReporter(FlexMonitor monitor) {
        this.monitor = monitor;
        monitor.register(GRACEFUL_LIFECYCLE_EVENT, FlexMetricType.GAUGE, FlexPriorityType.PRECISE);
    }

    public enum Event {
        HEALTH_CHECK_OFFLINE,
        ZK_NODE_OFFLINE,
        SHUTDOWN_COMPLETE,
        ZK_NODE_ONLINE,
        WARMER_COMPLETE,
        ONLINE_COMPLETE
    }

    public void reportDuration(Event event, long durationMs) {
        monitor.report(GRACEFUL_LIFECYCLE_EVENT, FlexMetricTags.of(TYPE_TAG,
                event.name().toLowerCase(Locale.ROOT), DURATION_MS_TAG, String.valueOf(durationMs)), 1);
    }

    public void reportProcessOk() {
        monitor.report(GRACEFUL_LIFECYCLE_EVENT, FlexMetricTags.of(TYPE_TAG, "process_ok"), 1);
    }
}
