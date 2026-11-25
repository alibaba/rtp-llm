package org.flexlb.service.monitor;

import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;

/**
 * Publishes P engine TTFT for closed 20-second windows and independent PLE waits.
 */
final class TtftStageReporter {
    static final String PREFIX = "app.prefill.engine.ttft.stages.";
    static final String[] METRICS =
            {"total.us", "kv.alloc.us", "remote.kv.us", "scheduler.wait.us", "prefill.inflight.us",
                    "other.us"};
    static final String WINDOW_PREFIX = PREFIX + "window20s.";
    private final FlexMonitor monitor;

    TtftStageReporter(FlexMonitor monitor) {
        this.monitor = monitor;
        for (String metric : METRICS) {
            monitor.register(WINDOW_PREFIX + metric, FlexMetricType.GAUGE, FlexPriorityType.NORMAL);
        }
        monitor.register(PREFIX + "ple.wait.us", FlexMetricType.COUNTER, FlexPriorityType.NORMAL);
        monitor.register(PREFIX + "ple.requests", FlexMetricType.COUNTER, FlexPriorityType.NORMAL);
        monitor.register(WINDOW_PREFIX + "samples", FlexMetricType.GAUGE, FlexPriorityType.NORMAL);
        monitor.register(WINDOW_PREFIX + "window.end.ms", FlexMetricType.GAUGE, FlexPriorityType.NORMAL);
    }

    void reportPle(long waitUs) {
        monitor.report(PREFIX + "ple.wait.us", waitUs);
        monitor.report(PREFIX + "ple.requests", 1);
    }

    void reportWindowStages(String slot, String statistic, double[] stages) {
        FlexMetricTags tags = FlexMetricTags.of("slot", slot, "statistic", statistic);
        for (int index = 0; index < stages.length; index++) {
            monitor.report(WINDOW_PREFIX + METRICS[index], tags, stages[index]);
        }
    }

    void reportWindowMetadata(String slot, long endMs, long samples) {
        FlexMetricTags tags = FlexMetricTags.of("slot", slot);
        monitor.report(WINDOW_PREFIX + "samples", tags, samples);
        monitor.report(WINDOW_PREFIX + "window.end.ms", tags, endMs);
    }
}
