package org.flexlb.service.monitor;

import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;

import java.util.Map;

/**
 * Reports cumulative durations and counts for dispatches, plus gauges for closed 20-second windows.
 * The six stages share a sample count. PLE waits and failed dispatches have separate counters.
 */
final class TtftStageReporter {
    static final String PREFIX = "app.frontend.ttft.stages.";
    static final String[] METRICS =
            {"total.us", "before.prefill.us", "kv.alloc.us", "remote.kv.us", "scheduler.wait.us", "prefill.inflight.us",
                    "other.us", "requests", "ple.wait.us", "ple.requests", "failed.us", "failed.requests"};
    static final String WINDOW_PREFIX = PREFIX + "window20s.";
    private final FlexMonitor monitor;

    TtftStageReporter(FlexMonitor monitor) {
        this.monitor = monitor;
        for (String metric : METRICS) {
            monitor.register(PREFIX + metric, FlexMetricType.COUNTER, FlexPriorityType.NORMAL);
        }
        for (int index = 0; index < 7; index++) {
            monitor.register(WINDOW_PREFIX + METRICS[index], FlexMetricType.GAUGE, FlexPriorityType.NORMAL);
        }
        monitor.register(WINDOW_PREFIX + "samples", FlexMetricType.GAUGE, FlexPriorityType.NORMAL);
        monitor.register(WINDOW_PREFIX + "window.end.ms", FlexMetricType.GAUGE, FlexPriorityType.NORMAL);
    }

    void report(String deploymentName, long[] stages, Long pleWaitUs) {
        FlexMetricTags tags = FlexMetricTags.of("deployment_name", deploymentName);
        if (stages != null) {
            for (int index = 0; index < stages.length; index++) {
                monitor.report(PREFIX + METRICS[index], tags, stages[index]);
            }
            monitor.report(PREFIX + "requests", tags, 1);
        }
        if (pleWaitUs != null) {
            monitor.report(PREFIX + "ple.wait.us", tags, pleWaitUs);
            monitor.report(PREFIX + "ple.requests", tags, 1);
        }
    }

    void reportFailure(String deploymentName, long durationUs) {
        var tags = FlexMetricTags.of("deployment_name", deploymentName);
        monitor.report(PREFIX + "failed.us", tags, durationUs);
        monitor.report(PREFIX + "failed.requests", tags, 1);
    }

    void reportWindowStages(String deployment, String slot, String statistic, double[] stages) {
        FlexMetricTags tags =
                FlexMetricTags.of(Map.of("deployment_name", deployment, "slot", slot, "statistic", statistic));
        for (int index = 0; index < stages.length; index++) {
            monitor.report(WINDOW_PREFIX + METRICS[index], tags, stages[index]);
        }
    }

    void reportWindowMetadata(String deployment, String slot, long endMs, long samples) {
        FlexMetricTags tags = FlexMetricTags.of("deployment_name", deployment, "slot", slot);
        monitor.report(WINDOW_PREFIX + "samples", tags, samples);
        monitor.report(WINDOW_PREFIX + "window.end.ms", tags, endMs);
    }
}
