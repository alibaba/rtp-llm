package org.flexlb.metric;

import org.flexlb.enums.FlexMetricType;
import org.flexlb.enums.FlexPriorityType;

/**
 * No-operation implementation of FlexMonitor interface.
 * All methods are no-ops and return safe default values.
 */
public class NoOpFlexMonitor implements FlexMonitor {

    private static final NoOpFlexMonitor INSTANCE = new NoOpFlexMonitor();

    public static NoOpFlexMonitor getInstance() {
        return INSTANCE;
    }

    @Override
    public void register(String metricName, FlexMetricType metricType) {
    }

    @Override
    public void register(String metricName, FlexMetricType metricType, FlexPriorityType priorityType) {
    }

    @Override
    public void register(String metricName, FlexMetricType metricType, int statisticsType) {
    }

    @Override
    public void report(String metricName, double value) {
        // No-op
    }

    @Override
    public void report(String metricName, FlexMetricTags metricsTags, double value) {
        // No-op
    }

    @Override
    public String toString() {
        return "NoOpFlexMonitor";
    }
}
