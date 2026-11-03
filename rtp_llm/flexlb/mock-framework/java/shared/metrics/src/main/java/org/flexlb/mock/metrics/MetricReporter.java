package org.flexlb.mock.metrics;

import io.prometheus.client.Collector.MetricFamilySamples;
import java.util.List;
import java.util.Map;

/**
 * Optional internal push transport. Public builds contain no provider.
 */
public interface MetricReporter {
    void initialize(Map<String, String> environment);
    default long intervalMillis() { return 1000; }
    /**
     * Values are already sampled: QPS values are rates, counters are cumulative.
     */
    void report(List<MetricFamilySamples> snapshot);
}
