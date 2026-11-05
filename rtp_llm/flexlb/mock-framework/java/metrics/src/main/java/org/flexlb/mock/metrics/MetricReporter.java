package org.flexlb.mock.metrics;

import io.prometheus.client.Collector.MetricFamilySamples;
import java.util.List;
import java.util.Map;

/** Optional internal push transport. Public builds contain no provider. */
public interface MetricReporter {
    void initialize(Map<String, String> environment);
    default long intervalMillis() { return 1000; }
    /** An original event sample; names ending in _qps carry counts, not precomputed rates. */
    void observe(String name, Map<String, String> tags, double value);
    /** Periodic state/rate snapshots; Prometheus-only counter and histogram helpers are ignored. */
    void report(List<MetricFamilySamples> snapshot);
}
