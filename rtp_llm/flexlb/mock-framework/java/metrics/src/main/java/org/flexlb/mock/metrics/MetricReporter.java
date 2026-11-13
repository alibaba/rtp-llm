package org.flexlb.mock.metrics;

import io.prometheus.client.Collector.MetricFamilySamples;
import java.util.List;
import java.util.Map;

/**
 * Receives typed events and periodic state snapshots for the optional push backend.
 */
public interface MetricReporter {
    void initialize(Map<String, String> environment);

    default long intervalMillis() { return 1000; }

    /** Records a counter delta or an original event sample. */
    void observe(String name, EventKind kind, Map<String, String> tags, double value);

    /** Periodic state/rate snapshots; Prometheus-only counter and histogram helpers are ignored. */
    void report(List<MetricFamilySamples> snapshot);

    enum EventKind { DELTA, SAMPLE }
}
