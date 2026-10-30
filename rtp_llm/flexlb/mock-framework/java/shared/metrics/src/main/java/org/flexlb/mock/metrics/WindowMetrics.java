package org.flexlb.mock.metrics;

import io.prometheus.client.*;
import java.util.*;
import java.util.function.LongSupplier;

/** Native event gauges are averaged over each scrape window; QPS is events per wall second. */
public final class WindowMetrics extends Collector {
    private static final class Reading {
        double sum, total;
        long count, samples;
    }
    private final String role;
    private final LongSupplier clock;
    private long started;
    private final Map<String, Reading> values = new TreeMap<>();

    public WindowMetrics(String role) { this(role, System::nanoTime); }
    WindowMetrics(String role, LongSupplier clock) {
        this.role = Objects.requireNonNull(role); this.clock = clock; started = clock.getAsLong();
    }
    public synchronized void observe(String name, double value) {
        if (!Double.isFinite(value) || value < 0) throw new IllegalArgumentException("Invalid metric observation");
        var reading = values.computeIfAbsent(name, ignored -> new Reading());
        reading.sum += value; reading.total += value; reading.count++; reading.samples++;
    }
    public void increment(String name) { observe(name, 1); }
    private GaugeMetricFamily gauge(String name, double value) {
        var family = new GaugeMetricFamily(name, "Native observation; event gauges use the scrape-window mean.", List.of("role"));
        family.addMetric(List.of(role), value); return family;
    }
    @Override public synchronized List<MetricFamilySamples> collect() {
        long now = clock.getAsLong(), elapsed = now - started;
        if (elapsed <= 0) return List.of();
        var result = new ArrayList<MetricFamilySamples>();
        for (var entry : values.entrySet()) {
            String name = entry.getKey(); var reading = entry.getValue();
            if (name.endsWith("_qps")) result.add(gauge(name, reading.sum * 1e9 / elapsed));
            else {
                if (reading.count > 0) result.add(gauge(name, reading.sum / reading.count));
                // Cumulative helpers retain rare events for queries between scrapes.
                result.add(gauge(name + "_sum", reading.total));
                result.add(gauge(name + "_count", reading.samples));
            }
            reading.sum = 0; reading.count = 0;
        }
        started = now; return result;
    }
}
