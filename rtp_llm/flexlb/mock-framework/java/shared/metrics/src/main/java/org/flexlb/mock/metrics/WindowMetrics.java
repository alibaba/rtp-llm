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
    private record Series(String name, Map<String, String> tags) {}
    private final String role;
    private final LongSupplier clock;
    private long started;
    private final Map<String, Set<String>> labelNames = new HashMap<>();
    private final Map<Series, Reading> values = new LinkedHashMap<>();

    public WindowMetrics(String role) { this(role, System::nanoTime); }
    public WindowMetrics(String role, LongSupplier clock) {
        this.role = Objects.requireNonNull(role); this.clock = clock; started = clock.getAsLong();
    }
    public void observe(String name, double value) { observe(name, value, Map.of()); }
    public synchronized void observe(String name, double value, Map<String, String> tags) {
        if (!Double.isFinite(value) || value < 0 || tags.containsKey("role")) throw new IllegalArgumentException("Invalid metric observation");
        var previous = labelNames.putIfAbsent(name, Set.copyOf(tags.keySet()));
        if (previous != null && !previous.equals(tags.keySet())) throw new IllegalArgumentException("Inconsistent metric labels: " + name);
        var series = new Series(name, Collections.unmodifiableMap(new TreeMap<>(tags)));
        var reading = values.computeIfAbsent(series, ignored -> new Reading());
        reading.sum += value; reading.total += value; reading.count++; reading.samples++;
    }
    public void increment(String name) { observe(name, 1); }
    private void add(Map<String, GaugeMetricFamily> result, String name, double value, Map<String, String> tags) {
        var names = new ArrayList<>(List.of("role")); names.addAll(tags.keySet());
        var labels = new ArrayList<>(List.of(role)); labels.addAll(tags.values());
        var family = result.computeIfAbsent(name, ignored -> new GaugeMetricFamily(name,
            "Native observation; event gauges use the scrape-window mean.", names));
        family.addMetric(labels, value);
    }
    @Override public synchronized List<MetricFamilySamples> collect() {
        long now = clock.getAsLong(), elapsed = now - started;
        if (elapsed <= 0) return List.of();
        var result = new LinkedHashMap<String, GaugeMetricFamily>();
        for (var entry : values.entrySet()) {
            String name = entry.getKey().name(); var tags = entry.getKey().tags(); var reading = entry.getValue();
            if (name.endsWith("_qps")) add(result, name, reading.sum * 1e9 / elapsed, tags);
            else {
                if (reading.count > 0) add(result, name, reading.sum / reading.count, tags);
                // Cumulative helpers retain rare events for queries between scrapes.
                add(result, name + "_sum", reading.total, tags);
                add(result, name + "_count", reading.samples, tags);
            }
            reading.sum = 0; reading.count = 0;
        }
        started = now; return new ArrayList<>(result.values());
    }
}
