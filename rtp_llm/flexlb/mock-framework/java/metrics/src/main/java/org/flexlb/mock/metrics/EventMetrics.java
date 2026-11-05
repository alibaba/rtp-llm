package org.flexlb.mock.metrics;

import io.prometheus.client.*;
import java.util.*;
import java.util.function.LongSupplier;

/** One event API: Prometheus retains sum/count; an installed SDK receives original observations. */
public final class EventMetrics extends Collector {
    private static final System.Logger LOG = System.getLogger(EventMetrics.class.getName());
    private record Series(String name, Map<String, String> tags) {}
    private final String role;
    private final LongSupplier clock;
    private final MetricReporter reporter;
    private final Map<String, String> identityTags;
    private final CollectorRegistry registry = new CollectorRegistry();
    private final Map<String, Set<String>> labelNames = new HashMap<>();
    private final Map<String, Summary> summaries = new LinkedHashMap<>();
    private final Map<Series, Double> qps = new LinkedHashMap<>();
    private long started;

    public EventMetrics(String role) { this(role, System::nanoTime); }
    public EventMetrics(String role, LongSupplier clock) {
        this(role, clock, MetricTags.fromEnvironment(System.getenv()));
    }
    public EventMetrics(String role, LongSupplier clock, Map<String, String> identityTags) {
        this(role, clock, MetricsOutput.eventReporter(), identityTags);
    }
    EventMetrics(String role, LongSupplier clock, MetricReporter reporter, Map<String, String> identityTags) {
        this.role = Objects.requireNonNull(role);
        this.clock = Objects.requireNonNull(clock);
        this.reporter = reporter;
        this.identityTags = Map.copyOf(identityTags);
        started = clock.getAsLong();
    }

    public void observe(String name, double value) { observe(name, value, Map.of()); }
    public synchronized void observe(String name, double value, Map<String, String> tags) {
        if (!Double.isFinite(value) || value < 0 || tags.containsKey("role"))
            throw new IllegalArgumentException("Invalid metric observation");
        var previous = labelNames.putIfAbsent(name, Set.copyOf(tags.keySet()));
        if (previous != null && !previous.equals(tags.keySet()))
            throw new IllegalArgumentException("Inconsistent metric labels: " + name);
        var ordered = new TreeMap<>(tags);
        if (reporter != null) {
            ordered.put("role", role);
            ordered.putAll(identityTags);
            try { reporter.observe(name, Map.copyOf(ordered), value); }
            catch (RuntimeException error) { LOG.log(System.Logger.Level.ERROR, "Engine metric observation failed: " + name, error); }
            return;
        }
        if (name.endsWith("_qps")) {
            qps.merge(new Series(name, Map.copyOf(ordered)), value, Double::sum);
            return;
        }
        var names = new ArrayList<>(List.of("role")); names.addAll(ordered.keySet());
        var labels = new ArrayList<>(List.of(role)); labels.addAll(ordered.values());
        summaries.computeIfAbsent(name, ignored -> Summary.build().name(name)
            .help("Original event observations; aggregate sum and count before division.")
            .labelNames(names.toArray(String[]::new)).register(registry))
            .labels(labels.toArray(String[]::new)).observe(value);
    }

    public void increment(String name) { observe(name, 1); }

    @Override public synchronized List<MetricFamilySamples> collect() {
        if (reporter != null) return List.of();
        var result = new ArrayList<MetricFamilySamples>();
        registry.metricFamilySamples().asIterator().forEachRemaining(result::add);
        long now = clock.getAsLong(), elapsed = now - started;
        if (elapsed > 0) {
            var rates = new LinkedHashMap<String, GaugeMetricFamily>();
            for (var entry : qps.entrySet()) {
                var tags = new TreeMap<>(entry.getKey().tags());
                var names = new ArrayList<>(List.of("role")); names.addAll(tags.keySet());
                var labels = new ArrayList<>(List.of(role)); labels.addAll(tags.values());
                rates.computeIfAbsent(entry.getKey().name(), name -> new GaugeMetricFamily(name,
                    "Events per wall second.", names)).addMetric(labels, entry.getValue() * 1e9 / elapsed);
                entry.setValue(0.0);
            }
            result.addAll(rates.values());
            started = now;
        }
        return result;
    }
}
