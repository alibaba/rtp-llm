package org.flexlb.mock.metrics;

import io.prometheus.client.*;
import java.util.*;
import org.flexlb.mock.metrics.MetricReporter.EventKind;

/**
 * Records counter deltas and original event samples for the selected backend.
 */
public final class EventMetrics extends Collector {
    private static final System.Logger LOG = System.getLogger(EventMetrics.class.getName());

    private final String role;
    private final MetricReporter reporter;
    private final Map<String, String> identityTags;
    private final CollectorRegistry registry = new CollectorRegistry();
    private final Map<String, Schema> schemas = new HashMap<>();
    private final Map<String, Summary> summaries = new LinkedHashMap<>();
    private final Map<String, Counter> counters = new HashMap<>();

    public EventMetrics(String role) { this(role, MetricTags.fromEnvironment(System.getenv())); }

    public EventMetrics(String role, Map<String, String> identityTags) {
        this(role, MetricsOutput.eventReporter(), identityTags);
    }

    EventMetrics(String role, MetricReporter reporter, Map<String, String> identityTags) {
        this.role = Objects.requireNonNull(role);
        this.reporter = reporter;
        this.identityTags = Map.copyOf(identityTags);
    }

    public void increment(String name) { add(name, 1); }

    public void add(String name, double delta) { add(name, delta, Map.of()); }

    public void add(String name, double delta, Map<String, String> tags) {
        record(name, EventKind.DELTA, delta, tags);
    }

    public void observe(String name, double value) { observe(name, value, Map.of()); }

    public void observe(String name, double value, Map<String, String> tags) {
        record(name, EventKind.SAMPLE, value, tags);
    }

    @Override public List<MetricFamilySamples> collect() {
        var result = new ArrayList<MetricFamilySamples>();
        registry.metricFamilySamples().asIterator().forEachRemaining(result::add);
        return result;
    }

    private synchronized void record(String name, EventKind kind, double value, Map<String, String> tags) {
        if (!Double.isFinite(value) || value < 0 || tags.containsKey("role"))
            throw new IllegalArgumentException("Invalid metric observation");
        var schema = schemas.get(name);
        if (schema == null) schemas.put(name, new Schema(kind, Set.copyOf(tags.keySet())));
        else {
            if (schema.kind() != kind) throw new IllegalArgumentException("Inconsistent metric type: " + name);
            if (!schema.labels().equals(tags.keySet())) throw new IllegalArgumentException("Inconsistent metric labels: " + name);
        }
        var ordered = new TreeMap<>(tags);
        if (reporter != null) {
            ordered.put("role", role);
            ordered.putAll(identityTags);
            try { reporter.observe(name, kind, Map.copyOf(ordered), value); }
            catch (RuntimeException error) { LOG.log(System.Logger.Level.ERROR, "Engine metric observation failed: " + name, error); }
            return;
        }
        var names = new ArrayList<>(List.of("role")); names.addAll(ordered.keySet());
        var labels = new ArrayList<>(List.of(role)); labels.addAll(ordered.values());
        if (kind == EventKind.DELTA) {
            counters.computeIfAbsent(name, ignored -> Counter.build().name(name + "_total")
                .help("Cumulative events; use rate() for events per second.")
                .labelNames(names.toArray(String[]::new)).register(registry))
                .labels(labels.toArray(String[]::new)).inc(value);
            return;
        }
        summaries.computeIfAbsent(name, ignored -> Summary.build().name(name)
            .help("Original event observations; aggregate sum and count before division.")
            .labelNames(names.toArray(String[]::new)).register(registry))
            .labels(labels.toArray(String[]::new)).observe(value);
    }

    private record Schema(EventKind kind, Set<String> labels) {}
}
