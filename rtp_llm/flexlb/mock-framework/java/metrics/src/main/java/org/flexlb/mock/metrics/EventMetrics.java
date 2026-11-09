package org.flexlb.mock.metrics;

import io.prometheus.client.*;
import java.util.*;

/**
 * One event API: Prometheus retains counters and sum/count; an installed SDK receives original observations.
 */
public final class EventMetrics extends Collector {
    private static final System.Logger LOG = System.getLogger(EventMetrics.class.getName());

    private final String role;
    private final MetricReporter reporter;
    private final Map<String, String> identityTags;
    private final CollectorRegistry registry = new CollectorRegistry();
    private final Map<String, Set<String>> labelNames = new HashMap<>();
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

    public void increment(String name) { observe(name, 1); }

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
        var names = new ArrayList<>(List.of("role")); names.addAll(ordered.keySet());
        var labels = new ArrayList<>(List.of(role)); labels.addAll(ordered.values());
        if (isRate(name)) {
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

    @Override public List<MetricFamilySamples> collect() {
        var result = new ArrayList<MetricFamilySamples>();
        registry.metricFamilySamples().asIterator().forEachRemaining(result::add);
        return result;
    }

    // Native ViT embedding TPS is an accumulating token metric, just like event QPS.
    static boolean isRate(String name) { return name.endsWith("_qps") || name.equals("py_rtp_vit_embedding_tps"); }
}
