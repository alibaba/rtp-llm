package org.flexlb.mock.metrics;

import io.prometheus.client.*;
import io.prometheus.client.exporter.common.TextFormat;
import java.io.*;
import java.util.*;
import java.util.function.Supplier;

/** Common engine observations. The engine decides when work/cache events actually occurred. */
public final class EngineMetrics {
    public static final String CACHE_LOOKUPS = "mock_engine_cache_lookup_tokens_total";
    public static final String CACHE_HITS = "mock_engine_cache_hit_tokens_total";
    public static final String FORWARD_TOKENS = "mock_engine_forward_tokens_total";
    public static final String FORWARD_SECONDS = "mock_engine_forward_seconds";
    public enum Outcome { success, cancelled, timeout, error }
    public record State(long waiting, long running, long totalTokens, long availableTokens,
                        long freeTokens, long heldTokens) {
        public State {
            if (waiting < 0 || running < 0 || totalTokens < 0 || availableTokens < 0 || freeTokens < 0
                    || heldTokens < 0 || availableTokens > totalTokens || freeTokens > availableTokens || heldTokens > totalTokens)
                throw new IllegalArgumentException("Invalid engine state");
        }
    }
    private static final String[] LABELS = {"role"};
    private final CollectorRegistry registry = new CollectorRegistry();
    private final Counter.Child started, cacheInputs, cacheHits;
    private final Map<Outcome, Counter.Child> finished = new EnumMap<>(Outcome.class);
    private final Map<String, Counter.Child> tokens = new HashMap<>();
    private final Histogram.Child forward;
    private final Gauge.Child actualForward, plannedForward;

    public EngineMetrics(String role, Supplier<State> state) {
        if (role == null || role.isBlank()) throw new IllegalArgumentException("Metric role must not be blank");
        Objects.requireNonNull(state);
        String[] identity = {role};
        started = counter("mock_engine_requests_started_total", "Accepted engine requests.").labels(identity);
        cacheInputs = counter(CACHE_LOOKUPS, "Input tokens considered by successful KV admission.").labels(identity);
        cacheHits = counter(CACHE_HITS, "Historical prefix tokens reused at KV admission.").labels(identity);
        var results = Counter.build().name("mock_engine_requests_finished_total").help("Terminal engine requests by outcome.")
            .labelNames("role", "outcome").register(registry);
        for (var outcome : Outcome.values()) finished.put(outcome, results.labels(role, outcome.name()));
        var work = Counter.build().name(FORWARD_TOKENS).help("Tokens in completed forwards; input includes cache, compute excludes it.")
            .labelNames("role", "kind").register(registry);
        for (String kind : List.of("input", "compute", "output")) tokens.put(kind, work.labels(role, kind));
        forward = Histogram.build().name(FORWARD_SECONDS).help("Elapsed time of completed simulated forwards, excluding request queue time.")
            .labelNames(LABELS).register(registry).labels(identity);
        actualForward = Gauge.build().name("rtp_llm_model_forward_us").help("Actual simulated Forward duration in microseconds.")
            .labelNames(LABELS).register(registry).labels(identity);
        plannedForward = Gauge.build().name("mock_model_forward_planned_us").help("Planned Forward duration in microseconds.")
            .labelNames(LABELS).register(registry).labels(identity);
        // Pull one state snapshot per scrape: no cached gauge values or background sampler.
        new Collector() {
            @Override public List<MetricFamilySamples> collect() {
                var value = Objects.requireNonNull(state.get());
                var tasks = new GaugeMetricFamily("mock_engine_requests", "Current accepted requests.", List.of("role", "state"));
                tasks.addMetric(List.of(role, "waiting"), value.waiting());
                tasks.addMetric(List.of(role, "running"), value.running());
                var kv = new GaugeMetricFamily("mock_engine_kv_tokens", "KV capacity in tokens; available includes evictable unreferenced cache.", List.of("role", "state"));
                kv.addMetric(List.of(role, "total"), value.totalTokens());
                kv.addMetric(List.of(role, "available"), value.availableTokens());
                kv.addMetric(List.of(role, "free"), value.freeTokens());
                kv.addMetric(List.of(role, "held"), value.heldTokens());
                return List.of(tasks, kv);
            }
        }.register(registry);
    }

    private Counter counter(String name, String help) {
        return Counter.build().name(name).help(help).labelNames(LABELS).register(registry);
    }
    public void started() { started.inc(); }
    public void finished(Outcome outcome) { finished.get(Objects.requireNonNull(outcome)).inc(); }
    public void cacheLookup(long input, long reused) {
        if (input < 0 || reused < 0 || reused > input) throw new IllegalArgumentException("Invalid cache observation");
        cacheInputs.inc(input); cacheHits.inc(reused);
    }
    public void forwardCompleted(long elapsedNanos, long plannedNanos, long input, long computed, long output) {
        if (elapsedNanos < 0 || plannedNanos < 0 || input < 0 || computed < 0 || computed > input || output < 0)
            throw new IllegalArgumentException("Invalid forward observation");
        tokens.get("input").inc(input); tokens.get("compute").inc(computed); tokens.get("output").inc(output);
        forward.observe(elapsedNanos / 1_000_000_000.0);
        actualForward.set(elapsedNanos / 1000.0);
        plannedForward.set(plannedNanos / 1000.0);
    }
    public String scrape() {
        var text = new StringWriter();
        try { TextFormat.write004(text, registry.metricFamilySamples()); }
        catch (IOException impossible) { throw new UncheckedIOException(impossible); }
        return text.toString();
    }
}
