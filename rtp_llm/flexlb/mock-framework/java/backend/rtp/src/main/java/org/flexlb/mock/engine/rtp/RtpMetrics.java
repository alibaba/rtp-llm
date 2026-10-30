package org.flexlb.mock.engine.rtp;

import org.flexlb.mock.metrics.MetricTags;
import org.flexlb.mock.metrics.WindowMetrics;

import io.prometheus.client.*;
import java.util.*;
import java.util.function.Supplier;
import java.util.function.LongSupplier;

/** RTP metric names and statistics; the scheduler supplies events and state snapshots. */
final class RtpMetrics {
    public static final String CACHE_LOOKUPS = "rtp_llm_cache_lookup_tokens_total";
    public static final String CACHE_HITS = "rtp_llm_cache_hit_tokens_total";
    public static final String FORWARD_TOKENS = "rtp_llm_forward_tokens_total";
    public static final String FORWARD_SECONDS = "rtp_llm_model_forward_seconds";
    public enum Outcome { success, cancelled, timeout, error }
    public record State(long waiting, long running, long loading, long ready, long totalTokens, long availableTokens,
                        long freeTokens, long heldTokens, long hostTotalTokens, long hostFreeTokens, int blockSize) {
        public State {
            if (blockSize <= 0 || waiting < 0 || running < 0 || loading < 0 || ready < 0 || totalTokens < 0 || availableTokens < 0 || freeTokens < 0
                    || heldTokens < 0 || hostTotalTokens < 0 || hostFreeTokens < 0 || hostFreeTokens > hostTotalTokens || availableTokens > totalTokens || freeTokens > availableTokens || heldTokens > totalTokens)
                throw new IllegalArgumentException("Invalid engine state");
            if (totalTokens % blockSize != 0 || availableTokens % blockSize != 0 || freeTokens % blockSize != 0
                    || hostTotalTokens % blockSize != 0 || hostFreeTokens % blockSize != 0)
                throw new IllegalArgumentException("KV capacity must contain whole blocks");
        }
    }
    private static final String[] LABELS = {"role"};
    private final CollectorRegistry registry = new CollectorRegistry();
    private final Map<String, String> tags;
    private final WindowMetrics events;
    private final Counter.Child started, cacheInputs, cacheHits, contextSeconds, contextWithCacheSeconds;
    private final Map<Outcome, Counter.Child> finished = new EnumMap<>(Outcome.class);
    private final Map<String, Counter.Child> tokens = new HashMap<>();
    private final Histogram.Child forward;
    private final Gauge.Child actualForward, plannedForward;

    private final Object observations = new Object();
    private final LongSupplier clock;
    private final boolean prefill;
    private long windowStarted, windowInput, windowComputed, windowOutput, windowComputeNanos, windowInputNanos;
    private double lastFinished, lastErrors, lastCancelled;

    public RtpMetrics(String role, Supplier<State> state) { this(role, state, System::nanoTime); }

    RtpMetrics(String role, Supplier<State> state, LongSupplier clock) {
        if (role == null || role.isBlank()) throw new IllegalArgumentException("Metric role must not be blank");
        Objects.requireNonNull(state);
        this.clock = Objects.requireNonNull(clock);
        prefill = role.equals("prefill");
        windowStarted = clock.getAsLong();
        events = new WindowMetrics(role, clock);
        events.register(registry);
        var physical = new HashMap<>(MetricTags.fromEnvironment(System.getenv()));
        physical.put("dp_rank", "0"); // The current Engine supports exactly one DP worker.
        tags = Map.copyOf(physical);
        String[] identity = {role};
        started = counter("rtp_llm_requests_started_total", "Accepted engine requests.").labels(identity);
        contextSeconds = counter("rtp_llm_context_forward_seconds_total", "Forward seconds with nonzero Prefill computation.").labels(identity);
        contextWithCacheSeconds = counter("rtp_llm_context_forward_with_cache_seconds_total", "Forward seconds with nonzero Prefill input.").labels(identity);
        cacheInputs = counter(CACHE_LOOKUPS, "Input tokens considered by successful KV admission.").labels(identity);
        cacheHits = counter(CACHE_HITS, "Historical prefix tokens reused at KV admission.").labels(identity);
        var results = Counter.build().name("rtp_llm_framework_requests_total").help("Terminal engine requests by outcome.")
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
                var tasks = new GaugeMetricFamily("rtp_llm_scheduler_requests", "Current accepted requests.", List.of("role", "state"));
                tasks.addMetric(List.of(role, "waiting"), value.waiting());
                tasks.addMetric(List.of(role, "running"), value.running());
                tasks.addMetric(List.of(role, "loading"), value.loading());
                tasks.addMetric(List.of(role, "ready"), value.ready());
                var kv = new GaugeMetricFamily("rtp_llm_kv_tokens", "KV capacity in tokens; available includes evictable unreferenced cache.", List.of("role", "state"));
                kv.addMetric(List.of(role, "total"), value.totalTokens());
                kv.addMetric(List.of(role, "available"), value.availableTokens());
                kv.addMetric(List.of(role, "free"), value.freeTokens());
                kv.addMetric(List.of(role, "held"), value.heldTokens());
                var host = new GaugeMetricFamily("rtp_llm_host_kv_tokens", "Local Host KV capacity in tokens.", List.of("role", "state"));
                host.addMetric(List.of(role, "total"), value.hostTotalTokens());
                host.addMetric(List.of(role, "free"), value.hostFreeTokens());
                var result = new ArrayList<MetricFamilySamples>();
                result.addAll(List.of(tasks, kv, host));
                result.add(gauge("rtp_llm_wait_stream_size", "Waiting scheduler requests.", role, value.waiting()));
                result.add(gauge("rtp_llm_running_stream_size", "Running scheduler requests.", role, value.running()));
                result.add(gauge("rtp_llm_kv_cache_left_seq", "Available Device KV token capacity.", role, value.availableTokens()));
                for (String name : List.of("total", "free", "available", "used")) {
                    var blocks = new GaugeMetricFamily("rtp_llm_kv_cache_" + name + "_blocks",
                        "KV pool capacity in blocks.", List.of("role", "pool_name", "tier"));
                    long device = switch (name) {
                        case "total" -> value.totalTokens(); case "free" -> value.freeTokens();
                        case "available" -> value.availableTokens(); default -> value.totalTokens() - value.freeTokens();
                    };
                    blocks.addMetric(List.of(role, "device", "DEVICE"), device / value.blockSize());
                    // Host available includes evictable entries; free alone cannot represent it.
                    if (!name.equals("available")) {
                        long hostValue = name.equals("total") ? value.hostTotalTokens() : name.equals("free")
                            ? value.hostFreeTokens() : value.hostTotalTokens() - value.hostFreeTokens();
                        blocks.addMetric(List.of(role, "host", "HOST"), hostValue / value.blockSize());
                    }
                    result.add(blocks);
                }
                var used = new GaugeMetricFamily("rtp_llm_kv_cache_used_ratio", "Unavailable Device KV capacity, percent.",
                    List.of("role", "pool_name", "tier"));
                used.addMetric(List.of(role, "device", "DEVICE"), value.totalTokens() == 0 ? 0
                    : 100.0 * (value.totalTokens() - value.availableTokens()) / value.totalTokens());
                result.add(used);
                synchronized (observations) {
                    long now = clock.getAsLong(), elapsed = now - windowStarted;
                    if (elapsed > 0) {
                        double seconds = elapsed / 1_000_000_000.0;
                        if (role.equals("prefill")) {
                            result.add(gauge("rtp_llm_context_tokens", "Computed Prefill tokens in the scrape window.", role, windowComputed));
                            result.add(gauge("rtp_llm_context_tokens_with_cache", "Input Prefill tokens in the scrape window.", role, windowInput));
                            result.add(gauge("rtp_llm_context_tps", "Computed Prefill tokens per active Forward second.", role,
                                windowComputeNanos == 0 ? 0 : windowComputed * 1_000_000_000.0 / windowComputeNanos));
                            result.add(gauge("rtp_llm_context_tps_with_cache", "Input Prefill tokens per active Forward second.", role,
                                windowInputNanos == 0 ? 0 : windowInput * 1_000_000_000.0 / windowInputNanos));
                            result.add(gauge("rtp_llm_context_wall_tps", "Computed Prefill tokens per wall second.", role, windowComputed / seconds));
                            result.add(gauge("rtp_llm_context_wall_tps_with_cache", "Input Prefill tokens per wall second.", role, windowInput / seconds));
                            result.add(gauge("rtp_llm_wall_tps_report_interval_us", "Actual wall-clock report window, microseconds.", role, elapsed / 1000.0));
                        } else if (role.equals("decode")) {
                            result.add(gauge("rtp_llm_generate_tps", "Committed Decode output tokens per wall second.", role, windowOutput / seconds));
                        }
                        result.add(gauge("rtp_llm_total_tps", "Model-computed tokens per wall second.", role, windowComputed / seconds));
                        double done = finished.values().stream().mapToDouble(Counter.Child::get).sum();
                        double errors = finished.get(Outcome.error).get() + finished.get(Outcome.timeout).get();
                        double cancelled = finished.get(Outcome.cancelled).get();
                        result.add(gauge("rtp_llm_framework_qps", "Terminal requests per wall second, including failures.", role, (done - lastFinished) / seconds));
                        result.add(gauge("rtp_llm_framework_error_qps", "Terminal errors and timeouts per wall second.", role, (errors - lastErrors) / seconds));
                        result.add(gauge("rtp_llm_cancel_qps", "Terminal cancellations per wall second.", role, (cancelled - lastCancelled) / seconds));
                        lastFinished = done; lastErrors = errors; lastCancelled = cancelled;
                        windowStarted = now;
                        windowInput = windowComputed = windowOutput = windowComputeNanos = windowInputNanos = 0;
                    }
                }
                return result;
            }
        }.register(registry);
    }

    private static GaugeMetricFamily gauge(String name, String help, String role, double value) {
        var metric = new GaugeMetricFamily(name, help, List.of("role"));
        metric.addMetric(List.of(role), value);
        return metric;
    }

    private Counter counter(String name, String help) {
        return Counter.build().name(name).help(help).labelNames(LABELS).register(registry);
    }
    public void observe(String name, double value) { events.observe(name, value); }
    public void started() { started.inc(); }
    public void finished(Outcome outcome) { finished.get(Objects.requireNonNull(outcome)).inc(); }
    public void cacheLookup(long input, long reused) {
        if (input < 0 || reused < 0 || reused > input) throw new IllegalArgumentException("Invalid cache observation");
        cacheInputs.inc(input); cacheHits.inc(reused);
        events.observe("rtp_llm_kv_cache_reuse_length", reused);
    }
    public void forwardCompleted(long elapsedNanos, long plannedNanos, long input, long computed, long output) {
        if (elapsedNanos < 0 || plannedNanos < 0 || input < 0 || computed < 0 || computed > input || output < 0)
            throw new IllegalArgumentException("Invalid forward observation");
        synchronized (observations) {
            windowInput += input; windowComputed += computed; windowOutput += output;
            if (computed > 0) {
                windowComputeNanos += elapsedNanos;
                if (prefill) contextSeconds.inc(elapsedNanos / 1_000_000_000.0);
            }
            if (input > 0) {
                windowInputNanos += elapsedNanos;
                if (prefill) contextWithCacheSeconds.inc(elapsedNanos / 1_000_000_000.0);
            }
            tokens.get("input").inc(input); tokens.get("compute").inc(computed); tokens.get("output").inc(output);
        }
        forward.observe(elapsedNanos / 1_000_000_000.0);
        actualForward.set(elapsedNanos / 1000.0);
        plannedForward.set(plannedNanos / 1000.0);
    }
    public String scrape() { return MetricTags.scrape(registry, tags); }
}
