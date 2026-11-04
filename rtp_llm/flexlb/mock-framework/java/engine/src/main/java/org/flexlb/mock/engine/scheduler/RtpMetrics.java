package org.flexlb.mock.engine.scheduler;

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
    record RequestStats(long elapsedNanos, long waitNanos, long enqueueToCanRunNanos,
                        long canRunToRunningNanos,
                        long firstTokenNanos, int inputTokens, int outputTokens, int iterations, int reusedTokens) {}

    record StepStats(int context, int generate, long maxSequence, long scheduleToCompletionNanos, long stepNanos,
                     long forwardNanos, long plannedNanos, long input, long computed, long output,
                     int outputStreams) {}

    public record State(long waiting, long running, long loading, long ready, long totalTokens, long availableTokens,
                        long freeTokens, long heldTokens, int blockSize) {
        public State {
            if (blockSize <= 0 || waiting < 0 || running < 0 || loading < 0 || ready < 0 || totalTokens < 0 || availableTokens < 0 || freeTokens < 0
                    || heldTokens < 0 || availableTokens > totalTokens || freeTokens > availableTokens || heldTokens > totalTokens)
                throw new IllegalArgumentException("Invalid engine state");
            if (totalTokens % blockSize != 0 || availableTokens % blockSize != 0 || freeTokens % blockSize != 0)
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
    private long windowStarted, windowInput, windowComputed, windowTpsTokens, windowTpsNanos, windowComputeNanos, windowInputNanos;
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
        actualForward = Gauge.build().name("mock_model_forward_actual_us").help("Latest completed simulated Forward duration in microseconds.")
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
                var result = new ArrayList<MetricFamilySamples>();
                result.addAll(List.of(tasks, kv));
                result.add(gauge("rtp_llm_wait_stream_size", "Waiting scheduler requests.", role, value.waiting()));
                result.add(gauge("rtp_llm_running_stream_size", "Running scheduler requests.", role, value.running()));
                result.add(gauge("rtp_llm_kv_cache_left_seq", "Available Device KV token capacity.", role, value.availableTokens()));
                for (String name : List.of("total", "free", "available", "used", "active")) {
                    var blocks = new GaugeMetricFamily("rtp_llm_kv_cache_" + name + "_blocks",
                        "KV pool capacity in blocks.", List.of("role", "pool_name", "tier"));
                    long device = switch (name) {
                        case "total" -> value.totalTokens(); case "free" -> value.freeTokens();
                        case "available" -> value.availableTokens(); case "active" -> value.totalTokens() - value.availableTokens(); default -> value.totalTokens() - value.freeTokens();
                    };
                    blocks.addMetric(List.of(role, "device", "DEVICE"), device / value.blockSize());
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
                        boolean idle = windowTpsNanos == 0 && (!prefill || windowInputNanos == 0);
                        if (prefill) {
                            if (idle || windowComputeNanos > 0)
                                result.add(gauge("rtp_llm_context_tps", "Computed Prefill tokens per active execution second.", role, tps(windowComputed, windowComputeNanos)));
                            if (idle || windowInputNanos > 0)
                                result.add(gauge("rtp_llm_context_tps_with_cache", "Input Prefill tokens per active execution second.", role, tps(windowInput, windowInputNanos)));
                            result.add(gauge("rtp_llm_context_wall_tps", "Computed Prefill tokens per wall second.", role, windowComputed / seconds));
                            result.add(gauge("rtp_llm_context_wall_tps_with_cache", "Input Prefill tokens per wall second.", role, windowInput / seconds));
                            result.add(gauge("rtp_llm_wall_tps_report_interval_us", "Actual wall-clock report window, microseconds.", role, elapsed / 1000.0));
                        } else {
                            result.add(gauge("rtp_llm_generate_tps", "Decode tokens per active execution second.", role, tps(windowTpsTokens, windowTpsNanos)));
                        }
                        if (idle || windowTpsNanos > 0)
                            result.add(gauge("rtp_llm_total_tps", "Executor tokens per active execution second.", role, tps(windowTpsTokens, windowTpsNanos)));
                        double done = finished.values().stream().mapToDouble(Counter.Child::get).sum();
                        double errors = finished.get(Outcome.error).get() + finished.get(Outcome.timeout).get();
                        double cancelled = finished.get(Outcome.cancelled).get();
                        result.add(gauge("rtp_llm_framework_qps", "Terminal requests per wall second, including failures.", role, (done - lastFinished) / seconds));
                        result.add(gauge("rtp_llm_framework_error_qps", "Terminal errors and timeouts per wall second.", role, (errors - lastErrors) / seconds));
                        result.add(gauge("rtp_llm_cancel_qps", "Terminal cancellations per wall second.", role, (cancelled - lastCancelled) / seconds));
                        lastFinished = done; lastErrors = errors; lastCancelled = cancelled;
                        windowStarted = now;
                        windowInput = windowComputed = windowTpsTokens = windowTpsNanos = windowComputeNanos = windowInputNanos = 0;
                    }
                }
                return result;
            }
        }.register(registry);
    }

    private static double tps(long tokens, long nanos) {
        return nanos == 0 ? 0 : tokens * 1_000_000_000.0 / nanos;
    }

    private static GaugeMetricFamily gauge(String name, String help, String role, double value) {
        var metric = new GaugeMetricFamily(name, help, List.of("role"));
        metric.addMetric(List.of(role), value);
        return metric;
    }

    private Counter counter(String name, String help) {
        return Counter.build().name(name).help(help).labelNames(LABELS).register(registry);
    }
    public void started() { started.inc(); }

    void requestFinished(Outcome outcome, RequestStats stats) {
        // Real emits stream timings for success, cancellation and timeout, not arbitrary internal errors.
        if (outcome != Outcome.error) {
            events.observe("rtp_llm_latency_us", stats.elapsedNanos() / 1000.0);
            events.observe("rtp_llm_wait_latency_us", stats.waitNanos() / 1000.0);
            positiveMicros("rtp_llm_stream_enqueue_to_canrun_us", stats.enqueueToCanRunNanos());
            positiveMicros("rtp_llm_stream_canrun_to_running_us", stats.canRunToRunningNanos());
            if (stats.firstTokenNanos() >= 0) events.observe("rtp_llm_first_token_latency_us", stats.firstTokenNanos() / 1000.0);
            events.observe("rtp_llm_input_token_length", stats.inputTokens());
            events.observe("rtp_llm_output_token_length", stats.outputTokens());
            events.observe("rtp_llm_iterate_count", stats.iterations());
            events.observe("rtp_llm_reuse_length", stats.reusedTokens());
            events.observe("rtp_llm_query_batch_size", 1); // Native query batch is beams, not the scheduler batch.
        }
        finished.get(outcome).inc();
    }

    // Native stream REPORT_GAUGE omits zero samples; allocator event gauges do not.
    private void positiveMicros(String name, long nanos) { if (nanos >= 1000) events.observe(name, nanos / 1000); }

    public void cacheLookup(long input, long reused) {
        if (input < 0 || reused < 0 || reused > input) throw new IllegalArgumentException("Invalid cache observation");
        cacheInputs.inc(input); cacheHits.inc(reused);
    }
    void prepared(long deviceInputTokens, long reused, long elapsedNanos) {
        events.observe("rtp_llm_device_input_length", deviceInputTokens);
        events.observe("rtp_llm_kv_cache_reuse_length", reused);
        events.observe("rtp_llm_stream_cache_device_reuse_length", reused);
        events.observe("rtp_llm_kv_cache_match_to_ready_latency_us", elapsedNanos / 1000.0,
            Map.of("load_attempted", "false"));
    }

    private void recordBatch(int context, int generate, long tokens, long maxSequence) {
        observePositive("rtp_llm_context_batch_size", context);
        observePositive("rtp_llm_generate_batch_size", generate);
        observePositive("rtp_llm_execute_token_size", tokens);
        observePositive("rtp_llm_max_seq_len", maxSequence);
        if (context > 0) {
            observePositive("rtp_llm_context_batch_size_when_has_context", context);
            observePositive("rtp_llm_generate_batch_size_when_has_context", generate);
            observePositive("rtp_llm_execute_token_size_when_has_context", tokens);
            observePositive("rtp_llm_max_seq_len_when_has_context", maxSequence);
        }
    }

    private void observePositive(String name, long value) {
        if (value > 0) events.observe(name, value);
    }

    void stepCompleted(StepStats step) {
        long executionNanos = prefill ? step.scheduleToCompletionNanos() : step.stepNanos();
        if (step.forwardNanos() < 0 || step.plannedNanos() < 0 || step.input() < 0 || step.computed() < 0
            || step.computed() > step.input() || step.output() < 0 || executionNanos < 0 || step.stepNanos() < 0
            || step.outputStreams() < 0 || step.outputStreams() > step.output())
            throw new IllegalArgumentException("Invalid step observation");
        recordBatch(step.context(), step.generate(), step.computed(), step.maxSequence());
        positiveMicros("rtp_llm_step_latency_us", step.stepNanos());
        if (!prefill && step.outputStreams() > 0)
            events.observe("mock_decode_estimate_tpot_us", step.stepNanos() / 1000.0 * step.outputStreams() / step.output());
        recordForward(step, executionNanos, prefill ? step.computed() : step.output());
    }

    private void recordForward(StepStats step, long executionNanos, long tpsTokens) {
        synchronized (observations) {
            if (tpsTokens > 0) {
                windowTpsTokens += tpsTokens;
                windowTpsNanos += executionNanos;
            }
            windowInput += step.input(); windowComputed += step.computed();
            if (step.computed() > 0) {
                windowComputeNanos += executionNanos;
                if (prefill) contextSeconds.inc(step.forwardNanos() / 1_000_000_000.0);
            }
            if (step.input() > 0) {
                windowInputNanos += executionNanos;
                if (prefill) contextWithCacheSeconds.inc(step.forwardNanos() / 1_000_000_000.0);
            }
            tokens.get("input").inc(step.input()); tokens.get("compute").inc(step.computed()); tokens.get("output").inc(step.output());
        }
        forward.observe(step.forwardNanos() / 1_000_000_000.0);
        // Compatibility baseline: production MTP Decode does not emit this collector field.
        if (prefill && step.forwardNanos() > 0) events.observe("rtp_llm_model_forward_us", step.forwardNanos() / 1000.0);
        actualForward.set(step.forwardNanos() / 1000.0);
        plannedForward.set(step.plannedNanos() / 1000.0);
    }
    public List<Collector.MetricFamilySamples> collect() { return MetricTags.collect(registry, tags); }
    public String scrape() { return MetricTags.scrape(registry, tags); }
}
