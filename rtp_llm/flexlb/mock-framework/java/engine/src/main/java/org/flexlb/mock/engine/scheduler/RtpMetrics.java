package org.flexlb.mock.engine.scheduler;

import org.flexlb.mock.metrics.MetricTags;
import org.flexlb.mock.metrics.EventMetrics;

import io.prometheus.client.*;
import java.util.*;
import java.util.concurrent.TimeUnit;
import java.util.function.Supplier;
import java.util.function.LongSupplier;

/**
 * Defines RTP metric names and statistics from scheduler events and state.
 */
final class RtpMetrics {
    public static final String CACHE_LOOKUPS = "rtp_llm_cache_lookup_tokens_total";
    public static final String CACHE_HITS = "rtp_llm_cache_hit_tokens_total";
    public static final String FORWARD_TOKENS = "rtp_llm_forward_tokens_total";
    public static final String FORWARD_SECONDS = "rtp_llm_model_forward_seconds";
    private static final String[] LABELS = {"role"};

    private static final long RECENT_CACHE_WINDOW_NANOS = TimeUnit.MINUTES.toNanos(30);

    private final CollectorRegistry registry = new CollectorRegistry();
    private final Map<String, String> tags;
    private final EventMetrics events;
    private final Counter.Child started, cacheInputs, cacheHits, contextSeconds, contextWithCacheSeconds;
    private final Map<Outcome, Counter.Child> finished = new EnumMap<>(Outcome.class);
    private final Map<String, Counter.Child> tokens = new HashMap<>();
    private final Histogram.Child forward;
    private final Gauge.Child actualForward, plannedForward;
    private final Object observations = new Object();
    private final LongSupplier clock;

    private final boolean prefill;

    private final LinkedHashMap<Long, Long> recentCacheKeys = new LinkedHashMap<>();
    private final Map<Integer, TokenCounts> windowTokens = new TreeMap<>();

    private long windowStarted, windowComputeNanos, windowInputNanos;
    private boolean computing;
    private boolean schedulingActive;

    public RtpMetrics(String role, Supplier<State> state) { this(role, state, System::nanoTime); }

    RtpMetrics(String role, Supplier<State> state, LongSupplier clock) {
        this(role, state, clock, () -> 0);
    }

    RtpMetrics(String role, Supplier<State> state, LongSupplier clock, java.util.function.IntSupplier cacheItems) {
        if (role == null || role.isBlank()) throw new IllegalArgumentException("Metric role must not be blank");
        Objects.requireNonNull(state);
        this.clock = Objects.requireNonNull(clock);
        prefill = role.equals("prefill");
        windowStarted = clock.getAsLong();
        var physical = new HashMap<>(MetricTags.fromEnvironment(System.getenv()));
        physical.remove("priority"); // Request priority belongs to each TPS bucket, not the process.
        physical.put("dp_rank", "0"); // The current Engine supports exactly one DP worker.
        tags = Map.copyOf(physical);
        events = new EventMetrics(role, tags);
        events.register(registry);
        String[] identity = {role};
        started = counter("rtp_llm_requests_started_total", "Accepted engine requests.").labels(identity);
        contextSeconds = counter("rtp_llm_context_forward_seconds_total", "Execution seconds with nonzero Prefill computation, matching context TPS.").labels(identity);
        contextWithCacheSeconds = counter("rtp_llm_context_forward_with_cache_seconds_total", "Execution seconds with nonzero Prefill input, matching context TPS with cache.").labels(identity);
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
        for (String name : List.of("rtp_llm_framework_qps", "rtp_llm_framework_error_qps", "rtp_llm_cancel_qps"))
            events.add(name, 0);
        // Operational gauges describe live state. Native scheduler metrics are round observations.
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
                result.add(gauge("rtp_llm_kv_cache_item_num", "Indexed KV blocks.", role, cacheItems.getAsInt()));
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
                    collectTps(result, role, clock.getAsLong());
                }
                return result;
            }
        }.register(registry);
    }

    public void started() { started.inc(); }

    public void cacheLookup(long input, long reused) {
        if (input < 0 || reused < 0 || reused > input) throw new IllegalArgumentException("Invalid cache observation");
        cacheInputs.inc(input); cacheHits.inc(reused);
    }

    public List<Collector.MetricFamilySamples> collect() { return MetricTags.collect(registry, tags); }

    public String scrape() { return MetricTags.scrape(registry, tags); }

    void schedulerRound(int waiting, int running) {
        boolean active = waiting > 0 || running > 0;
        if (active || schedulingActive) {
            events.observe("rtp_llm_wait_stream_size", waiting);
            events.observe("rtp_llm_running_stream_size", running);
            events.observe("rtp_llm_loading_cache_stream_size", 0); // KV prepare is synchronous.
        }
        schedulingActive = active;
    }

    void computeStarted() { synchronized (observations) { computing = true; } }

    void computeFinished() { synchronized (observations) { computing = false; } }

    void requestFinished(Outcome outcome, RequestStats stats, boolean reportCacheReuse) {
        // Real emits stream timings for success, cancellation and timeout, not arbitrary internal errors.
        if (outcome != Outcome.error) {
            positiveMicros("rtp_llm_latency_us", stats.elapsedNanos());
            positiveMicros("rtp_llm_wait_latency_us", stats.waitNanos());
            positiveMicros("rtp_llm_stream_enqueue_to_canrun_us", stats.enqueueToCanRunNanos());
            positiveMicros("rtp_llm_stream_canrun_to_running_us", stats.canRunToRunningNanos());
            positiveMicros("rtp_llm_first_token_latency_us", stats.firstTokenNanos());
            observePositive("rtp_llm_input_token_length", stats.inputTokens());
            if (reportCacheReuse) events.observe("rtp_llm_kv_cache_reuse_length", stats.reusedTokens());
            observePositive("rtp_llm_output_token_length", stats.outputTokens());
            observePositive("rtp_llm_iterate_count", stats.iterations());
            observePositive("rtp_llm_reuse_length", stats.reusedTokens());
            events.observe("rtp_llm_query_batch_size", 1); // Native query batch is beams, not the scheduler batch.
        }
        finished.get(outcome).inc();
        events.increment("rtp_llm_framework_qps");
        if (outcome == Outcome.cancelled) events.increment("rtp_llm_cancel_qps");
        if (outcome == Outcome.error || outcome == Outcome.timeout) events.increment("rtp_llm_framework_error_qps");
    }

    void recordPrefillCacheKeys(List<Long> keys, int inputTokens, int blockSize) {
        long now = clock.getAsLong();
        while (!recentCacheKeys.isEmpty()
                && now - recentCacheKeys.firstEntry().getValue() >= RECENT_CACHE_WINDOW_NANOS) {
            recentCacheKeys.pollFirstEntry();
        }
        long hitTokens = 0;
        for (long key : keys) {
            if (recentCacheKeys.containsKey(key)) hitTokens += blockSize;
        }
        // Count against prior requests before refreshing the last occurrence of each key.
        for (long key : keys) recentCacheKeys.putLast(key, now);
        events.observe("rtp_llm_prefill_worker_recent_cache_key_hit_count", Math.min(hitTokens, inputTokens));
        events.observe("rtp_llm_prefill_worker_recent_cache_key_total_count", inputTokens);
    }

    void cacheMatched(long nanos) { events.observe("rtp_llm_match_cost_time_us", nanos / 1000.0); }

    void blockEvicted(long lifetimeNanos, long idleNanos) {
        var tags = Map.of("tier", "DEVICE", "trigger_type", "capacity", "group_type", "full");
        events.observe("rtp_llm_kv_cache_evicted_block_lifetime_ms", lifetimeNanos / 1_000_000.0, tags);
        events.observe("rtp_llm_kv_cache_evicted_block_idletime_ms", idleNanos / 1_000_000.0, tags);
    }

    void batchPrepared(long nanos) { positiveMicros("rtp_llm_gather_model_input_us", nanos); }

    void streamingFinished(boolean streaming) {
        events.increment(streaming ? "rtp_llm_is_streaming_qps" : "rtp_llm_not_streaming_qps");
    }

    void prepared(long deviceInputTokens, long reused, long elapsedNanos) {
        events.observe("rtp_llm_device_input_length", deviceInputTokens);
        events.observe("rtp_llm_stream_cache_device_reuse_length", reused);
        events.observe("rtp_llm_kv_cache_match_to_ready_latency_us", elapsedNanos / 1000.0,
            Map.of("load_attempted", "false"));
    }

    void stepCompleted(StepStats step) {
        stepCompleted(step, Map.of(0, new TokenCounts(prefill ? step.computed() : 0, prefill ? step.input() : 0,
            prefill ? 0 : step.acceptedTokens(), prefill ? step.computed() : step.acceptedTokens())));
    }

    void stepCompleted(StepStats step, Map<Integer, TokenCounts> byPriority) {
        long executionNanos = prefill ? step.scheduleToCompletionNanos() : step.stepNanos();
        if (step.forwardNanos() < 0 || step.plannedNanos() < 0 || step.input() < 0 || step.computed() < 0
            || step.computed() > step.input() || step.output() < 0 || executionNanos < 0 || step.stepNanos() < 0
            || step.acceptedTokens() < 0 || step.tpotStepNanos() < 0)
            throw new IllegalArgumentException("Invalid step observation");
        recordBatch(step.context(), step.generate(), step.computed(), step.maxSequence());
        positiveMicros("rtp_llm_step_latency_us", step.stepNanos());
        if (!prefill && step.generate() > 0 && step.acceptedTokens() > 0)
            events.observe("rtp_llm_sp_estimate_tpot_us", step.tpotStepNanos() / 1000.0 * step.generate() / step.acceptedTokens());
        recordForward(step, executionNanos, byPriority);
    }

    private void collectTps(List<Collector.MetricFamilySamples> result, String role, long now) {
        long elapsed = now - windowStarted;
        // Like Real's ActiveGuard: do not reset a long in-flight forward's wall window.
        if (elapsed <= 0 || windowTokens.isEmpty() && computing) return;
        boolean idle = windowTokens.isEmpty();
        var buckets = idle ? Map.of(0, new TokenCounts(0, 0, 0, 0)) : windowTokens;
        var tpsMetrics = new LinkedHashMap<String, GaugeMetricFamily>();
        for (var bucket : buckets.entrySet()) {
            int priority = bucket.getKey();
            var tokens = bucket.getValue();
            if (prefill) {
                if (idle || windowComputeNanos > 0) {
                    addTps(tpsMetrics, "context_tokens", "Computed Prefill tokens in the report window.", role, priority, tokens.context());
                    addTps(tpsMetrics, "context_tps", "Computed Prefill tokens per active execution second.", role, priority, tps(tokens.context(), windowComputeNanos));
                }
                if (idle || windowInputNanos > 0) {
                    addTps(tpsMetrics, "context_tokens_with_cache", "Input Prefill tokens in the report window.", role, priority, tokens.withCache());
                    addTps(tpsMetrics, "context_tps_with_cache", "Input Prefill tokens per active execution second.", role, priority, tps(tokens.withCache(), windowInputNanos));
                }
                addTps(tpsMetrics, "context_wall_tps", "Computed Prefill tokens per wall second.", role, priority, tps(tokens.context(), elapsed));
                addTps(tpsMetrics, "context_wall_tps_with_cache", "Input Prefill tokens per wall second.", role, priority, tps(tokens.withCache(), elapsed));
            } else if (idle || tokens.generate() > 0) {
                addTps(tpsMetrics, "generate_tps", "Accepted Decode tokens in the report window (native RTP contract).", role, priority, tokens.generate());
            }
            if (idle || tokens.total() > 0)
                addTps(tpsMetrics, "total_tps", "Executor tokens in the report window (native RTP contract).", role, priority, tokens.total());
        }
        result.addAll(tpsMetrics.values());
        // Window duration is process-wide, not additive across priorities.
        result.add(gauge("rtp_llm_wall_tps_report_interval_us", "Actual TPS report window, microseconds.", role, elapsed / 1000.0));
        windowTokens.clear(); windowComputeNanos = windowInputNanos = 0; windowStarted = now;
    }

    private static void addTps(Map<String, GaugeMetricFamily> metrics, String suffix, String help,
                               String role, int priority, double value) {
        metrics.computeIfAbsent(suffix, ignored -> new GaugeMetricFamily("rtp_llm_" + suffix, help, List.of("role", "priority")))
            .addMetric(List.of(role, Integer.toString(priority)), value);
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

    // Native stream REPORT_GAUGE omits zero samples; allocator event gauges do not.
    private void positiveMicros(String name, long nanos) { if (nanos >= 1000) events.observe(name, nanos / 1000); }

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

    private void recordForward(StepStats step, long executionNanos, Map<Integer, TokenCounts> byPriority) {
        synchronized (observations) {
            for (var entry : byPriority.entrySet()) {
                var counts = entry.getValue();
                var timed = new TokenCounts(executionNanos > 0 ? counts.context() : 0,
                    executionNanos > 0 ? counts.withCache() : 0, counts.generate(), counts.total());
                if (timed.context() > 0 || timed.withCache() > 0 || timed.generate() > 0 || timed.total() > 0)
                    windowTokens.merge(entry.getKey(), timed, TokenCounts::plus);
            }
            if (step.computed() > 0 && executionNanos > 0) {
                windowComputeNanos += executionNanos;
                if (prefill) contextSeconds.inc(executionNanos / 1_000_000_000.0);
            }
            if (step.input() > 0 && executionNanos > 0) {
                windowInputNanos += executionNanos;
                if (prefill) contextWithCacheSeconds.inc(executionNanos / 1_000_000_000.0);
            }
            tokens.get("input").inc(step.input()); tokens.get("compute").inc(step.computed()); tokens.get("output").inc(step.output());
        }
        forward.observe(step.forwardNanos() / 1_000_000_000.0);
        // Compatibility baseline: production MTP Decode does not emit this collector field.
        if (prefill && step.forwardNanos() > 0) events.observe("rtp_llm_model_forward_us", step.forwardNanos() / 1000.0);
        actualForward.set(step.forwardNanos() / 1000.0);
        plannedForward.set(step.plannedNanos() / 1000.0);
    }

    public enum Outcome { success, cancelled, timeout, error }

    record RequestStats(long elapsedNanos, long waitNanos, long enqueueToCanRunNanos,
                        long canRunToRunningNanos,
                        long firstTokenNanos, int inputTokens, int outputTokens, int iterations, int reusedTokens) {}

    record StepStats(int context, int generate, long maxSequence, long scheduleToCompletionNanos, long stepNanos,
                     long forwardNanos, long plannedNanos, long input, long computed, long output,
                     long acceptedTokens, long tpotStepNanos) {}

    /**
     * Captures live scheduler counts and logical KV capacity.
     */
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

    /**
     * Carries completed Forward token counts that can be summed across rounds.
     */
    record TokenCounts(long context, long withCache, long generate, long total) {
        TokenCounts plus(TokenCounts other) {
            return new TokenCounts(context + other.context, withCache + other.withCache,
                generate + other.generate, total + other.total);
        }
    }
}
