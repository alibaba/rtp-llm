package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import java.util.Map;
import java.util.SplittableRandom;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/** Mock time, independent of real scheduler/deadline configuration. One draw per forward/RPC leg. */
final class MockTiming {
    static final String[] RPC_METHODS = {"get_worker_status", "get_cache_status", "enqueue_batch", "cancel",
            "fetch_response", "generate_stream_call", "remote_generate", "check_health",
            "remote_multimodal_embedding", "get_multimodal_hashes", "async_submit_embedding",
            "wait_green_net_verdict", "release_rdma_lease"};
    record Sample(double baseMs, double jitterMs, double pauseMs, long nanos) {
        double milliseconds() { return nanos / 1e6; }
    }
    record Spec(double delayMs, double stddevMs, double stddevRatio, double maxDeviationMs,
                double probability, double durationMs) {
        static Spec read(JsonNode node, boolean rpc) {
            if (rpc && !node.isMissingNode()) MockEngineConfig.fields(node, "delay_ms", "jitter", "pause");
            var jitter = node.path("jitter"); var pause = node.path("pause");
            if (!jitter.isMissingNode()) MockEngineConfig.fields(jitter, "stddev_ms", "stddev_ratio", "max_deviation_ms");
            if (!pause.isMissingNode()) MockEngineConfig.fields(pause, "probability", "duration_ms");
            double std = number(jitter, "stddev_ms", 0), ratio = number(jitter, "stddev_ratio", 0);
            double cap = number(jitter, "max_deviation_ms", 0), probability = number(pause, "probability", 0);
            double duration = number(pause, "duration_ms", 0);
            if (probability > 1 || ((std > 0 || ratio > 0) && cap == 0) || (probability > 0 && duration == 0))
                throw new IllegalArgumentException("jitter requires a bound; pause requires probability in [0,1] and duration_ms");
            return new Spec(rpc ? number(node, "delay_ms", 0) : 0, std, ratio, cap, probability, duration);
        }
    }
    private volatile MockTiming replacement;
    void configure(JsonNode mock) { replacement = new MockTiming(mock, engine); }
    private final JsonNode config;
    private final long seed;
    private final double scale;
    private final String engine;
    private final Map<String, Spec> specs = new ConcurrentHashMap<>();
    private final Map<String, AtomicLong> sequences = new ConcurrentHashMap<>();
    MockTiming(JsonNode mock) { this(mock, ""); }
    private MockTiming(JsonNode mock, String engine) {
        config = mock.deepCopy(); this.engine = engine;
        var timing = mock.path("timing");
        if (!timing.isMissingNode()) MockEngineConfig.fields(timing, "seed", "time_scale");
        if (timing.has("seed") && (!timing.get("seed").isIntegralNumber() || !timing.get("seed").canConvertToLong()))
            throw new IllegalArgumentException("mock.timing.seed must be int64");
        seed = timing.path("seed").asLong(0); scale = number(timing, "time_scale", 1);
        if (scale <= 0) throw new IllegalArgumentException("mock.timing.time_scale must be positive");
        for (String role : new String[]{"prefill", "decode", "vit"})
            specs.put("forward." + role, Spec.read(mock.path("forward").path(role), false));
        var rpc = mock.path("rpc");
        if (!rpc.isMissingNode()) {
            MockEngineConfig.fields(rpc, RPC_METHODS);
            rpc.fields().forEachRemaining(entry -> {
                MockEngineConfig.fields(entry.getValue(), "request", "response");
                for (String leg : new String[]{"request", "response"})
                    specs.put("rpc." + entry.getKey() + "." + leg, Spec.read(entry.getValue().path(leg), true));
            });
        }
    }
    static MockTiming defaults() { return new MockTiming(new ObjectMapper().createObjectNode()); }
    MockTiming forEngine(String id) { return replacement == null ? new MockTiming(config, id) : replacement.forEngine(id); }
    private static double number(JsonNode node, String name, double fallback) {
        if (!node.has(name)) return fallback;
        var value = node.get(name); double result = value.asDouble(Double.NaN);
        if (!value.isNumber() || !Double.isFinite(result) || result < 0 || result > 86_400_000)
            throw new IllegalArgumentException("invalid timing field " + name);
        return result;
    }
    private static long hash(String value) {
        long hash = 0xcbf29ce484222325L;
        for (int i = 0; i < value.length(); i++) { hash ^= value.charAt(i); hash *= 0x100000001b3L; }
        return hash;
    }
    boolean rpcEnabled(String method) {
        if (replacement != null) return replacement.rpcEnabled(method);
        for (String leg : new String[]{"request", "response"}) {
            var value = specs.get("rpc." + method + "." + leg);
            if (value != null && (value.delayMs > 0 || value.stddevMs > 0 || value.stddevRatio > 0 || value.probability > 0)) return true;
        }
        return false;
    }
    double scaledMilliseconds(double value) {
        return replacement == null ? value * scale : replacement.scaledMilliseconds(value);
    }
    Sample sample(String path, double baseMs) {
        if (replacement != null) return replacement.sample(path, baseMs);
        var spec = specs.getOrDefault(path, new Spec(0,0,0,0,0,0));
        long sequence = sequences.computeIfAbsent(path, ignored -> new AtomicLong()).getAndIncrement();
        var random = new SplittableRandom(seed ^ hash(engine + "/" + path) ^ Long.rotateLeft(sequence * 0x9e3779b97f4a7c15L, 23));
        double base = baseMs + spec.delayMs;
        double sigma = Math.hypot(spec.stddevMs, base * spec.stddevRatio);
        double z = Math.sqrt(-2 * Math.log(Math.max(1e-16, random.nextDouble()))) * Math.cos(2 * Math.PI * random.nextDouble());
        double jitter = Math.max(-spec.maxDeviationMs, Math.min(spec.maxDeviationMs, z * sigma));
        double pause = random.nextDouble() < spec.probability ? spec.durationMs : 0;
        double nanos = Math.max(0, base + jitter + pause) * scale * 1e6;
        if (!Double.isFinite(nanos) || nanos > Long.MAX_VALUE / 4.0)
            throw new IllegalArgumentException("timing duration overflow");
        return new Sample(base * scale, jitter * scale, pause * scale, Math.round(nanos));
    }
    Map<String, Number> metrics(String path, Sample sample, long elapsedNanos) {
        String prefix = "mock_timing_" + path.replace('.', '_') + "_";
        return Map.of(prefix + "base_us", sample.baseMs * 1000,
                prefix + "jitter_us", sample.jitterMs * 1000,
                prefix + "pause_us", sample.pauseMs * 1000,
                prefix + "planned_us", sample.nanos / 1000.0,
                prefix + "elapsed_us", elapsedNanos / 1000.0,
                prefix + "timer_lag_us", Math.max(0, elapsedNanos - sample.nanos) / 1000.0);
    }
}
