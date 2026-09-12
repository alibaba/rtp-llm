package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.JsonNode;
import org.flexlb.engine.grpc.EngineRpcService;

/** Test-only stopping-length model. Does not alter the request seen by the master. */
final class MockEosModel {
    private final boolean enabled;
    private final long seed;
    private final double meanTokens;

    private MockEosModel(boolean enabled, long seed, double meanTokens) {
        this.enabled = enabled;
        this.seed = seed;
        this.meanTokens = meanTokens;
    }

    static MockEosModel load(JsonNode config) {
        if (!config.path("enabled").asBoolean(false)) return new MockEosModel(false, 0, 0);
        if (!config.path("distribution").asText().equals("geometric")) {
            throw new IllegalArgumentException("decode.eos.distribution must be geometric");
        }
        double mean = config.path("mean_tokens").asDouble(Double.NaN);
        if (!Double.isFinite(mean) || mean < 1) {
            throw new IllegalArgumentException("decode.eos.mean_tokens must be finite and >= 1");
        }
        if (!config.path("seed").isIntegralNumber()) {
            throw new IllegalArgumentException("decode.eos.seed must be an explicit integer");
        }
        return new MockEosModel(true, config.path("seed").asLong(), mean);
    }

    int outputLength(EngineRpcService.GenerateInputPB input, int legacyLength, boolean explicitLength) {
        if (!enabled) return legacyLength;
        var config = input.getGenerateConfig();
        int cap = Math.max(1, config.getMaxNewTokens());
        if (config.getIgnoreEos()) return cap;
        int minimum = Math.min(cap, Math.max(1, config.getMinNewTokens()));
        // An explicit replay length is observed data; do not resample it.
        if (explicitLength) return Math.min(cap, Math.max(minimum, legacyLength));
        // SplitMix64: stable per request/seed across engines, retries and JVMs.
        long z = input.getRequestId() + seed + 0x9e3779b97f4a7c15L;
        z = (z ^ (z >>> 30)) * 0xbf58476d1ce4e5b9L;
        z = (z ^ (z >>> 27)) * 0x94d049bb133111ebL;
        z ^= z >>> 31;
        double u = (z >>> 11) * 0x1.0p-53;
        // EOS has constant hazard 1/mean once minimum length is reached.
        // Sampling once is equivalent to per-token Bernoulli draws, without
        // depending on timing, batching or MTP step width.
        double extra = meanTokens == 1 ? 0 : Math.floor(Math.log1p(-u) / Math.log1p(-1 / meanTokens));
        return (int) Math.min(cap, minimum + extra);
    }
}
