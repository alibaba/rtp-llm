package org.flexlb.mock.engine.forward;

import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.security.*;
import java.time.Duration;
import java.util.SplittableRandom;

/**
 * Samples timing noise from the invocation key so results are independent of execution order.
 */
public final class NoiseModel {
    private final Config spec;

    NoiseModel(Config spec) { this.spec = java.util.Objects.requireNonNull(spec); }

    public Sample apply(Duration base, String invocationKey) {
        if (base.isNegative()) throw new IllegalArgumentException("Negative base duration");
        double mean = base.toNanos() / 1e6;
        double sigma = Math.hypot(spec.stddev_ms(), mean * spec.stddev_ratio());
        double jitter = 0;
        if (sigma > 0) {
            var random = new SplittableRandom(keyedSeed(spec.seed(), invocationKey));
            double gaussian = Math.sqrt(-2 * Math.log(Math.max(1e-16, random.nextDouble())))
                * Math.cos(2 * Math.PI * random.nextDouble());
            jitter = Math.max(-spec.max_deviation_ms(), Math.min(spec.max_deviation_ms(), gaussian * sigma));
        }
        var planned = FormulaForward.milliseconds(Math.max(0, mean + jitter));
        return new Sample(base, jitter, planned);
    }

    static long keyedSeed(long seed, String key) {
        if (key == null || key.isBlank()) throw new IllegalArgumentException("Invocation key is required");
        try {
            var hash = MessageDigest.getInstance("SHA-256");
            hash.update(ByteBuffer.allocate(Long.BYTES).putLong(seed).array());
            return ByteBuffer.wrap(hash.digest(key.getBytes(StandardCharsets.UTF_8))).getLong();
        } catch (NoSuchAlgorithmException error) { throw new IllegalStateException(error); }
    }

    /**
     * Carries the base and jittered duration for one simulated forward.
     */
    public record Sample(Duration base, double jitterMs, Duration planned) {
        public Sample {
            if (base == null || planned == null || base.isNegative() || planned.isNegative()
                    || !Double.isFinite(jitterMs))
                throw new IllegalArgumentException("Invalid compute sample");
        }
    }

    /**
     * Validates Gaussian jitter parameters and supplies deterministic defaults.
     */
    record Config(Double stddev_ms, Double stddev_ratio, Double max_deviation_ms, Long seed) {
        Config {
            stddev_ms = number(stddev_ms, 0);
            stddev_ratio = number(stddev_ratio, 0); max_deviation_ms = number(max_deviation_ms, 0);
            boolean random = stddev_ms > 0 || stddev_ratio > 0;
            if (random && (seed == null || max_deviation_ms == 0))
                throw new IllegalArgumentException("Gaussian noise requires seed and positive max_deviation_ms");
            seed = seed == null ? 0 : seed;
        }
        private static double number(Double value, double fallback) {
            if (value == null) return fallback;
            if (!Double.isFinite(value) || value < 0)
                throw new IllegalArgumentException("Noise values must be finite and nonnegative");
            return value;
        }
    }
}
