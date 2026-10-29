package org.flexlb.mock.api.model;

import org.flexlb.mock.api.Argument;

/** One component's formula and noise; constants live in expression param(name, value). */
public record ComputeSpec(String expression, @Argument(optional=true) Noise noise) {
    public ComputeSpec {
        if (expression == null || expression.isBlank() || expression.length() > 32768)
            throw new IllegalArgumentException("Compute expression must contain 1..32768 characters");
        noise = noise == null ? new Noise(null, null, null, null) : noise;
    }

    public record Noise(@Argument(optional=true, minimum=0) Double stddev_ms,
                        @Argument(optional=true, minimum=0) Double stddev_ratio,
                        @Argument(optional=true, minimum=0) Double max_deviation_ms,
                        @Argument(optional=true) Long seed) {
        public Noise {
            stddev_ms = number(stddev_ms, 0);
            stddev_ratio = number(stddev_ratio, 0); max_deviation_ms = number(max_deviation_ms, 0);
            boolean random = stddev_ms > 0 || stddev_ratio > 0;
            if (random && (seed == null || max_deviation_ms == 0))
                throw new IllegalArgumentException("Gaussian noise requires seed and positive max_deviation_ms");
            seed = seed == null ? 0 : seed;
        }
        private static double number(Double value, double fallback) {
            if (value == null) return fallback;
            if (!Double.isFinite(value) || value < 0) throw new IllegalArgumentException("Noise values must be finite and nonnegative");
            return value;
        }
    }
}
