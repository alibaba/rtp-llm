package org.flexlb.mock.engine.forward;

/** One component's Forward, Decode progress and EOS models. */
public record ComputeSpec(Forward forward, DecodeSpec decode,
                          StoppingSpec eos) {
    public ComputeSpec {
        if (forward == null) throw new IllegalArgumentException("Compute requires forward");
        decode = decode == null ? new DecodeSpec(null) : decode;
        eos = eos == null ? new StoppingSpec(null, null) : eos;
    }

    public record Forward(String expression, Noise noise) {
        public Forward {
            if (expression == null || expression.isBlank() || expression.length() > 32768)
                throw new IllegalArgumentException("Compute expression must contain 1..32768 characters");
            noise = noise == null ? new Noise(null, null, null, null) : noise;
        }
    }

    public record Noise(Double stddev_ms,
                        Double stddev_ratio,
                        Double max_deviation_ms,
                        Long seed) {
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
            if (!Double.isFinite(value) || value < 0)
                throw new IllegalArgumentException("Noise values must be finite and nonnegative");
            return value;
        }
    }
}
