package org.flexlb.mock.engine.forward;

import java.util.SplittableRandom;

/**
 * Applies a constant per-token EOS probability at each absolute output position.
 */
record EosPolicy(Config configuration) {
    EosPolicy { java.util.Objects.requireNonNull(configuration); }

    public ComputeModel.Result complete(String requestKey, int generated, int candidates, int minimum, int maximum, boolean ignoreEos) {
        if (requestKey == null || requestKey.isBlank() || generated < 0 || candidates < 1
                || minimum < 0 || maximum < 1 || generated >= maximum)
            throw new IllegalArgumentException("Invalid output identity/limits");
        int count = Math.min(candidates, maximum - generated);
        double p = configuration.probability();
        if (!ignoreEos && p > 0 && (long) generated + count >= Math.max(1, minimum)) {
            long seed = NoiseModel.keyedSeed(configuration.seed(), "eos/" + requestKey);
            for (long offset = Math.max(1, minimum - generated); offset <= count; offset++) {
                // Absolute output position makes EOS independent of batch membership and Decode round width.
                if (p == 1 || new SplittableRandom(seed ^ ((long) generated + offset)).nextDouble() < p)
                    return new ComputeModel.Result((int) offset, true, candidates);
            }
        }
        return new ComputeModel.Result(count, generated + count == maximum, candidates);
    }

    /**
     * Validates EOS probability and requires a seed when stopping is random.
     */
    record Config(Double probability,
                               Long seed) {
        Config {
            probability = probability == null ? 0 : probability;
            if (!Double.isFinite(probability) || probability < 0 || probability > 1)
                throw new IllegalArgumentException("probability must be within [0,1]");
            if (probability > 0 && probability < 1 && seed == null)
                throw new IllegalArgumentException("Random EOS stopping requires seed");
            seed = seed == null ? 0 : seed;
        }
    }
}
