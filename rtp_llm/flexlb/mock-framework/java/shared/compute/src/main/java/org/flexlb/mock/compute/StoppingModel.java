package org.flexlb.mock.compute;

import java.util.SplittableRandom;
import org.flexlb.mock.api.model.StoppingSpec;

/** shortcut: constant per-output-token EOS probability; calibrate a richer model when workload data requires it. */
public record StoppingModel(StoppingSpec configuration) {
    public StoppingModel { java.util.Objects.requireNonNull(configuration); }

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
                    return new ComputeModel.Result((int) offset, true);
            }
        }
        return new ComputeModel.Result(count, generated + count == maximum);
    }
}
