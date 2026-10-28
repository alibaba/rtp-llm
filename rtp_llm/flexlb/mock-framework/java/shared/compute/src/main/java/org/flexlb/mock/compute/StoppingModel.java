package org.flexlb.mock.compute;

import java.util.SplittableRandom;
import org.flexlb.mock.api.model.StoppingSpec;

/** shortcut: constant EOS hazard, replace with observed lengths when calibrating a real workload. */
public record StoppingModel(StoppingSpec configuration) {
    public StoppingModel { java.util.Objects.requireNonNull(configuration); }
    public int length(String requestKey, int minimum, int maximum, boolean ignoreEos) {
        if (minimum < 0 || maximum < 1 || minimum > maximum) throw new IllegalArgumentException("Invalid output limits");
        if (requestKey == null || requestKey.isBlank()) throw new IllegalArgumentException("Request key is required");
        double p = configuration.eos_probability();
        if (ignoreEos || p == 0) return maximum;
        int first = Math.max(1, minimum);
        if (p == 1) return first;
        double u = 1 - new SplittableRandom(NoiseModel.keyedSeed(configuration.seed(), "stop/" + requestKey)).nextDouble();
        int failures = (int) Math.min(maximum - first, Math.floor(Math.log(u) / Math.log1p(-p)));
        return first + failures;
    }
}
