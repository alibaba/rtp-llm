package org.flexlb.mock.compute;

import java.util.SplittableRandom;
import org.flexlb.mock.api.model.DecodeSpec;

/** Speculative accepted-prefix sampling or an explicit deterministic mean-token budget. */
public record DecodeStepModel(DecodeSpec configuration) {
    public DecodeStepModel { java.util.Objects.requireNonNull(configuration); }
    public record Step(int proposedTokens, int acceptedTokens, int committedTokens, int verifyTokens) {}

    public Step sample(String requestKey, long round, int remainingBudget) {
        if (round < 0 || remainingBudget < 1) throw new IllegalArgumentException("Invalid Decode round/budget");
        if (requestKey == null || requestKey.isBlank()) throw new IllegalArgumentException("Request key is required");
        if (configuration.tokens_per_step() != null) {
            double mean = configuration.tokens_per_step();
            // Preserve fractional progress across rounds without request-local mutable state.
            long committed = Math.round(Math.floor((round + 1.0) * mean) - Math.floor(round * mean));
            if (committed < 1) throw new IllegalArgumentException("Decode round exceeds numeric progress precision");
            int count = (int) Math.min(remainingBudget, committed);
            return new Step(count - 1, count - 1, count, count);
        }
        int proposed = Math.min(configuration.draft_tokens(), remainingBudget - 1);
        double alpha = configuration.acceptance_rate();
        int accepted = alpha == 1 ? proposed : 0;
        if (proposed > 0 && alpha > 0 && alpha < 1) {
            double u = 1 - new SplittableRandom(NoiseModel.keyedSeed(configuration.seed(), "decode/" + requestKey + "/" + round)).nextDouble();
            accepted = (int) Math.min(proposed, Math.floor(Math.log(u) / Math.log(alpha)));
        }
        return new Step(proposed, accepted, accepted + 1, proposed + 1);
    }
}
