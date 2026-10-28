package org.flexlb.mock.compute;

import java.util.SplittableRandom;
import org.flexlb.mock.api.model.DecodeSpec;

/** Capped geometric accepted prefix, followed by the classical correction/bonus token. */
public record DecodeStepModel(DecodeSpec configuration) {
    public DecodeStepModel { java.util.Objects.requireNonNull(configuration); }
    public record Step(int proposedTokens, int acceptedTokens, int committedTokens, int verifyTokens) {}

    public Step sample(String requestKey, long round, int remainingBudget) {
        if (round < 0 || remainingBudget < 1) throw new IllegalArgumentException("Invalid Decode round/budget");
        if (requestKey == null || requestKey.isBlank()) throw new IllegalArgumentException("Request key is required");
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
