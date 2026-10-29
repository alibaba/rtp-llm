package org.flexlb.mock.compute;

import java.util.SplittableRandom;
import org.flexlb.mock.api.model.DecodeSpec;

/** Plan verification before Forward; sample accepted output only after Forward completes. */
public record DecodeStepModel(DecodeSpec configuration) {
    public DecodeStepModel { java.util.Objects.requireNonNull(configuration); }
    public record Plan(int proposedTokens, int verifyTokens, int additionalKvTokens, int fixedCandidates) {}

    public Plan plan(long round, int remaining) {
        if (round < 0 || remaining < 1) throw new IllegalArgumentException("Invalid Decode round/budget");
        if (configuration.tokens_per_step() != null) {
            double mean = configuration.tokens_per_step();
            long count = Math.round(Math.ceil((round + 1.0) * mean) - Math.ceil(round * mean));
            if (count < 1 || count > Integer.MAX_VALUE) throw new IllegalArgumentException("Decode progress exceeds numeric precision");
            // Average progress is not a count of drafts or target verification inputs.
            return new Plan(0, 1, (int) Math.min(remaining, count) - 1, (int) count);
        }
        // RTP verifies the full fixed-width window; output limits are applied after acceptance.
        int proposed = configuration.draft_tokens();
        return new Plan(proposed, proposed + 1, proposed, 0);
    }

    public int candidates(Plan plan, String requestKey, long round) {
        if (requestKey == null || requestKey.isBlank() || round < 0) throw new IllegalArgumentException("Invalid Decode identity/round");
        if (plan.fixedCandidates() > 0) return plan.fixedCandidates();
        int proposed = plan.proposedTokens();
        double alpha = configuration.acceptance_rate();
        int accepted = alpha == 1 ? proposed : 0;
        if (proposed > 0 && alpha > 0 && alpha < 1) {
            double u = 1 - new SplittableRandom(NoiseModel.keyedSeed(configuration.seed(), "decode/" + requestKey + "/" + round)).nextDouble();
            accepted = (int) Math.min(proposed, Math.floor(Math.log(u) / Math.log(alpha)));
        }
        return accepted + 1;
    }
}
