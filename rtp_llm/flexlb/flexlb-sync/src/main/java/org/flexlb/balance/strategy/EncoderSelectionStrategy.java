package org.flexlb.balance.strategy;

import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.scheduler.PlacementKey;

import java.util.List;

/**
 * Chooses an Encoder from read-only load observations; the caller acquires its generation.
 */
@FunctionalInterface
public interface EncoderSelectionStrategy {
    interface Candidate {
        String address();

        String group();

        boolean available();

        long requestCount();

        long uncachedTokens();
    }

    PlacementResult<Candidate, PlacementKey> select(
            List<? extends Candidate> candidates, String group, boolean weighted, int requestLimit);
}
