package org.flexlb.balance.strategy;

import org.flexlb.balance.RoutingResult;
import org.flexlb.balance.scheduler.CapacityKey;

import java.util.List;

/**
 * Chooses an Encoder from read-only load observations; the caller acquires its generation.
 */
@FunctionalInterface
public interface EncoderSelectionStrategy {
    RoutingResult<Candidate, CapacityKey> select(
            List<? extends Candidate> candidates, String group, boolean weighted, int requestLimit);

    /**
     * Exposes the Encoder availability and load inputs needed to select a worker.
     */
    interface Candidate {
        String address();

        String group();

        boolean available();

        long requestCount();

        long uncachedTokens();
    }
}
