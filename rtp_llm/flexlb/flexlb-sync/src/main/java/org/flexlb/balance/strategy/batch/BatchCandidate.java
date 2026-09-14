package org.flexlb.balance.strategy.batch;

import java.util.Objects;

/**
 * Immutable worker cost and capacity snapshot consumed by {@link BatchPlanner}.
 *
 * <p>The snapshot is planning-only: it describes a worker's observed state but
 * does not reserve any capacity. Candidate indexes are local to one
 * {@link BatchPlanningRequest}.</p>
 */
public record BatchCandidate(String worker, long ttftMs, long workMs, long hitTokens,
                             long kvTokens, long availableKvTokens, int availableRequests) {

    public BatchCandidate {
        Objects.requireNonNull(worker, "worker");
        if (ttftMs < 0L || workMs < 0L || hitTokens < 0L
                || kvTokens < 0L || availableKvTokens < 0L
                || availableRequests < 0) {
            throw new IllegalArgumentException(
                    "candidate costs and capacity must be non-negative");
        }
    }

    /** Creates an unconstrained candidate for an algorithm-only test or caller. */
    public BatchCandidate(String worker, long ttftMs, long workMs, long hitTokens) {
        this(worker, ttftMs, workMs, hitTokens, 0L, Long.MAX_VALUE, Integer.MAX_VALUE);
    }
}
