package org.flexlb.balance.strategy.batch;

import org.springframework.lang.NonNull;

import java.util.List;

/**
 * Final global batch assignment together with the bounded planner work that produced it.
 *
 * <p>The trace distinguishes the initial regret-ordered greedy assignment from completion,
 * TTFT, and cache-affinity optimization. It describes planning only; it does not imply that
 * later endpoint reservations or request publication succeeded.</p>
 */
public record BatchPlan(@NonNull List<Integer> selections,
                        int greedyPlacedCount,
                        int finalPlacedCount,
                        int finalChangedFromGreedyCount,
                        @NonNull CompletionSearch completionSearch,
                        @NonNull Optimization ttftOptimization,
                        @NonNull Optimization cacheAffinityOptimization,
                        @NonNull List<List<RequestChange>> requestChanges) {

    public BatchPlan {
        selections = List.copyOf(selections);
        requestChanges = requestChanges.stream().map(List::copyOf).toList();
        if (greedyPlacedCount < 0 || finalPlacedCount < 0
                || finalChangedFromGreedyCount < 0
                || requestChanges.size() != selections.size()) {
            throw new IllegalArgumentException("batch plan counts must match its selections");
        }
    }

    /** Planner phases that changed an individual request from its immediately preceding plan. */
    public enum RequestChange {
        COMPLETION_REPAIR,
        COMPLETION_REASSIGNMENT,
        TTFT_MOVE,
        TTFT_SWAP,
        CACHE_AFFINITY_MOVE,
        CACHE_AFFINITY_SWAP
    }

    /** Completion-search activity after greedy left one or more requests unplaced. */
    public record CompletionSearch(boolean invoked,
                                   int evaluations,
                                   boolean budgetExhausted,
                                   int recoveredPlacements,
                                   int reassignedRequests) {
        public CompletionSearch {
            if (evaluations < 0 || recoveredPlacements < 0 || reassignedRequests < 0) {
                throw new IllegalArgumentException("completion counters must be non-negative");
            }
        }
    }

    /** Accepted local optimization operations and their aggregate objective deltas. */
    public record Optimization(boolean invoked,
                               int evaluations,
                               boolean budgetExhausted,
                               int moveCount,
                               int swapCount,
                               int changedRequestCount,
                               long virtualTtftDeltaMs,
                               long cacheHitTokenDelta) {
        public Optimization {
            if (evaluations < 0 || moveCount < 0 || swapCount < 0 || changedRequestCount < 0) {
                throw new IllegalArgumentException("optimization counters must be non-negative");
            }
        }

        static Optimization notInvoked() {
            return new Optimization(false, 0, false, 0, 0, 0, 0L, 0L);
        }
    }
}
