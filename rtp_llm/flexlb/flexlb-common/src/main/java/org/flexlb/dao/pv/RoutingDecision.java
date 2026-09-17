package org.flexlb.dao.pv;

import com.fasterxml.jackson.annotation.JsonInclude;
import org.flexlb.dao.route.RoleType;
import org.springframework.lang.NonNull;

import java.util.List;
import java.util.Map;

/** Immutable snapshot of an actual selector attempt; unknown estimates are null. */
public record RoutingDecision(RoleType role,
                              String group,
                              String strategy,
                              String selectionReason,
                              long decisionTimeMs,
                              int routingAttempt,
                              String selectedEndpoint,
                              int totalWorkerCount,
                              int candidateWorkerCount,
                              boolean snapshotTruncated,
                              Map<String, Integer> rejections,
                              List<Candidate> candidates,
                              PrefillPolicy prefillPolicy,
                              @JsonInclude(JsonInclude.Include.NON_NULL) GlobalPlanning globalPlanning) {
    public RoutingDecision {
        rejections = Map.copyOf(rejections);
        candidates = List.copyOf(candidates);
    }

    /** Configuration and population summary used by the current prefill selector, in explicit units. */
    public record PrefillPolicy(long requestInputTokens,
                                String candidateChoice,
                                Long minimumTtftMs,
                                double relativeTolerance,
                                long minimumToleranceMs,
                                int shortestTtftPoolSize,
                                Long cacheAffinityMaxExtraTtftMs,
                                Double cacheAffinityMinPrefixHitPercent,
                                Double remoteDiscount,
                                Long maxOutstandingUncachedTokens,
                                long minimumEffectiveHitTokens,
                                long maximumEffectiveHitTokens,
                                double maxPendingVsAverageMultiplier,
                                double maxDrainVsAverageMultiplier) {
    }

    /**
     * Summary of the joint planner invocation that selected this request.
     *
     * <p>The global decision id is shared by every modeled request in one
     * priority-tier plan. {@code requestChanges} names only phases which
     * changed this individual request; the phase summaries remain group-wide.
     * This is planner evidence, not a reservation or engine-execution trace.</p>
     */
    public record GlobalPlanning(@NonNull String decisionId,
                                 int requestCount,
                                 int greedyPlacedCount,
                                 int finalPlacedCount,
                                 int finalChangedFromGreedyCount,
                                 @NonNull CompletionSearch completionSearch,
                                 @NonNull Optimization ttftOptimization,
                                 @NonNull Optimization cacheAffinityOptimization,
                                 @NonNull List<RequestChange> requestChanges) {
        public GlobalPlanning {
            if (requestCount < 1 || greedyPlacedCount < 0 || finalPlacedCount < 0
                    || finalChangedFromGreedyCount < 0) {
                throw new IllegalArgumentException("global planning counts must be non-negative");
            }
            requestChanges = List.copyOf(requestChanges);
        }
    }

    /** Completion-search work performed only after greedy left a request unplaced. */
    public record CompletionSearch(boolean invoked,
                                   int evaluations,
                                   boolean budgetExhausted,
                                   int recoveredPlacements,
                                   int reassignedRequests) {
    }

    /** Accepted operations and aggregate objective deltas for one local-improvement phase. */
    public record Optimization(boolean invoked,
                               int evaluations,
                               boolean budgetExhausted,
                               int moveCount,
                               int swapCount,
                               int changedRequestCount,
                               long virtualTtftDeltaMs,
                               long cacheHitTokenDelta) {
    }

    /** A phase that changed this request relative to the immediately preceding plan. */
    public enum RequestChange {
        COMPLETION_REPAIR,
        COMPLETION_REASSIGNMENT,
        TTFT_MOVE,
        TTFT_SWAP,
        CACHE_AFFINITY_MOVE,
        CACHE_AFFINITY_SWAP
    }

    public record Candidate(String endpoint,
                            boolean selected,
                            Long projectedTtftMs,
                            Long projectedDrainMs,
                            Long incomingPrefillMs,
                            Long effectiveHitTokens,
                            Long routingMatchTokens,
                            long pendingRequests,
                            Long usedKvTokens,
                            Long availableKvTokens,
                            Double logWeight,
                            String predictionState,
                            Long ownershipVersion) {
    }
}
