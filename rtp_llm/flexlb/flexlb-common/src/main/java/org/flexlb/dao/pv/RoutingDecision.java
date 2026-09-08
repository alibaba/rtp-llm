package org.flexlb.dao.pv;

import org.flexlb.dao.route.RoleType;

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
                              PrefillPolicy prefillPolicy) {
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
                                Double p2pHitDiscount,
                                Long maxOutstandingUncachedTokens,
                                long minimumEffectiveHitTokens,
                                long maximumEffectiveHitTokens,
                                double maxPendingVsAverageMultiplier,
                                double maxDrainVsAverageMultiplier) {
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
