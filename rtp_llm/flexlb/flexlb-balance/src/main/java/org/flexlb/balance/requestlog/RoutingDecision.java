package org.flexlb.balance.requestlog;

import org.flexlb.config.service.RoleType;

import java.util.List;
import java.util.Map;

/**
 * Immutable snapshot of an actual selector attempt; unknown estimates are null.
 */
public record RoutingDecision(
        RoleType role,
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
        List<Candidate> candidates) {
    public RoutingDecision {
        rejections = Map.copyOf(rejections);
        candidates = List.copyOf(candidates);
    }

    public record Candidate(
            String endpoint,
            boolean selected,
            Long projectedWorkTokens,
            Long projectedDrainMs,
            Long incomingComputeTokens,
            Long effectiveHitTokens,
            Long routingMatchTokens,
            long pendingRequests,
            Long usedKvTokens,
            Long availableKvTokens,
            Double logWeight,
            String predictionState,
            Long ownershipVersion) {}
}
