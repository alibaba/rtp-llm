package org.flexlb.dao.pv;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;
import org.flexlb.dao.route.RoleType;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** Immutable snapshot of an actual selector attempt; unknown estimates are null. */
public record RoutingDecision(RoleType role,
                              String group,
                              String strategy,
                              String selectionReason,
                              @JsonInclude(JsonInclude.Include.NON_NULL) String cacheAffinityDecision,
                              @JsonInclude(JsonInclude.Include.NON_NULL) CacheAffinity cacheAffinity,
                              @JsonInclude(JsonInclude.Include.NON_NULL) KvcmMatches kvcmMatches,
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

    @JsonProperty("decisionTime")
    @JsonInclude(JsonInclude.Include.NON_NULL)
    public String getDecisionTime() {
        return PvLogTimeFormatter.format(decisionTimeMs);
    }

    public record KvcmMatches(int reportedNodeCount, Map<String, Long> topHits, Long selectedHitTokens) {
        public KvcmMatches {
            topHits = Collections.unmodifiableMap(new LinkedHashMap<>(topHits));
        }
    }

    /**
     * Decision-time thresholds and the fastest higher-hit candidate meeting the minimum hit rate.
     * Unknown candidate estimates and the corresponding over-cap amount are absent.
     */
    @JsonInclude(JsonInclude.Include.NON_NULL)
    public record CacheAffinity(long minimumTtftMs,
                                long maxExtraTtftMs,
                                double minPrefixHitPercent,
                                long ttftCutoffMs,
                                String fastestHigherHitEndpoint,
                                Long fastestHigherHitTtftMs,
                                Long fastestHigherHitTokens,
                                Long overCapMs) {
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
