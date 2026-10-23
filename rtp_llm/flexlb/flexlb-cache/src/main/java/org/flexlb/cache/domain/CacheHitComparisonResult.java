package org.flexlb.cache.domain;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyOrder;
import org.flexlb.dao.master.WorkerIdentity;

/**
 * Cache-hit comparison result and PV log payload.
 * Actual hit tokens are -1 when engine feedback is unavailable; theory and prediction fields are nullable.
 *
 * @param workerIdentity worker identity providing the routing/PV identity
 *                       {@code ip:port@engineIndex}; omitted from PV JSON itself
 */
@JsonPropertyOrder({
        "event", "requestId", "source", "role", "group", "worker", "state", "inputTokens",
        "actualHitTokens", "theoryHitTokens", "kvcmPrediction", "localSyncPrediction", "localStandbyPrediction"
})
public record CacheHitComparisonResult(
        String event,
        String requestId,
        String source,
        String role,
        String group,
        @JsonIgnore WorkerIdentity workerIdentity,
        String state,
        long inputTokens,
        long actualHitTokens,
        Long theoryHitTokens,
        CachePrediction kvcmPrediction,
        CachePrediction localSyncPrediction,
        CachePrediction localStandbyPrediction) {

    public CacheHitComparisonResult withTheoryHitTokens(Long tokens) {
        return new CacheHitComparisonResult(event, requestId, source, role, group, workerIdentity, state,
                inputTokens, actualHitTokens, tokens, kvcmPrediction, localSyncPrediction, localStandbyPrediction);
    }

    /** Routing/PV identity in {@code ip:port@engineIndex} format. */
    @JsonProperty("worker")
    public String worker() {
        return workerIdentity == null ? null : workerIdentity.getLogicalIpPort();
    }

    /**
     * Prediction tokens used for one cache-match source. Local and global prediction tokens are
     * {@code -1} when that source does not calculate separate local and global values.
     */
    public record CachePrediction(long predictedHitTokens,
                                  long localPredictionTokens,
                                  long globalPredictionTokens) {
    }
}
