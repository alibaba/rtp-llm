package org.flexlb.cache.domain;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyOrder;
import org.flexlb.dao.master.WorkerIdentity;

/**
 * Cache-hit comparison result and PV log payload.
 *
 * @param workerIdentity worker identity providing the routing/PV identity
 *                       {@code ip:port@engineIndex}; omitted from PV JSON itself
 */
@JsonPropertyOrder({
        "event", "requestId", "source", "role", "group", "worker", "state", "inputTokens",
        "actualHitTokens", "kvcmPrediction", "localSyncPrediction", "localStandbyPrediction"
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
        CachePrediction kvcmPrediction,
        CachePrediction localSyncPrediction,
        CachePrediction localStandbyPrediction) {

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
