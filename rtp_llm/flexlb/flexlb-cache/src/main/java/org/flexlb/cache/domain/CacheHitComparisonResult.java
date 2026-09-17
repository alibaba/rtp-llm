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
        "actual", "routing", "kvcm", "localStandby"
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
        Actual actual,
        @JsonIgnore HitComparison routing,
        HitComparison localStandby,
        @JsonIgnore KvcmDetails kvcmDetails) {

    /** Routing/PV identity in {@code ip:port@engineIndex} format. */
    @JsonProperty("worker")
    public String worker() {
        return workerIdentity == null ? null : workerIdentity.getLogicalIpPort();
    }

    @JsonProperty("routing")
    public HitComparison routingPrediction() {
        return CacheMatchSource.KVCM.name().equals(source) ? null : routing;
    }

    @JsonProperty("kvcm")
    public KvcmComparison kvcm() {
        if (!CacheMatchSource.KVCM.name().equals(source) || routing == null) {
            return null;
        }
        return new KvcmComparison(
                routing.hit(),
                routing.delta(),
                kvcmDetails == null ? null : kvcmDetails.local(),
                kvcmDetails == null ? null : kvcmDetails.global());
    }

    public record Actual(long hit) {
    }

    public record HitComparison(long hit, long delta) {
    }

    /**
     * KVCM prediction drill-down. {@code hit}/{@code delta} are the blended prediction used for
     * routing; {@code local} and {@code global} compare the actual hit against the local-only
     * match and the full local+remote match respectively. {@code global.hit} includes
     * {@code local.hit}.
     */
    public record KvcmComparison(long hit,
                                 long delta,
                                 HitComparison local,
                                 HitComparison global) {
    }

    public record KvcmDetails(HitComparison local, HitComparison global) {
    }
}
