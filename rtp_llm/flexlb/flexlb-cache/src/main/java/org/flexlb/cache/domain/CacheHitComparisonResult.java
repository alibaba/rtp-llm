package org.flexlb.cache.domain;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyOrder;

/**
 * Cache-hit comparison result and PV log payload.
 */
@JsonPropertyOrder({
        "event", "requestId", "source", "role", "group", "worker", "state", "inputTokens",
        "actual", "kvcm", "localStandby"
})
public record CacheHitComparisonResult(
        String event,
        String requestId,
        String source,
        String role,
        String group,
        String worker,
        String state,
        long inputTokens,
        Actual actual,
        @JsonIgnore HitComparison routing,
        HitComparison localStandby,
        @JsonIgnore KvcmDetails kvcmDetails) {

    @JsonProperty("kvcm")
    public KvcmComparison kvcm() {
        if (!CacheMatchSource.KVCM.name().equals(source) || routing == null) {
            return null;
        }
        return new KvcmComparison(
                routing.hit(),
                routing.delta(),
                kvcmDetails == null ? null : kvcmDetails.local(),
                kvcmDetails == null ? null : kvcmDetails.p2pTotal());
    }

    public record Actual(long hit) {
    }

    public record HitComparison(long hit, long delta) {
    }

    /**
     * KVCM prediction drill-down. {@code hit}/{@code delta} are the blended prediction used for
     * routing; {@code local} and {@code p2pTotal} compare the actual hit against the local-only
     * match and the full local+P2P match respectively. {@code p2pTotal.hit} includes
     * {@code local.hit}.
     */
    public record KvcmComparison(long hit,
                                 long delta,
                                 HitComparison local,
                                 HitComparison p2pTotal) {
    }

    public record KvcmDetails(HitComparison local, HitComparison p2pTotal) {
    }
}
