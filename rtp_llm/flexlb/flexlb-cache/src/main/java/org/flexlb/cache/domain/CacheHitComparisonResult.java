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
    public HitComparison kvcm() {
        return CacheMatchSource.KVCM.name().equals(source) ? routing : null;
    }

    public record Actual(long hit) {
    }

    public record HitComparison(long hit, long delta) {
    }

    public record KvcmDetails(long localDelta, long p2pTotalMatchDelta) {
    }
}
