package org.flexlb.cache.domain;

import com.fasterxml.jackson.annotation.JsonIgnore;

/**
 * Cache-hit comparison result and PV log payload.
 */
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
        HitComparison kvcm,
        HitComparison localStandby,
        @JsonIgnore long routingPredictedHitTokens,
        @JsonIgnore long routingDeltaHitTokens,
        @JsonIgnore boolean kvcmPredictionAvailable,
        @JsonIgnore long kvcmLocalDeltaHitTokens,
        @JsonIgnore long kvcmP2pTotalMatchDeltaHitTokens) {

    public CacheHitComparisonResult(
            String event,
            String requestId,
            String source,
            String role,
            String group,
            String worker,
            String state,
            long inputTokens,
            long routingPredictedHitTokens,
            boolean kvcmPredictionAvailable,
            long localStandbyPredictedHitTokens,
            boolean localStandbyPredictionAvailable,
            long actualHitTokens,
            long routingDeltaHitTokens,
            long kvcmLocalDeltaHitTokens,
            long kvcmP2pTotalMatchDeltaHitTokens,
            long localStandbyDeltaHitTokens) {
        this(
                event,
                requestId,
                source,
                role,
                group,
                worker,
                state,
                inputTokens,
                new Actual(actualHitTokens),
                CacheMatchSource.KVCM.name().equals(source)
                        ? new HitComparison(routingPredictedHitTokens, routingDeltaHitTokens)
                        : null,
                localStandbyPredictionAvailable
                        ? new HitComparison(localStandbyPredictedHitTokens, localStandbyDeltaHitTokens)
                        : null,
                routingPredictedHitTokens,
                routingDeltaHitTokens,
                kvcmPredictionAvailable,
                kvcmLocalDeltaHitTokens,
                kvcmP2pTotalMatchDeltaHitTokens);
    }

    public CacheHitComparisonResult(
            String event,
            String requestId,
            String source,
            String role,
            String group,
            String worker,
            String state,
            long inputTokens,
            long routingPredictedHitTokens,
            long localStandbyPredictedHitTokens,
            boolean localStandbyPredictionAvailable,
            long actualHitTokens,
            long routingDeltaHitTokens,
            long localStandbyDeltaHitTokens) {
        this(
                event,
                requestId,
                source,
                role,
                group,
                worker,
                state,
                inputTokens,
                routingPredictedHitTokens,
                false,
                localStandbyPredictedHitTokens,
                localStandbyPredictionAvailable,
                actualHitTokens,
                routingDeltaHitTokens,
                0,
                0,
                localStandbyDeltaHitTokens);
    }

    @JsonIgnore
    public String cacheMatchSource() {
        return source;
    }

    @JsonIgnore
    public String workerIp() {
        return worker;
    }

    @JsonIgnore
    public String taskState() {
        return state;
    }

    @JsonIgnore
    public long actualHitTokens() {
        return actual == null ? 0 : actual.hit();
    }

    @JsonIgnore
    public boolean localStandbyPredictionAvailable() {
        return localStandby != null;
    }

    @JsonIgnore
    public long localStandbyPredictedHitTokens() {
        return localStandby == null ? 0 : localStandby.hit();
    }

    @JsonIgnore
    public long localStandbyDeltaHitTokens() {
        return localStandby == null ? 0 : localStandby.delta();
    }

    public record Actual(long hit) {
    }

    public record HitComparison(long hit, long delta) {
    }
}
