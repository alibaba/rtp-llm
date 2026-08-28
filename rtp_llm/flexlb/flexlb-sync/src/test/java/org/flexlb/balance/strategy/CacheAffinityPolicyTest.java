package org.flexlb.balance.strategy;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class CacheAffinityPolicyTest {

    @Test
    void selectsLongestPrefixInsideCapInsteadOfRejectingAllForGlobalLeader() {
        long[] scores = {100, 105, 1000};
        long[] hits = {0, 500, 600};

        CacheAffinityPolicy.Decision decision = evaluate(scores, hits, 100, 0, 1000, 10, 5);

        assertTrue(decision.hasPreference());
        assertEquals(CacheAffinityPolicy.Reason.CACHE_LEADER, decision.reason());
        assertArrayEquals(new int[] {1}, preferredIndexes(decision));
    }

    @Test
    void neverFallsBackToLessCacheThanBestMinimumScoreCandidate() {
        long[] scores = {100, 102, 103};
        long[] hits = {100, 50, 0};

        CacheAffinityPolicy.Decision decision = evaluate(scores, hits, 100, 100, 1000, 10, 0);

        assertArrayEquals(new int[] {0}, preferredIndexes(decision));
    }

    @Test
    void returnsOverCapWhenOnlyCacheAdvantageIsTooExpensive() {
        CacheAffinityPolicy.Decision decision = evaluate(
                new long[] {100, 111}, new long[] {0, 500}, 100, 0, 1000, 10, 5);

        assertFalse(decision.hasPreference());
        assertEquals(CacheAffinityPolicy.Reason.OVER_CAP, decision.reason());
    }

    @Test
    void returnsLowHitWhenRelativeAdvantageMissesRateGate() {
        CacheAffinityPolicy.Decision decision = evaluate(
                new long[] {100, 100}, new long[] {0, 49}, 100, 0, 1000, 10, 5);

        assertFalse(decision.hasPreference());
        assertEquals(CacheAffinityPolicy.Reason.LOW_CACHE_HIT, decision.reason());
    }

    @Test
    void equalCacheAcrossWorkersIsNotAffinity() {
        CacheAffinityPolicy.Decision decision = evaluate(
                new long[] {100, 101}, new long[] {500, 500}, 100, 500, 1000, 10, 5);

        assertFalse(decision.hasPreference());
        assertEquals(CacheAffinityPolicy.Reason.NO_CACHE_LEAD, decision.reason());
    }

    @Test
    void capAdditionSaturates() {
        CacheAffinityPolicy.Decision decision = evaluate(
                new long[] {Long.MAX_VALUE - 1, Long.MAX_VALUE},
                new long[] {0, 500},
                Long.MAX_VALUE - 1,
                0,
                1000,
                100,
                5);

        assertTrue(decision.hasPreference());
        assertEquals(Long.MAX_VALUE, decision.projectedTtftCutoffMs());
    }

    @Test
    void nanMinimumHitRateFailsClosedAtOneHundredPercent() {
        CacheAffinityPolicy.Decision decision = evaluate(
                new long[] {100, 100},
                new long[] {0, 500},
                100,
                0,
                1000,
                10,
                Double.NaN);

        assertFalse(decision.hasPreference());
        assertEquals(CacheAffinityPolicy.Reason.LOW_CACHE_HIT, decision.reason());
    }

    private CacheAffinityPolicy.Decision evaluate(
            long[] scores,
            long[] hits,
            long minScore,
            long referenceHit,
            long seqLen,
            long maxExtra,
            double minRate) {
        return CacheAffinityPolicy.evaluate(
                scores.length,
                index -> scores[index],
                index -> hits[index],
                minScore,
                referenceHit,
                seqLen,
                maxExtra,
                minRate);
    }

    private int[] preferredIndexes(CacheAffinityPolicy.Decision decision) {
        int[] result = new int[decision.preferredCount()];
        for (int i = 0; i < result.length; i++) {
            result[i] = decision.preferredIndex(i);
        }
        return result;
    }
}
