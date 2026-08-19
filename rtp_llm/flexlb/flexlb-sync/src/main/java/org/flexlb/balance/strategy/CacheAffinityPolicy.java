package org.flexlb.balance.strategy;

import java.util.function.IntToLongFunction;

/** Shared bounded cache-affinity policy layered on top of a baseline prefill strategy. */
final class CacheAffinityPolicy {

    enum Reason {
        CACHE_LEADER,
        NO_CACHE_LEAD,
        LOW_CACHE_HIT,
        OVER_CAP
    }

    record Decision(int[] preferredIndexes,
                    int preferredCount,
                    Reason reason,
                    long minScoreMs,
                    long scoreCutoffMs) {
        boolean hasPreference() {
            return preferredCount > 0;
        }

        int preferredIndex(int position) {
            if (position < 0 || position >= preferredCount) {
                throw new IndexOutOfBoundsException(position);
            }
            return preferredIndexes[position];
        }
    }

    private CacheAffinityPolicy() {}

    static Decision evaluate(
            int candidateCount,
            IntToLongFunction scoreAt,
            IntToLongFunction cacheHitAt,
            long minScoreMs,
            long referenceHitTokens,
            long seqLen,
            long configuredMaxExtraTtftMs,
            double configuredMinHitRate) {
        if (candidateCount <= 0) {
            return new Decision(
                    new int[0], 0, Reason.NO_CACHE_LEAD, Long.MAX_VALUE, Long.MAX_VALUE);
        }

        long minHitTokens = Long.MAX_VALUE;
        long maxHitTokens = 0L;
        for (int i = 0; i < candidateCount; i++) {
            long hitTokens = clampHit(cacheHitAt.applyAsLong(i), seqLen);
            minHitTokens = Math.min(minHitTokens, hitTokens);
            maxHitTokens = Math.max(maxHitTokens, hitTokens);
        }

        minScoreMs = Math.max(0L, minScoreMs);
        referenceHitTokens = clampHit(referenceHitTokens, seqLen);
        long maxExtraTtftMs = Math.max(0L, configuredMaxExtraTtftMs);
        long scoreCutoffMs = saturatingAdd(minScoreMs, maxExtraTtftMs);
        if (maxHitTokens <= minHitTokens) {
            return new Decision(
                    new int[0], 0, Reason.NO_CACHE_LEAD, minScoreMs, scoreCutoffMs);
        }

        boolean minimumHitRateMet = false;
        int[] preferredIndexes = new int[candidateCount];
        int preferredCount = 0;
        for (int i = 0; i < candidateCount; i++) {
            long hitTokens = clampHit(cacheHitAt.applyAsLong(i), seqLen);
            if (hitTokens <= minHitTokens
                    || hitTokens < referenceHitTokens
                    || !meetsMinimumHitRate(hitTokens, seqLen, configuredMinHitRate)) {
                continue;
            }
            minimumHitRateMet = true;
            if (Math.max(0L, scoreAt.applyAsLong(i)) <= scoreCutoffMs) {
                preferredIndexes[preferredCount++] = i;
            }
        }

        if (preferredCount == 0) {
            return new Decision(
                    new int[0],
                    0,
                    minimumHitRateMet ? Reason.OVER_CAP : Reason.LOW_CACHE_HIT,
                    minScoreMs,
                    scoreCutoffMs);
        }

        sortPreferred(
                preferredIndexes, 0, preferredCount - 1, scoreAt, cacheHitAt, seqLen);
        return new Decision(
                preferredIndexes,
                preferredCount,
                Reason.CACHE_LEADER,
                minScoreMs,
                scoreCutoffMs);
    }

    private static boolean meetsMinimumHitRate(
            long hitTokens, long seqLen, double configuredMinHitRate) {
        double minimumHitRate;
        if (Double.isNaN(configuredMinHitRate)
                || configuredMinHitRate == Double.POSITIVE_INFINITY) {
            minimumHitRate = 100.0;
        } else if (configuredMinHitRate == Double.NEGATIVE_INFINITY) {
            minimumHitRate = 0.0;
        } else {
            minimumHitRate = Math.min(100.0, Math.max(0.0, configuredMinHitRate));
        }
        return minimumHitRate <= 0.0
                || seqLen > 0L && hitTokens * 100.0 / seqLen >= minimumHitRate;
    }

    private static long saturatingAdd(long left, long right) {
        return left > Long.MAX_VALUE - right ? Long.MAX_VALUE : left + right;
    }

    private static long clampHit(long hitTokens, long seqLen) {
        long nonNegativeHit = Math.max(0L, hitTokens);
        return seqLen > 0L ? Math.min(nonNegativeHit, seqLen) : 0L;
    }

    private static void sortPreferred(
            int[] indexes,
            int low,
            int high,
            IntToLongFunction scoreAt,
            IntToLongFunction cacheHitAt,
            long seqLen) {
        int left = low;
        int right = high;
        int pivot = indexes[(low + high) >>> 1];
        while (left <= right) {
            while (compare(indexes[left], pivot, scoreAt, cacheHitAt, seqLen) < 0) {
                left++;
            }
            while (compare(indexes[right], pivot, scoreAt, cacheHitAt, seqLen) > 0) {
                right--;
            }
            if (left <= right) {
                int value = indexes[left];
                indexes[left] = indexes[right];
                indexes[right] = value;
                left++;
                right--;
            }
        }
        if (low < right) {
            sortPreferred(indexes, low, right, scoreAt, cacheHitAt, seqLen);
        }
        if (left < high) {
            sortPreferred(indexes, left, high, scoreAt, cacheHitAt, seqLen);
        }
    }

    private static int compare(
            int leftIndex,
            int rightIndex,
            IntToLongFunction scoreAt,
            IntToLongFunction cacheHitAt,
            long seqLen) {
        int hitOrder = Long.compare(
                clampHit(cacheHitAt.applyAsLong(rightIndex), seqLen),
                clampHit(cacheHitAt.applyAsLong(leftIndex), seqLen));
        if (hitOrder != 0) {
            return hitOrder;
        }
        int scoreOrder = Long.compare(
                Math.max(0L, scoreAt.applyAsLong(leftIndex)),
                Math.max(0L, scoreAt.applyAsLong(rightIndex)));
        return scoreOrder != 0 ? scoreOrder : Integer.compare(leftIndex, rightIndex);
    }
}
