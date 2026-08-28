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
                    long minProjectedTtftMs,
                    long projectedTtftCutoffMs) {
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
            IntToLongFunction projectedTtftAt,
            IntToLongFunction cacheHitAt,
            long minProjectedTtftMs,
            long referenceHitTokens,
            long seqLen,
            long configuredMaxExtraTtftMs,
            double configuredMinPrefixHitPercent) {
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

        minProjectedTtftMs = Math.max(0L, minProjectedTtftMs);
        referenceHitTokens = clampHit(referenceHitTokens, seqLen);
        long maxExtraTtftMs = Math.max(0L, configuredMaxExtraTtftMs);
        long projectedTtftCutoffMs = saturatingAdd(
                minProjectedTtftMs, maxExtraTtftMs);
        if (maxHitTokens <= minHitTokens) {
            return new Decision(
                    new int[0], 0, Reason.NO_CACHE_LEAD,
                    minProjectedTtftMs, projectedTtftCutoffMs);
        }

        boolean minimumHitRateMet = false;
        int[] preferredIndexes = new int[candidateCount];
        int preferredCount = 0;
        for (int i = 0; i < candidateCount; i++) {
            long hitTokens = clampHit(cacheHitAt.applyAsLong(i), seqLen);
            if (hitTokens <= minHitTokens
                    || hitTokens < referenceHitTokens
                    || !meetsMinimumHitRate(
                            hitTokens, seqLen, configuredMinPrefixHitPercent)) {
                continue;
            }
            minimumHitRateMet = true;
            if (Math.max(0L, projectedTtftAt.applyAsLong(i))
                    <= projectedTtftCutoffMs) {
                preferredIndexes[preferredCount++] = i;
            }
        }

        if (preferredCount == 0) {
            return new Decision(
                    new int[0],
                    0,
                    minimumHitRateMet ? Reason.OVER_CAP : Reason.LOW_CACHE_HIT,
                    minProjectedTtftMs,
                    projectedTtftCutoffMs);
        }

        sortPreferred(
                preferredIndexes, 0, preferredCount - 1,
                projectedTtftAt, cacheHitAt, seqLen);
        return new Decision(
                preferredIndexes,
                preferredCount,
                Reason.CACHE_LEADER,
                minProjectedTtftMs,
                projectedTtftCutoffMs);
    }

    private static boolean meetsMinimumHitRate(
            long hitTokens,
            long seqLen,
            double configuredMinPrefixHitPercent) {
        double minimumHitRate;
        if (Double.isNaN(configuredMinPrefixHitPercent)
                || configuredMinPrefixHitPercent == Double.POSITIVE_INFINITY) {
            minimumHitRate = 100.0;
        } else if (configuredMinPrefixHitPercent == Double.NEGATIVE_INFINITY) {
            minimumHitRate = 0.0;
        } else {
            minimumHitRate = Math.min(
                    100.0, Math.max(0.0, configuredMinPrefixHitPercent));
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
            IntToLongFunction projectedTtftAt,
            IntToLongFunction cacheHitAt,
            long seqLen) {
        int left = low;
        int right = high;
        int pivot = indexes[(low + high) >>> 1];
        while (left <= right) {
            while (compare(
                    indexes[left], pivot,
                    projectedTtftAt, cacheHitAt, seqLen) < 0) {
                left++;
            }
            while (compare(
                    indexes[right], pivot,
                    projectedTtftAt, cacheHitAt, seqLen) > 0) {
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
            sortPreferred(
                    indexes, low, right, projectedTtftAt, cacheHitAt, seqLen);
        }
        if (left < high) {
            sortPreferred(
                    indexes, left, high, projectedTtftAt, cacheHitAt, seqLen);
        }
    }

    private static int compare(
            int leftIndex,
            int rightIndex,
            IntToLongFunction projectedTtftAt,
            IntToLongFunction cacheHitAt,
            long seqLen) {
        int hitOrder = Long.compare(
                clampHit(cacheHitAt.applyAsLong(rightIndex), seqLen),
                clampHit(cacheHitAt.applyAsLong(leftIndex), seqLen));
        if (hitOrder != 0) {
            return hitOrder;
        }
        int ttftOrder = Long.compare(
                Math.max(0L, projectedTtftAt.applyAsLong(leftIndex)),
                Math.max(0L, projectedTtftAt.applyAsLong(rightIndex)));
        return ttftOrder != 0 ? ttftOrder : Integer.compare(leftIndex, rightIndex);
    }
}
