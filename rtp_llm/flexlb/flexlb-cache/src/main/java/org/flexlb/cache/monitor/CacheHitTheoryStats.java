package org.flexlb.cache.monitor;

import java.util.function.LongSupplier;

/**
 * Cumulative token counters for request-level theory cache-hit statistics.
 */
public class CacheHitTheoryStats {

    private final LongSupplier nowSupplier;

    private long allHitCount;
    private long allTotalCount;

    public CacheHitTheoryStats() {
        this(System::currentTimeMillis);
    }

    CacheHitTheoryStats(LongSupplier nowSupplier) {
        this.nowSupplier = nowSupplier == null ? System::currentTimeMillis : nowSupplier;
    }

    public synchronized Snapshot record(long hitCount, long totalCount) {
        return record(hitCount, totalCount, nowSupplier.getAsLong());
    }

    synchronized Snapshot record(long hitCount, long totalCount, long nowMs) {
        long normalizedHit = Math.max(0L, hitCount);
        long normalizedTotal = Math.max(0L, totalCount);

        if (normalizedTotal > 0L) {
            allHitCount += normalizedHit;
            allTotalCount += normalizedTotal;
        }

        return new Snapshot(
                nowMs,
                normalizedHit,
                normalizedTotal,
                allHitCount,
                allTotalCount);
    }

    private static double ratio(long hitCount, long totalCount) {
        return totalCount > 0L ? (double) hitCount / totalCount : 0.0D;
    }

    public static class Snapshot {
        private final long nowMs;
        private final long requestHitCount;
        private final long requestTotalCount;
        private final long allHitCount;
        private final long allTotalCount;

        private Snapshot(long nowMs,
                         long requestHitCount,
                         long requestTotalCount,
                         long allHitCount,
                         long allTotalCount) {
            this.nowMs = nowMs;
            this.requestHitCount = requestHitCount;
            this.requestTotalCount = requestTotalCount;
            this.allHitCount = allHitCount;
            this.allTotalCount = allTotalCount;
        }

        public long getNowMs() {
            return nowMs;
        }

        public long getRequestHitCount() {
            return requestHitCount;
        }

        public long getRequestTotalCount() {
            return requestTotalCount;
        }

        public double getRequestHitRatio() {
            return ratio(requestHitCount, requestTotalCount);
        }

        public long getAllHitCount() {
            return allHitCount;
        }

        public long getAllTotalCount() {
            return allTotalCount;
        }

        public double getAllHitRatio() {
            return ratio(allHitCount, allTotalCount);
        }
    }
}
