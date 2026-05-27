package org.flexlb.cache.monitor;

import java.util.function.LongSupplier;

/**
 * Constant-memory counters for request-level theory cache-hit statistics.
 */
public class CacheHitTheoryStats {

    private static final int MAX_WINDOW_SECONDS = 15 * 60;
    private static final int BUCKET_COUNT = MAX_WINDOW_SECONDS + 2;

    private final LongSupplier nowSupplier;
    private final long[] bucketSeconds = new long[BUCKET_COUNT];
    private final long[] bucketHitCounts = new long[BUCKET_COUNT];
    private final long[] bucketTotalCounts = new long[BUCKET_COUNT];

    private long allHitCount;
    private long allTotalCount;

    public CacheHitTheoryStats() {
        this(System::currentTimeMillis);
    }

    CacheHitTheoryStats(LongSupplier nowSupplier) {
        this.nowSupplier = nowSupplier == null ? System::currentTimeMillis : nowSupplier;
        java.util.Arrays.fill(bucketSeconds, Long.MIN_VALUE);
    }

    public synchronized Snapshot record(long hitCount, long totalCount) {
        return record(hitCount, totalCount, nowSupplier.getAsLong());
    }

    synchronized Snapshot record(long hitCount, long totalCount, long nowMs) {
        long normalizedHit = Math.max(0L, hitCount);
        long normalizedTotal = Math.max(0L, totalCount);
        long currentSecond = Math.floorDiv(nowMs, 1000L);

        if (normalizedTotal > 0L) {
            int index = bucketIndex(currentSecond);
            if (bucketSeconds[index] != currentSecond) {
                bucketSeconds[index] = currentSecond;
                bucketHitCounts[index] = 0L;
                bucketTotalCounts[index] = 0L;
            }
            bucketHitCounts[index] += normalizedHit;
            bucketTotalCounts[index] += normalizedTotal;
            allHitCount += normalizedHit;
            allTotalCount += normalizedTotal;
        }

        return buildSnapshot(nowMs, currentSecond, normalizedHit, normalizedTotal);
    }

    private Snapshot buildSnapshot(long nowMs, long currentSecond, long requestHitCount, long requestTotalCount) {
        return new Snapshot(
                nowMs,
                requestHitCount,
                requestTotalCount,
                allHitCount,
                allTotalCount,
                windowSnapshot("1m", 60_000L, currentSecond),
                windowSnapshot("5m", 5 * 60_000L, currentSecond),
                windowSnapshot("10m", 10 * 60_000L, currentSecond),
                windowSnapshot("15m", 15 * 60_000L, currentSecond));
    }

    private WindowSnapshot windowSnapshot(String label, long windowMs, long currentSecond) {
        long windowSeconds = windowMs / 1000L;
        long hitCount = 0L;
        long totalCount = 0L;
        for (int i = 0; i < BUCKET_COUNT; i++) {
            long bucketSecond = bucketSeconds[i];
            long ageSeconds = currentSecond - bucketSecond;
            if (ageSeconds >= 0L && ageSeconds < windowSeconds) {
                hitCount += bucketHitCounts[i];
                totalCount += bucketTotalCounts[i];
            }
        }
        return new WindowSnapshot(label, windowMs, hitCount, totalCount);
    }

    private static int bucketIndex(long second) {
        return (int) Math.floorMod(second, BUCKET_COUNT);
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
        private final WindowSnapshot window1m;
        private final WindowSnapshot window5m;
        private final WindowSnapshot window10m;
        private final WindowSnapshot window15m;

        private Snapshot(long nowMs,
                         long requestHitCount,
                         long requestTotalCount,
                         long allHitCount,
                         long allTotalCount,
                         WindowSnapshot window1m,
                         WindowSnapshot window5m,
                         WindowSnapshot window10m,
                         WindowSnapshot window15m) {
            this.nowMs = nowMs;
            this.requestHitCount = requestHitCount;
            this.requestTotalCount = requestTotalCount;
            this.allHitCount = allHitCount;
            this.allTotalCount = allTotalCount;
            this.window1m = window1m;
            this.window5m = window5m;
            this.window10m = window10m;
            this.window15m = window15m;
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

        public WindowSnapshot getWindow1m() {
            return window1m;
        }

        public WindowSnapshot getWindow5m() {
            return window5m;
        }

        public WindowSnapshot getWindow10m() {
            return window10m;
        }

        public WindowSnapshot getWindow15m() {
            return window15m;
        }
    }

    public static class WindowSnapshot {
        private final String label;
        private final long windowMs;
        private final long hitCount;
        private final long totalCount;

        private WindowSnapshot(String label, long windowMs, long hitCount, long totalCount) {
            this.label = label;
            this.windowMs = windowMs;
            this.hitCount = hitCount;
            this.totalCount = totalCount;
        }

        public String getLabel() {
            return label;
        }

        public long getWindowMs() {
            return windowMs;
        }

        public long getHitCount() {
            return hitCount;
        }

        public long getTotalCount() {
            return totalCount;
        }

        public double getHitRatio() {
            return ratio(hitCount, totalCount);
        }
    }
}
