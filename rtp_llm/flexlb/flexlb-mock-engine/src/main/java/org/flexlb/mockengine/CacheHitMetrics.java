package org.flexlb.mockengine;

import java.util.OptionalDouble;

/** KVCacheManager::collectCacheHitRates: one-minute, token-weighted percentage. */
final class CacheHitMetrics {
    record Snapshot(long startedNanos, long inputTokens, long reuseTokens) {}
    private long startedNanos = System.nanoTime();
    private long inputTokens;
    private long reuseTokens;

    synchronized void reset(long now) { startedNanos = now; inputTokens = reuseTokens = 0; }

    synchronized void record(long input, long reuse) {
        inputTokens += input;
        reuseTokens += reuse;
    }

    synchronized Snapshot snapshot() {
        return new Snapshot(startedNanos, inputTokens, reuseTokens);
    }

    static final class Reader {
        private static final long WINDOW_NANOS = 60_000_000_000L;
        private Snapshot previous;
        private long reportedAt;

        synchronized OptionalDouble sample(Snapshot current, long now) {
            if (previous == null || previous.startedNanos() != current.startedNanos()) {
                previous = new Snapshot(current.startedNanos(), 0, 0);
                reportedAt = current.startedNanos();
            }
            if (now - reportedAt < WINDOW_NANOS) return OptionalDouble.empty();
            long input = current.inputTokens() - previous.inputTokens();
            long reuse = current.reuseTokens() - previous.reuseTokens();
            previous = current;
            reportedAt = now;
            return input > 0 ? OptionalDouble.of(100.0 * reuse / input) : OptionalDouble.empty();
        }
    }
}
