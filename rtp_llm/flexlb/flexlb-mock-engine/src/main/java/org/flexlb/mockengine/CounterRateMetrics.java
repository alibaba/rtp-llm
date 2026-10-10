package org.flexlb.mockengine;

/** Cumulative execution work with independent wall-clock readers for each sink. */
final class CounterRateMetrics {
    record Snapshot(long generation, long startedNanos, long tokens) {}
    private long generation;
    private long startedNanos = System.nanoTime();
    private long tokens;
    synchronized void add(long count) { tokens += count; }
    synchronized Snapshot snapshot() { return new Snapshot(generation, startedNanos, tokens); }
    synchronized void reset(long now) { generation++; startedNanos = now; tokens = 0; }
    static double rate(long delta, long nanos) {
        return nanos > 0 ? Math.max(0, delta) * 1_000_000_000.0 / nanos : 0;
    }
    static final class Reader {
        private Snapshot previous;
        private long reportedAt;
        private double last;
        synchronized double last() { return last; }
        synchronized void reset() { previous = null; reportedAt = 0; last = 0; }
        synchronized double sample(Snapshot current, long now) {
            if (previous == null || previous.generation() != current.generation()) {
                previous = new Snapshot(current.generation(), current.startedNanos(), 0);
                reportedAt = current.startedNanos();
                last = 0;
            }
            if (now <= reportedAt) return last;
            last = rate(current.tokens() - previous.tokens(), now - reportedAt);
            previous = current;
            reportedAt = now;
            return last;
        }
    }
}
