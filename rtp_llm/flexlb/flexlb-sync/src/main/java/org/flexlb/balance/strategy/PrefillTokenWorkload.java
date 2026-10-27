package org.flexlb.balance.strategy;

import java.util.ArrayDeque;

/** Windowed useful work plus all accepted work still outstanding. No TTFT or cached tokens.
 * Owned by one endpoint generation; completion observations are counted once by its ownership ledger.
 * The caller supplies monotonic milliseconds (also supports deterministic simulation).
 */
public final class PrefillTokenWorkload {
    private record Completion(long atMs, long tokens) { }
    public record Snapshot(long completedTokens, long pendingTokens, long plannedTokens, long plannedExecutionMs,
                           long earliestPlannedTargetMs) {
        public Snapshot(long completedTokens, long pendingTokens, long plannedTokens) {
            this(completedTokens, pendingTokens, plannedTokens, 0L, Long.MAX_VALUE);
        }
        public Snapshot(long completedTokens, long pendingTokens) { this(completedTokens, pendingTokens, 0L); }
        /** Only outstanding compute competes for current service. Completed-window
         * expiry must not move new traffic between otherwise identical workers. */
        public double score() { return (double) pendingTokens + plannedTokens; }
    }
    private final long windowMs;
    private final ArrayDeque<Completion> completions = new ArrayDeque<>();
    private long completedTokens;
    private long pendingTokens;
    private long plannedTokens;
    private long plannedExecutionMs;
    private final java.util.TreeMap<Long, Integer> plannedTargets = new java.util.TreeMap<>();

    /** Planned work shared by a selection and its canonical route copy.
     * It blocks competing selections and transfers exactly once to committed ownership or releases.
     */
    public final class Reservation implements AutoCloseable {
        private final long tokens;
        private boolean open = true;
        private final long executionMs;
        private final long targetMs;
        private Reservation(long tokens, long executionMs, long targetMs) {
            this.tokens = tokens; this.executionMs = executionMs; this.targetMs = targetMs;
        }
        @Override public void close() { release(); }

        /** True only for the caller that releases this shared reservation. */
        public boolean release() {
            synchronized (PrefillTokenWorkload.this) {
                if (open) {
                    plannedTokens -= tokens; plannedExecutionMs -= executionMs;
                    if (targetMs != Long.MAX_VALUE) {
                        plannedTargets.compute(targetMs, (key, count) -> count == 1 ? null : count - 1);
                    }
                    open = false;
                    return true;
                }
                return false;
            }
        }
    }

    public synchronized Reservation plan(long tokens) {
        return plan(tokens, 0L);
    }

    public synchronized Reservation plan(long tokens, long executionMs) {
        return plan(tokens, executionMs, Long.MAX_VALUE);
    }

    public synchronized Reservation plan(long tokens, long executionMs, long targetMs) {
        if (tokens < 0 || executionMs < 0) { throw new IllegalArgumentException("negative planned work"); }
        long nextTokens = Math.addExact(plannedTokens, tokens);
        long nextMs = Math.addExact(plannedExecutionMs, executionMs);
        plannedTokens = nextTokens;
        plannedExecutionMs = nextMs;
        if (targetMs != Long.MAX_VALUE) { plannedTargets.merge(targetMs, 1, Math::addExact); }
        return new Reservation(tokens, executionMs, targetMs);
    }

    public PrefillTokenWorkload(long windowMs) {
        if (windowMs <= 0) { throw new IllegalArgumentException("window must be positive"); }
        this.windowMs = windowMs;
    }

    public synchronized void accept(long tokens) {
        if (tokens < 0) { throw new IllegalArgumentException("negative work"); }
        pendingTokens = Math.addExact(pendingTokens, tokens);
    }

    /** Cancellation/failed admission removes pending work without inventing successful work. */
    public synchronized void release(long tokens) {
        if (tokens < 0 || tokens > pendingTokens) { throw new IllegalArgumentException("unowned work"); }
        pendingTokens -= tokens;
    }

    public synchronized void complete(long tokens, long nowMs) {
        expire(nowMs);
        if (tokens < 0) { throw new IllegalArgumentException("negative work"); }
        completedTokens = Math.addExact(completedTokens, tokens);
        completions.addLast(new Completion(nowMs, tokens));
    }

    public synchronized Snapshot snapshot(long nowMs) {
        expire(nowMs);
        return new Snapshot(completedTokens, pendingTokens, plannedTokens, plannedExecutionMs,
                plannedTargets.isEmpty() ? Long.MAX_VALUE : plannedTargets.firstKey());
    }

    private void expire(long nowMs) {
        while (!completions.isEmpty() && nowMs - completions.getFirst().atMs >= windowMs) {
            completedTokens -= completions.removeFirst().tokens;
        }
    }
}
