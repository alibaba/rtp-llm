package org.flexlb.balance.strategy;

import java.util.ArrayDeque;

/** Windowed useful work plus all accepted work still outstanding. No TTFT or cached tokens.
 * Owned by one endpoint generation; completion observations are counted once by its ownership ledger.
 * The caller supplies monotonic milliseconds (also supports deterministic simulation).
 */
public final class PrefillTokenWorkload {
    private record Completion(long atMs, long tokens) { }
    public record Snapshot(long completedTokens, long pendingTokens, long plannedTokens) {
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

    /** Planned work shared by a selection and its canonical route copy.
     * It blocks competing selections and transfers exactly once to committed ownership or releases.
     */
    public final class Reservation implements AutoCloseable {
        private final long tokens;
        private boolean open = true;
        private Reservation(long tokens) { this.tokens = tokens; }
        @Override public void close() { release(); }

        /** True only for the caller that releases this shared reservation. */
        public boolean release() {
            synchronized (PrefillTokenWorkload.this) {
                if (open) {
                    plannedTokens -= tokens;
                    open = false;
                    return true;
                }
                return false;
            }
        }
    }

    public synchronized Reservation plan(long tokens) {
        if (tokens < 0) { throw new IllegalArgumentException("negative planned work"); }
        plannedTokens = Math.addExact(plannedTokens, tokens);
        return new Reservation(tokens);
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
        return new Snapshot(completedTokens, pendingTokens, plannedTokens);
    }

    private void expire(long nowMs) {
        while (!completions.isEmpty() && nowMs - completions.getFirst().atMs >= windowMs) {
            completedTokens -= completions.removeFirst().tokens;
        }
    }
}
