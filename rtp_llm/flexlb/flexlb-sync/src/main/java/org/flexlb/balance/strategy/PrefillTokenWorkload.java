package org.flexlb.balance.strategy;

/** Outstanding uncached work owned by one endpoint generation.
 * Completed work no longer competes for service; no history window or clock is needed.
 */
public final class PrefillTokenWorkload {
    public record Snapshot(long pendingTokens, long plannedTokens) {
        public Snapshot(long pendingTokens) { this(pendingTokens, 0L); }
        public double score() { return (double) pendingTokens + plannedTokens; }
    }
    private long pendingTokens;
    private long plannedTokens;

    /** Soft scoring reservation shared by a selection and its canonical route copy.
     * It never consumes admission capacity and closes exactly once on commit or abandoned planning.
     */
    public final class Reservation implements AutoCloseable {
        private final long tokens;
        private boolean open = true;
        private Reservation(long tokens) { this.tokens = tokens; }
        @Override public void close() {
            synchronized (PrefillTokenWorkload.this) {
                if (open) { plannedTokens -= tokens; open = false; }
            }
        }
    }

    public synchronized Reservation plan(long tokens) {
        if (tokens < 0) { throw new IllegalArgumentException("negative planned work"); }
        plannedTokens = Math.addExact(plannedTokens, tokens);
        return new Reservation(tokens);
    }

    public synchronized void accept(long tokens) {
        if (tokens < 0) { throw new IllegalArgumentException("negative work"); }
        pendingTokens = Math.addExact(pendingTokens, tokens);
    }

    /** Completion, cancellation or failed admission releases the original reservation. */
    public synchronized void release(long tokens) {
        if (tokens < 0 || tokens > pendingTokens) { throw new IllegalArgumentException("unowned work"); }
        pendingTokens -= tokens;
    }

    public synchronized Snapshot snapshot() {
        return new Snapshot(pendingTokens, plannedTokens);
    }
}
