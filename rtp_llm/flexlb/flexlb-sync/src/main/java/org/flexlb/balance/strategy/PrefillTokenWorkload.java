package org.flexlb.balance.strategy;

/** Outstanding uncached work, with optional completed history used only for ranking. */
public final class PrefillTokenWorkload {
    public record Snapshot(long completedTokens, long pendingTokens, long plannedTokens) {
        public Snapshot(long pendingTokens, long plannedTokens) { this(0L, pendingTokens, plannedTokens); }
        public Snapshot(long pendingTokens) { this(0L, pendingTokens, 0L); }
        public double score() { return (double) completedTokens + pendingTokens + plannedTokens; }
    }
    private record Completion(long atMs, long tokens) { }
    private final long windowMs;
    private final java.util.function.LongSupplier clock;
    private final java.util.ArrayDeque<Completion> completions;
    private long completedTokens;

    public PrefillTokenWorkload() { this(0L); }
    public PrefillTokenWorkload(long windowMs) { this(windowMs, () -> System.nanoTime() / 1_000_000L); }
    PrefillTokenWorkload(long windowMs, java.util.function.LongSupplier clock) {
        if (windowMs < 0L) { throw new IllegalArgumentException("negative window"); }
        this.windowMs = windowMs;
        this.clock = clock;
        completions = windowMs == 0L ? null : new java.util.ArrayDeque<>();
    }

    /** Record successful work separately from releasing its original reservation. */
    public synchronized void complete(long tokens) {
        if (tokens < 0L) { throw new IllegalArgumentException("negative completion"); }
        if (completions == null || tokens == 0L) { return; }
        long nowMs = clock.getAsLong();
        expire(nowMs);
        completedTokens = Math.addExact(completedTokens, tokens);
        completions.addLast(new Completion(nowMs, tokens));
    }

    private void expire(long nowMs) {
        while (!completions.isEmpty() && nowMs - completions.getFirst().atMs() >= windowMs) {
            completedTokens -= completions.removeFirst().tokens();
        }
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
        if (completions != null) { expire(clock.getAsLong()); }
        return new Snapshot(completedTokens, pendingTokens, plannedTokens);
    }
}
