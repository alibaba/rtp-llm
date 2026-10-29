package org.flexlb.balance.strategy;

/** Outstanding uncached work owned by one endpoint generation.
 * Completed work no longer competes for service; no history window or clock is needed.
 */
public final class PrefillTokenWorkload {
    public record Snapshot(long pendingTokens, long plannedTokens, long plannedExecutionMs,
                           long earliestPlannedTargetMs) {
        public Snapshot(long pendingTokens, long plannedTokens) {
            this(pendingTokens, plannedTokens, 0L, Long.MAX_VALUE);
        }
        public Snapshot(long pendingTokens) { this(pendingTokens, 0L); }
        public double score() { return (double) pendingTokens + plannedTokens; }
    }
    public record CacheWork(long inputTokens, long computeTokens) {
        public double hitFraction() {
            return inputTokens == 0 ? 0.0 : (double) (inputTokens - computeTokens) / inputTokens;
        }
    }
    private static final class CacheCounters {
        long input;
        long compute;
        CacheCounters(long input, long compute) { this.input = input; this.compute = compute; }
    }
    private final java.util.NavigableMap<Integer, CacheCounters> cacheWork = new java.util.TreeMap<>();

    public synchronized CacheWork cacheWorkAtOrAbove(int priority) {
        long input = 0, compute = 0;
        for (var work : cacheWork.tailMap(priority, true).values()) {
            input = Math.addExact(input, work.input);
            compute = Math.addExact(compute, work.compute);
        }
        return new CacheWork(input, compute);
    }

    private void changeCacheWork(long tokens, long input, int priority, boolean add) {
        if (tokens < 0 || input < tokens) { throw new IllegalArgumentException("invalid cache work"); }
        var previous = cacheWork.get(priority);
        long oldInput = previous == null ? 0 : previous.input;
        long oldCompute = previous == null ? 0 : previous.compute;
        long nextInput = add ? Math.addExact(oldInput, input) : oldInput - input;
        long nextCompute = add ? Math.addExact(oldCompute, tokens) : oldCompute - tokens;
        if (nextInput < 0 || nextCompute < 0 || nextCompute > nextInput) {
            throw new IllegalArgumentException("unowned cache work");
        }
        if (nextInput == 0) { cacheWork.remove(priority); }
        else if (previous == null) { cacheWork.put(priority, new CacheCounters(nextInput, nextCompute)); }
        else { previous.input = nextInput; previous.compute = nextCompute; }
    }

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
        private final long inputTokens;
        private final int priority;
        private Reservation(long tokens, long executionMs, long targetMs, long inputTokens, int priority) {
            this.tokens = tokens; this.executionMs = executionMs; this.targetMs = targetMs;
            this.inputTokens = inputTokens; this.priority = priority;
        }
        @Override public void close() { release(); }

        /** True only for the caller that releases this shared reservation. */
        public boolean release() {
            synchronized (PrefillTokenWorkload.this) {
                if (open) {
                    changeCacheWork(tokens, inputTokens, priority, false);
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
        return plan(tokens, executionMs, targetMs, tokens, 0);
    }

    public synchronized Reservation plan(long tokens, long executionMs, long targetMs, long inputTokens, int priority) {
        if (tokens < 0 || executionMs < 0) { throw new IllegalArgumentException("negative planned work"); }
        long nextTokens = Math.addExact(plannedTokens, tokens);
        long nextMs = Math.addExact(plannedExecutionMs, executionMs);
        changeCacheWork(tokens, inputTokens, priority, true);
        plannedTokens = nextTokens;
        plannedExecutionMs = nextMs;
        if (targetMs != Long.MAX_VALUE) { plannedTargets.merge(targetMs, 1, Math::addExact); }
        return new Reservation(tokens, executionMs, targetMs, inputTokens, priority);
    }

    public synchronized void accept(long tokens) { accept(tokens, tokens, 0); }

    public synchronized void accept(long tokens, long inputTokens, int priority) {
        if (tokens < 0) { throw new IllegalArgumentException("negative work"); }
        long next = Math.addExact(pendingTokens, tokens);
        changeCacheWork(tokens, inputTokens, priority, true);
        pendingTokens = next;
    }

    /** Completion, cancellation or failed admission releases the original reservation. */
    public synchronized void release(long tokens) { release(tokens, tokens, 0); }

    public synchronized void release(long tokens, long inputTokens, int priority) {
        if (tokens < 0 || tokens > pendingTokens) { throw new IllegalArgumentException("unowned work"); }
        changeCacheWork(tokens, inputTokens, priority, false);
        pendingTokens -= tokens;
    }

    public synchronized Snapshot snapshot() {
        return new Snapshot(pendingTokens, plannedTokens, plannedExecutionMs,
                plannedTargets.isEmpty() ? Long.MAX_VALUE : plannedTargets.firstKey());
    }
}
