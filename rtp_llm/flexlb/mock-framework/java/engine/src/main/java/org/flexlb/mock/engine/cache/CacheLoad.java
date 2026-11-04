package org.flexlb.mock.engine.cache;

/** One cache preparation and its request lease. All mutation is serialized by its KvCache. */
public final class CacheLoad implements AutoCloseable {
    /** A remote error falls back to the valid local prefix; it never creates remote hits. */
    public record Result(long reusedTokens, long memoryTokens, long remoteTokens, Exception remoteError) {}
    final KvCache owner;
    final KvCache.Lease lease;
    final long startedNanos, preparedNanos;
    boolean attempted;
    Result result;
    boolean closed;
    Thread worker;

    CacheLoad(KvCache owner, KvCache.Lease lease, long startedNanos) {
        this.owner = owner; this.lease = lease; this.startedNanos = startedNanos; preparedNanos = System.nanoTime();
    }
    public long startedNanos() { return startedNanos; }
    public long preparedNanos() { return preparedNanos; }
    public boolean loadAttempted() { return attempted; }
    public KvCache.Lease lease() { return lease; }
    public Result result() { synchronized (owner) { return result; } }
    @Override public void close() { owner.cancel(this); }
}
