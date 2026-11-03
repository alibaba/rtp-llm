package org.flexlb.mock.cache;

/** One cache preparation and its request lease. All mutation is serialized by its KvCache. */
public final class CacheLoad implements AutoCloseable {
    /** A remote error falls back to the valid local prefix; it never creates remote hits. */
    public record Result(long reusedTokens, long remoteTokens, Exception remoteError) {}
    final KvCache owner;
    final KvCache.Lease lease;
    Result result;
    boolean closed;
    Thread worker;

    CacheLoad(KvCache owner, KvCache.Lease lease) { this.owner = owner; this.lease = lease; }
    public KvCache.Lease lease() { return lease; }
    public Result result() { synchronized (owner) { return result; } }
    @Override public void close() { owner.cancel(this); }
}
