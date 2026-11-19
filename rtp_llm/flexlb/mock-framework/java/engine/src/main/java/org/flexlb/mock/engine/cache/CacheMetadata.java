package org.flexlb.mock.engine.cache;

/**
 * Immutable views of valid cached blocks; allocated working blocks are not metadata.
 */
public final class CacheMetadata {
    private CacheMetadata() {}
    public record Change(long version, long key, boolean added) {}
    /**
     * Caller owns this primitive array and its matching mutation cutoff.
     */
    public record SnapshotKeys(long version, long[] keys) {}
}
