package org.flexlb.mock.cache;

import java.util.*;

/** Immutable views of valid cached blocks; allocated working blocks are not metadata. */
public final class CacheMetadata {
    private CacheMetadata() {}
    public enum Medium { HBM, HOST, NONE }
    public record Change(long version, long key, Medium before, Medium after) {}
    public record Snapshot(long version, Map<Long, Medium> blocks) {
        public Snapshot { blocks = Map.copyOf(blocks); }
        public Set<Long> keys() { return blocks.keySet(); }
        public Set<Long> keys(Medium medium) {
            var keys = new HashSet<Long>();
            blocks.forEach((key, location) -> { if (location == medium) keys.add(key); });
            return Set.copyOf(keys);
        }
    }
}
