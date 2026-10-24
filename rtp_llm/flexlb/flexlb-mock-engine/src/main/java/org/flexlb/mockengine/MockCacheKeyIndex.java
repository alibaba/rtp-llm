package org.flexlb.mockengine;

import java.util.HashMap;
import java.util.Map;

/** Published logical nodes, independent of tier backing and request references.
 * Pool mutations may enter this lock; readers never enter pool locks from here.
 */
final class MockCacheKeyIndex {
    private final Map<Long, Integer> tiers = new HashMap<>();
    private long version = -1;
    private CacheSyncFixture fixture;
    private long fixtureVersion;

    synchronized void reset() {
        tiers.clear(); version = -1; fixture = null; fixtureVersion = 0;
    }

    synchronized void update(long key, boolean added) {
        int before = tiers.getOrDefault(key, 0);
        int after = before + (added ? 1 : -1);
        if (after < 0) throw new IllegalStateException("unbalanced cache index removal");
        if (after == 0) tiers.remove(key); else tiers.put(key, after);
        if ((before == 0) != (after == 0)) version++;
    }

    synchronized java.util.Set<Long> keys() { return java.util.Set.copyOf(tiers.keySet()); }

    record Snapshot(long version, long[] natural, CacheFixtureWire.Keys external) {}

    synchronized long version(CacheSyncFixture source) {
        snapshot(source, Long.MIN_VALUE, false);
        return version;
    }

    synchronized Snapshot snapshot(CacheSyncFixture source, long latest, boolean needKeys) {
        CacheSyncFixture.Snapshot external = source == null ? null : source.capture();
        if (fixture != source) {
            // Controlled fixtures are a harness-only source of logical keys.
            version++;
            fixture = source;
            fixtureVersion = external == null ? 0 : external.version();
        } else if (external != null && external.version() != fixtureVersion) {
            version += external.version() - fixtureVersion;
            fixtureVersion = external.version();
        }
        boolean full = needKeys && latest != version;
        if (source != null && needKeys) source.recordResponse(full, external.keys());
        long[] keys = new long[full ? tiers.size() : 0];
        if (full) {
            int i = 0;
            for (long key : tiers.keySet()) keys[i++] = key;
        }
        return new Snapshot(needKeys ? version : latest, keys, full && external != null ? external.keys() : null);
    }
}
