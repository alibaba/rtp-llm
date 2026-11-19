package org.flexlb.cache.primitive;

import it.unimi.dsi.fastutil.longs.Long2BooleanMap;
import it.unimi.dsi.fastutil.longs.Long2BooleanOpenHashMap;
import it.unimi.dsi.fastutil.longs.LongSet;

/**
 * Primitive long/boolean storage with bounded sizing from recent full snapshots.
 */
public final class PackedBooleanCacheMap extends Long2BooleanOpenHashMap {
    private static final int MAX_EXPECTED_KEYS = 4_000_000;

    private static volatile int recentSnapshotSize;

    public PackedBooleanCacheMap() {
        super(recentSnapshotSize);
    }

    @Override
    public int size() {
        observeSize();
        return super.size();
    }

    @Override
    public LongSet keySet() {
        observeSize();
        return super.keySet();
    }

    @Override
    public Long2BooleanMap.FastEntrySet long2BooleanEntrySet() {
        observeSize();
        return super.long2BooleanEntrySet();
    }

    private void observeSize() {
        if (size > 0) {
            recentSnapshotSize = Math.min(size, MAX_EXPECTED_KEYS);
        }
    }
}
