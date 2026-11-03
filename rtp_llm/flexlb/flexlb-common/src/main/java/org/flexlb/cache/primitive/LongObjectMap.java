package org.flexlb.cache.primitive;

import it.unimi.dsi.fastutil.longs.Long2ObjectOpenHashMap;
import it.unimi.dsi.fastutil.HashCommon;
import java.util.Objects;
import java.util.function.LongFunction;

/** Primitive index with guarded reads so resize never exposes a partial table. */
public final class LongObjectMap<V> {
    private static final int SHARDS = 256;
    private final Long2ObjectOpenHashMap<V>[] shards;

    @SuppressWarnings("unchecked")
    public LongObjectMap() {
        shards = (Long2ObjectOpenHashMap<V>[]) new Long2ObjectOpenHashMap<?>[SHARDS];
        for (int index = 0; index < SHARDS; index++) {
            shards[index] = new Long2ObjectOpenHashMap<>();
        }
    }

    private Long2ObjectOpenHashMap<V> shard(long key) {
        return shards[(int) HashCommon.mix(key) & (SHARDS - 1)];
    }

    public V get(long key) {
        var shard = shard(key);
        synchronized (shard) {
            return shard.get(key);
        }
    }

    public V computeIfAbsent(long key, LongFunction<? extends V> factory) {
        var shard = shard(key);
        synchronized (shard) {
            V value = shard.get(key);
            if (value == null) {
                value = Objects.requireNonNull(factory.apply(key));
                shard.put(key, value);
            }
            return value;
        }
    }

    public V remove(long key) {
        var shard = shard(key);
        synchronized (shard) {
            return shard.remove(key);
        }
    }

    public long mappingCount() {
        long size = 0;
        for (var shard : shards) {
            synchronized (shard) {
                size += shard.size();
            }
        }
        return size;
    }

    public void clear() {
        for (var shard : shards) {
            synchronized (shard) {
                shard.clear();
            }
        }
    }
}
