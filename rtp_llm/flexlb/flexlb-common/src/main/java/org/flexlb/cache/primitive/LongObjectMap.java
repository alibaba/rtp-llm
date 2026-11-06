package org.flexlb.cache.primitive;

import it.unimi.dsi.fastutil.HashCommon;
import it.unimi.dsi.fastutil.longs.Long2ObjectOpenHashMap;

import java.util.Objects;
import java.util.function.Function;
import java.util.function.LongFunction;
import java.util.function.UnaryOperator;

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
        // The table uses the low bits of the same hash for its initial slot.
        // Select shards from the high bits to preserve slot diversity inside each table.
        return shards[(int) (HashCommon.mix(key) >>> 56)];
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

    /** Atomically mutate an entry; null removes it. Callbacks must not reenter this map. */
    public void compute(long key, UnaryOperator<V> update) {
        var shard = shard(key);
        synchronized (shard) {
            V previous = shard.get(key);
            V next = update.apply(previous);
            if (next == null) {
                shard.remove(key);
            } else if (next != previous) {
                shard.put(key, next);
            }
        }
    }

    /** Copy/project a value while protected; do not return a mutable stored value. */
    public <R> R read(long key, Function<? super V, R> snapshot) {
        var shard = shard(key);
        synchronized (shard) {
            return snapshot.apply(shard.get(key));
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
        clear(() -> {});
    }

    /** Clear and reset associated accounting atomically with respect to mutations. */
    public void clear(Runnable resetAccounting) {
        clearLocked(0, resetAccounting);
    }

    private void clearLocked(int index, Runnable resetAccounting) {
        if (index == SHARDS) {
            for (var shard : shards) shard.clear();
            resetAccounting.run();
            return;
        }
        // Fixed order; ordinary operations acquire only one shard.
        synchronized (shards[index]) {
            clearLocked(index + 1, resetAccounting);
        }
    }
}
