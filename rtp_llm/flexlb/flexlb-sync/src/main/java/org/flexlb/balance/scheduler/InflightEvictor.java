package org.flexlb.balance.scheduler;

import java.util.Map;
import java.util.function.Consumer;

/**
 * Generic TTL eviction manager for inflight maps across all scheduling layers.
 *
 * <p>Does NOT own the map — works on any {@link Map} whose values
 * implement {@link TtlTracked}. Callers invoke {@link #evictExpired(long)} from
 * their own {@code @Scheduled} cleanup methods.
 *
 * @param <K> key type
 * @param <V> value type, must implement {@link TtlTracked}
 */
public class InflightEvictor<K, V extends InflightEvictor.TtlTracked> {

    /** Interface required for inflight entries to be evictable by age. */
    public interface TtlTracked {
        /** @return epoch-millis timestamp when this entry was created */
        long createdAtMs();
    }

    private final Map<K, V> map;
    private final Consumer<V> onEvict;

    /**
     * @param map     the map to evict from (not owned by this evictor)
     * @param onEvict called for each evicted entry (e.g. to adjust counters);
     *                may be null if no side effects are needed
     */
    public InflightEvictor(Map<K, V> map, Consumer<V> onEvict) {
        this.map = map;
        this.onEvict = onEvict;
    }

    /**
     * Remove all entries older than {@code ttlMs} milliseconds.
     *
     * @param ttlMs max age before eviction
     * @return number of entries evicted
     */
    public int evictExpired(long ttlMs) {
        long now = System.currentTimeMillis();
        int count = 0;
        for (Map.Entry<K, V> entry : map.entrySet()) {
            if (now - entry.getValue().createdAtMs() > ttlMs) {
                // Use map.remove() instead of iterator.remove() to avoid race with
                // concurrent release()/calibrate() map.remove(key). If another thread
                // already removed the entry, map.remove() returns null and we skip
                // the onEvict callback — preventing double-deduction of counters.
                V removed = map.remove(entry.getKey());
                if (removed != null) {
                    count++;
                    if (onEvict != null) {
                        onEvict.accept(removed);
                    }
                }
            }
        }
        return count;
    }

    /**
     * Compute the age (ms) of the oldest entry in the map, or 0 if empty.
     */
    public static <K, V extends TtlTracked> long maxAgeMs(Map<K, V> map, long nowMs) {
        long oldest = Long.MAX_VALUE;
        for (V v : map.values()) {
            oldest = Math.min(oldest, v.createdAtMs());
        }
        return oldest == Long.MAX_VALUE ? 0 : Math.max(0, nowMs - oldest);
    }
}
