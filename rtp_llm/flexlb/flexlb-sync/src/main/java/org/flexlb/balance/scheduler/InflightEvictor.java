package org.flexlb.balance.scheduler;

import java.util.Iterator;
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
        for (Iterator<Map.Entry<K, V>> it = map.entrySet().iterator(); it.hasNext(); ) {
            Map.Entry<K, V> entry = it.next();
            if (now - entry.getValue().createdAtMs() > ttlMs) {
                it.remove();
                count++;
                if (onEvict != null) {
                    onEvict.accept(entry.getValue());
                }
            }
        }
        return count;
    }
}
