package org.flexlb.balance.execution;

import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.BiConsumer;
import java.util.function.Predicate;

/**
 * Generic TTL eviction manager for inflight maps across all scheduling layers.
 *
 * <p>Does NOT own the map — works on any {@link Map} whose values
 * implement {@link TtlTracked}. Callers invoke
 * {@link #evictExpired(long, Predicate)} from their own scheduled cleanup.
 *
 * <p>Inflight-leak attribution: every actual eviction is counted and surfaced
 * through a rate-limited (10s) info line so shutdown dumps can be
 * cross-checked against TTL reaping during post-mortems.</p>
 *
 * @param <K> key type
 * @param <V> value type, must implement {@link TtlTracked}
 */
public class TtlEvictor<K, V extends TtlEvictor.TtlTracked> {

    /** Interface required for inflight entries to be evictable by age. */
    public interface TtlTracked {
        /** @return epoch-millis timestamp when this entry was created */
        long createdAtMs();
    }

    private static final AtomicLong EVICTED_TOTAL = new AtomicLong();
    private static final AtomicLong EVICT_LOG_LAST_MS = new AtomicLong();
    private static final long EVICT_LOG_INTERVAL_MS = 10_000L;

    private final Map<K, V> map;
    private final BiConsumer<K, V> onEvict;

    /**
     * Key-aware eviction callback for owners whose secondary accounting is
     * indexed by the same request key.
     */
    public static <K, V extends TtlTracked> TtlEvictor<K, V> withKeyCallback(
            Map<K, V> map, BiConsumer<K, V> onEvict) {
        return new TtlEvictor<>(map, onEvict);
    }

    private TtlEvictor(Map<K, V> map, BiConsumer<K, V> onEvict) {
        this.map = map;
        this.onEvict = onEvict;
    }

    /**
     * Remove expired entries accepted by {@code canEvict}. The caller may use
     * this predicate to protect entries held by a stronger ownership protocol
     * while retaining the generic counter callback.
     */
    public int evictExpired(long ttlMs, Predicate<K> canEvict) {
        long now = System.currentTimeMillis();
        int count = 0;
        for (Map.Entry<K, V> entry : map.entrySet()) {
            if (now - entry.getValue().createdAtMs() > ttlMs
                    && canEvict.test(entry.getKey())) {
                // Use map.remove() instead of iterator.remove() to avoid race with
                // concurrent release()/calibrate() map.remove(key). If another thread
                // already removed the entry, map.remove() returns null and we skip
                // the onEvict callback — preventing double-deduction of counters.
                V candidate = entry.getValue();
                if (map.remove(entry.getKey(), candidate)) {
                    count++;
                    recordEvictionForAttribution(entry.getKey(), candidate, now);
                    if (onEvict != null) {
                        onEvict.accept(entry.getKey(), candidate);
                    }
                }
            }
        }
        return count;
    }

    private static <K, V extends TtlTracked> void recordEvictionForAttribution(
            K key, V candidate, long nowMs) {
        long total = EVICTED_TOTAL.incrementAndGet();
        long last = EVICT_LOG_LAST_MS.get();
        if (nowMs - last >= EVICT_LOG_INTERVAL_MS
                && EVICT_LOG_LAST_MS.compareAndSet(last, nowMs)) {
            org.flexlb.util.Logger.info(
                    "ttl-evict census: evicted_total={} latest_key={} age_ms={}",
                    total, key, Math.max(0L, nowMs - candidate.createdAtMs()));
        }
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
