package org.flexlb.balance.scheduler;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;

/**
 * Thread-safe store for inflight {@link InflightItem}s, keyed by string-form
 * request ID.
 *
 * <p>Provides O(1) put / get / remove backed by {@link ConcurrentHashMap}.
 * A background evictor thread periodically removes tombstones (items that
 * have reached a terminal state) after a safety TTL, preventing unbounded
 * growth from leaked entries that were never explicitly removed.
 *
 * <p>The TTL is a safety net — normal operation does NOT remove items on
 * terminal transition. Instead, terminated items remain as tombstones
 * (terminated=true + terminalReason) so that late cancel lookups return
 * false (already terminal) rather than null (not found). The evictor
 * only catches items whose TTL has expired since termination.
 */
public class InflightStore {

    private final ConcurrentMap<String, InflightItem> store = new ConcurrentHashMap<>();
    private final ScheduledExecutorService evictor;

    /** Tombstone retention period: items stay in the store this long after termination. */
    private static final long TTL_MS = 60_000;

    /** Evictor runs at this interval. */
    private static final long EVICT_INTERVAL_MS = 10_000;

    public InflightStore() {
        this.evictor = Executors.newSingleThreadScheduledExecutor(r -> {
            Thread t = new Thread(r, "inflight-evictor");
            t.setDaemon(true);
            return t;
        });
        this.evictor.scheduleAtFixedRate(this::evict,
                EVICT_INTERVAL_MS, EVICT_INTERVAL_MS, TimeUnit.MILLISECONDS);
    }

    public void put(String requestId, InflightItem item) {
        store.put(requestId, item);
    }

    public InflightItem get(String requestId) {
        return store.get(requestId);
    }

    public InflightItem remove(String requestId) {
        return store.remove(requestId);
    }

    public int size() {
        return store.size();
    }

    /**
     * TTL eviction — remove tombstones (terminated items) older than {@link #TTL_MS}.
     *
     * <p>Non-terminated items are never evicted; they represent genuinely inflight
     * requests that must be tracked until a response or timeout is delivered.
     */
    private void evict() {
        long now = System.currentTimeMillis();
        store.forEach((reqId, item) -> {
            if (item.isTerminated() && (now - item.getTerminalTime()) > TTL_MS) {
                store.remove(reqId);
            }
        });
    }

    public void shutdown() {
        evictor.shutdown();
    }
}
