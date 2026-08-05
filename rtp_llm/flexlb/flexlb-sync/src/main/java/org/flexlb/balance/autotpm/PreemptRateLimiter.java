package org.flexlb.balance.autotpm;

import java.util.concurrent.ConcurrentHashMap;

/**
 * Rate limiter for running preemption: enforces both per-node (per decode
 * endpoint) and global QPS limits using a fixed 1-second sliding window.
 *
 * <p>Thread-safe. The per-node window for each endpoint is guarded by
 * synchronizing on the {@link Window} object itself; the global window is
 * guarded by synchronizing on the single global {@link Window} instance.
 *
 * <p>Acquisition order: per-node first, then global. If the global check
 * fails after the per-node slot was tentatively acquired, the per-node
 * counter is rolled back so the slot is not leaked.
 */
public class PreemptRateLimiter {

    private final int perNodeLimit;
    private final int globalLimit;

    private final ConcurrentHashMap<String, Window> nodeWindows = new ConcurrentHashMap<>();
    private final Window globalWindow = new Window();

    /**
     * @param perNodeLimit max preemptions per second per decode endpoint
     * @param globalLimit  max preemptions per second across all endpoints
     */
    public PreemptRateLimiter(int perNodeLimit, int globalLimit) {
        this.perNodeLimit = perNodeLimit;
        this.globalLimit = globalLimit;
    }

    /**
     * Attempt to acquire a preemption slot for the given endpoint.
     *
     * @param endpointKey ip:port of the decode endpoint
     * @return {@code true} if both per-node and global limits allow the preemption
     */
    public boolean tryAcquire(String endpointKey) {
        long now = System.currentTimeMillis();

        // 1. Per-node check (tentative acquire)
        Window nodeWindow = nodeWindows.computeIfAbsent(endpointKey, k -> new Window());
        synchronized (nodeWindow) {
            if (now - nodeWindow.startMs >= 1000L) {
                nodeWindow.startMs = now;
                nodeWindow.count = 0;
            }
            if (nodeWindow.count >= perNodeLimit) {
                return false;
            }
            nodeWindow.count++;
        }

        // 2. Global check
        synchronized (globalWindow) {
            if (now - globalWindow.startMs >= 1000L) {
                globalWindow.startMs = now;
                globalWindow.count = 0;
            }
            if (globalWindow.count >= globalLimit) {
                // Rollback per-node acquisition
                synchronized (nodeWindow) {
                    nodeWindow.count--;
                }
                return false;
            }
            globalWindow.count++;
        }

        return true;
    }

    /** Fixed 1-second window counter. */
    private static final class Window {
        long startMs = System.currentTimeMillis();
        int count = 0;
    }
}
