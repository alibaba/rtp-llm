package org.flexlb.autotpm;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.LongSupplier;

/**
 * Dual-layer preemption rate limiter (guardrail D8): a global 60s fixed
 * window plus a per-endpoint 1s fixed window.
 *
 * <p>Written from scratch to rule out the v2 limiter's boundary bug where a
 * window reset racing an acquire could decrement the global count to -1.
 * Each window packs {@code (windowId << 32) | count} into a single
 * {@link AtomicLong}, so window roll-over and count changes are one CAS —
 * there is no separate reset step to race against, no bare
 * {@code decrementAndGet} anywhere, and a rollback that observes
 * {@code count <= 0} or a stale window returns without touching the state
 * (a count can never go negative).
 *
 * <p>Contract with the preemption orchestrator:
 * <ul>
 *   <li>{@code tryAcquire} takes the global permit first, then the endpoint
 *       permit; an endpoint failure rolls the global permit back</li>
 *   <li>{@code rollback} undoes both layers — called when the cancel turns
 *       out to be a no-op (e.g. engine answered found=false)</li>
 *   <li>{@code globalLimitPerMin <= 0} never admits (guardrail hard-off);
 *       {@code endpointQpsLimit <= 0} means the endpoint layer is unlimited</li>
 * </ul>
 */
public final class PreemptRateLimiter {

    private static final long GLOBAL_WINDOW_MS = 60_000L;
    private static final long ENDPOINT_WINDOW_MS = 1_000L;

    private final int globalLimitPerMin;
    private final int endpointQpsLimit;
    private final LongSupplier clock;

    private final AtomicLong globalState = new AtomicLong(0);
    private final ConcurrentHashMap<String, AtomicLong> endpointStates = new ConcurrentHashMap<>();

    public PreemptRateLimiter(int globalLimitPerMin, int endpointQpsLimit) {
        this(globalLimitPerMin, endpointQpsLimit, System::currentTimeMillis);
    }

    /** Clock-injected variant for deterministic window tests. */
    public PreemptRateLimiter(int globalLimitPerMin, int endpointQpsLimit, LongSupplier clock) {
        this.globalLimitPerMin = globalLimitPerMin;
        this.endpointQpsLimit = endpointQpsLimit;
        this.clock = clock;
    }

    /**
     * Try to take one preemption permit for {@code endpoint}.
     *
     * @return true when both the global and the endpoint window admitted;
     *         false otherwise (no partial state is left behind)
     */
    public boolean tryAcquire(String endpoint) {
        if (globalLimitPerMin <= 0) {
            return false;
        }
        long now = clock.getAsLong();
        if (!tryAcquireWindow(globalState, globalLimitPerMin, GLOBAL_WINDOW_MS, now)) {
            return false;
        }
        if (endpointQpsLimit > 0) {
            AtomicLong endpointState = endpointStates.computeIfAbsent(endpoint, key -> new AtomicLong(0));
            if (!tryAcquireWindow(endpointState, endpointQpsLimit, ENDPOINT_WINDOW_MS, now)) {
                // Endpoint layer refused — return the already-taken global permit.
                rollbackWindow(globalState, GLOBAL_WINDOW_MS, now);
                return false;
            }
        }
        return true;
    }

    /**
     * Return a previously acquired permit on both layers. Safe to call at
     * most once per successful {@code tryAcquire}; a rollback landing after
     * the window rolled over is a no-op (stale-window guard).
     */
    public void rollback(String endpoint) {
        long now = clock.getAsLong();
        rollbackWindow(globalState, GLOBAL_WINDOW_MS, now);
        if (endpointQpsLimit > 0 && endpoint != null) {
            AtomicLong endpointState = endpointStates.get(endpoint);
            if (endpointState != null) {
                rollbackWindow(endpointState, ENDPOINT_WINDOW_MS, now);
            }
        }
    }

    /** Current count in the global window (test/metric visibility). */
    public int globalCount() {
        return countIn(globalState, GLOBAL_WINDOW_MS);
    }

    /** Current count in the endpoint window, 0 if never seen (test/metric visibility). */
    public int endpointCount(String endpoint) {
        AtomicLong endpointState = endpointStates.get(endpoint);
        return endpointState == null ? 0 : countIn(endpointState, ENDPOINT_WINDOW_MS);
    }

    // ==================== packed-state window primitives ====================

    /**
     * CAS loop: read → check window/limit → CAS. A roll-over claims the new
     * window and the first permit in one CAS, so a concurrent acquire in the
     * old window can never be "reset" into a negative count.
     */
    private static boolean tryAcquireWindow(AtomicLong state, int limit, long windowMs, long now) {
        long windowId = now / windowMs;
        while (true) {
            long snapshot = state.get();
            long snapshotWindow = snapshot >>> 32;
            long count = snapshot & 0xFFFFFFFFL;
            if (snapshotWindow != windowId) {
                if (state.compareAndSet(snapshot, (windowId << 32) | 1L)) {
                    return true;
                }
                continue;
            }
            if (count >= limit) {
                return false;
            }
            if (state.compareAndSet(snapshot, (windowId << 32) | (count + 1L))) {
                return true;
            }
        }
    }

    /**
     * CAS loop returning a permit. Returns without touching the state when
     * the count is already 0 (never negative) or the permit's window has
     * expired (stale rollback must not corrupt the fresh window).
     */
    private static void rollbackWindow(AtomicLong state, long windowMs, long now) {
        long windowId = now / windowMs;
        while (true) {
            long snapshot = state.get();
            long snapshotWindow = snapshot >>> 32;
            long count = snapshot & 0xFFFFFFFFL;
            if (snapshotWindow != windowId || count <= 0) {
                return;
            }
            if (state.compareAndSet(snapshot, (windowId << 32) | (count - 1L))) {
                return;
            }
        }
    }

    private int countIn(AtomicLong state, long windowMs) {
        long windowId = clock.getAsLong() / windowMs;
        long snapshot = state.get();
        return (snapshot >>> 32) == windowId ? (int) (snapshot & 0xFFFFFFFFL) : 0;
    }
}
