package org.flexlb.balance.strategy;

import org.flexlb.dao.route.RoleType;

import java.util.HashMap;
import java.util.Map;

/** Pool-wide evidence for global shedding, separate from worker admission limits. */
final class PrefillQueuePressure {
    private static final long HOLD_MS = 10_000L;
    private record Pool(RoleType role, String group) { }
    private final Map<Pool, Long> waitingSinceMs = new HashMap<>();

    synchronized boolean observe(RoleType role, String group, int workers, int waiting, long nowMs) {
        var pool = new Pool(role, group);
        if (workers <= 0 || waiting <= workers / 2) {
            waitingSinceMs.remove(pool);
            return false;
        }
        long since = waitingSinceMs.computeIfAbsent(pool, ignored -> nowMs);
        if (nowMs < since) { waitingSinceMs.put(pool, nowMs); return false; }
        return nowMs - since >= HOLD_MS;
    }
}
