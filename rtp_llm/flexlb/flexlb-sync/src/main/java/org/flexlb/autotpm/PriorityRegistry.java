package org.flexlb.autotpm;

import java.util.concurrent.ConcurrentHashMap;

/**
 * Master-side requestId → Auto-TPM priority registry.
 *
 * <p>The engine never reports priority, so victim selection needs a local
 * record of what priority each dispatched request carries. Entries are
 * registered by {@code BatchScheduler#submit} right after the request wins
 * the {@code InflightStore} registration, and removed when the request's
 * result future settles (any terminal state — the future completion is the
 * exactly-once terminal signal in the v2 scheduler).
 *
 * <p>Unregistered requests (other masters' traffic, non-BATCH paths) simply
 * have no entry and are therefore never eligible as preemption victims.
 */
public final class PriorityRegistry {

    private final ConcurrentHashMap<Long, Integer> priorities = new ConcurrentHashMap<>();

    /** Record the priority a request was dispatched with. */
    public void register(long requestId, int priority) {
        priorities.put(requestId, priority);
    }

    /** Drop the entry once the request reaches a terminal state. Idempotent. */
    public void remove(long requestId) {
        priorities.remove(requestId);
    }

    /**
     * @return the registered priority, or {@code null} when this master never
     *         registered the request (→ not eligible for victim selection)
     */
    public Integer priorityOf(long requestId) {
        return priorities.get(requestId);
    }

    /** Number of tracked (non-terminal) registrations. */
    public int size() {
        return priorities.size();
    }
}
