package org.flexlb.balance.dp;

import org.flexlb.dao.master.WorkerStatus;

import java.util.List;

/**
 * Plug-point for "which DP group should this batch go to".
 *
 * <p>Called by {@link DispatchPlanner} once per planned batch (NOT per request)
 * — the group decision is amortised across {@code dpSize} requests, so the
 * selector can afford to look at richer signals (per-rank queue depth from
 * {@code WorkerStatus.dpStatuses}, prefix-cache match from
 * {@code WorkerStatus.cacheStatus.dpCaches}, length-aware bin packing, …).
 *
 * <p>V1 ships {@link RoundRobinGroupSelector}. Future strategies plug in by
 * implementing this interface — no changes needed in the planner or batcher.
 *
 * <p>Contract:
 * <ul>
 *   <li>{@code candidates} is pre-filtered: alive, {@code dp_size > 1}, resource available.</li>
 *   <li>Implementations MUST be thread-safe (one batcher thread + size-trigger callers).</li>
 *   <li>Returning {@code null} signals "no candidate suitable" — the planner will
 *       fail every request in the batch with {@code NO_PREFILL_WORKER}.</li>
 * </ul>
 */
public interface GroupSelector {

    WorkerStatus select(List<WorkerStatus> candidates, BatchHint hint);

    /** Stable identifier used by config to pick a strategy. */
    String name();
}
