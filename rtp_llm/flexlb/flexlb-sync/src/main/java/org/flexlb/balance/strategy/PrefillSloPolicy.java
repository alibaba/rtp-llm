package org.flexlb.balance.strategy;

import org.flexlb.balance.projection.RouteProjection;

import static com.google.common.math.LongMath.saturatedAdd;

/** TTFT admission in milliseconds. Counts local plans before they appear in Engine reports. */
public final class PrefillSloPolicy {
    public static long executionBudgetMs(long sloMs) { return Math.max(1, sloMs - sloMs / 10); }

    public static boolean admits(long sloMs, long createdAtMs, long nowMs, long projectedMs,
                                 long plannedMs, RouteProjection.Inputs inputs, boolean idle) {
        if (projectedMs < 0 || inputs.work().hasUnknownWork()) { return false; }
        long elapsed = Math.max(0, nowMs - createdAtMs);
        long remaining = executionBudgetMs(sloMs) - elapsed;
        // FIFO prevents overtaking. A new batch member must also leave all existing queued
        // members within their original target, even if grouping changes their execution time.
        for (var item : inputs.queue().activeItems()) {
            long originalTarget = item.expiresAtMs() - sloMs; // queue deadline = arrival + 2*SLO
            remaining = Math.min(remaining, originalTarget - nowMs - sloMs / 10);
        }
        long predicted = saturatedAdd(projectedMs, plannedMs);
        if (predicted <= remaining) { return true; }
        // A free worker may drain a burst that has missed its target, or obtain its first
        // calibration sample. This never renews the original finite queue deadline.
        return idle && predicted <= saturatedAdd(sloMs, sloMs) - elapsed;
    }

    public static long collectionWindowMs(long maximumMs, long budgetMs, long serviceMs) {
        // Keep one clock tick for very short requests; integer truncation must not disable batching.
        return Math.min(maximumMs, Math.min(Math.max(1, serviceMs / 10), Math.max(0, budgetMs - serviceMs)));
    }

    private PrefillSloPolicy() { }
}
