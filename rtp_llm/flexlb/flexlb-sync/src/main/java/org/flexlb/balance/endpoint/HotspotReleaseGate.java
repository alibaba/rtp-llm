package org.flexlb.balance.endpoint;

/**
 * Integral hotspot release line of one Prefill endpoint. The line is the
 * hit-rate percentage below which a pinned cache-affinity request may drift to
 * balanced ranking: it climbs only while the Engine waiting queue stays above
 * the admission setpoint for a sustained stretch AND the pool still has an
 * idle worker, decays when pressure falls back, and retreats at a fixed rate
 * while the whole pool is saturated, because releasing into a full pool only
 * burns cache hit for compute that cannot complete. Routing threads read the
 * published line; one throttled governor round owned by the strategy performs
 * all updates, so a single writer integrates each sample.
 */
public final class HotspotReleaseGate {

    /** Waiting-queue setpoint, mirroring the Engine queue admission threshold. */
    private static final long WAITING_SETPOINT = 10L;
    /** Sustained-above-setpoint window before the line may climb. */
    private static final long SUSTAINED_MS = 5_000L;
    /** Climb slope in released hit-percent per second per waiting item. */
    private static final double CLIMB_PER_ITEM = 1.2;
    /** Relief decay slope; four times slower than the climb. */
    private static final double DECAY_PER_ITEM = 0.3;
    /** Saturated-pool retreat in released hit-percent per second. */
    private static final double SATURATION_RETREAT = DECAY_PER_ITEM * WAITING_SETPOINT;
    /** A pool that saw no governor round for this long restarts closed. */
    private static final long STALE_MS = 1_000L;

    private volatile double line;
    private long aboveSinceMs = -1L;
    private long lastGovernMs = -1L;

    HotspotReleaseGate() {
    }

    /**
     * Integrate one governor sample. Caller serializes governor rounds;
     * routing threads may read {@link #line()} concurrently. A round whose
     * clock moved backwards (NTP step, test timeline restart) is dropped
     * instead of integrated, so the line never runs the wrong direction.
     */
    void govern(long waiting, boolean poolSpare, long nowMs) {
        long lastMs = lastGovernMs;
        lastGovernMs = nowMs;
        if (lastMs < 0L) {
            return;
        }
        if (nowMs < lastMs) {
            return;
        }
        if (nowMs - lastMs > STALE_MS) {
            line = 0.0;
            aboveSinceMs = -1L;
            return;
        }
        double dtSeconds = (nowMs - lastMs) / 1000.0;
        if (waiting > WAITING_SETPOINT) {
            if (aboveSinceMs < 0L) {
                aboveSinceMs = nowMs;
            }
        } else {
            aboveSinceMs = -1L;
        }
        boolean sustained = aboveSinceMs >= 0L && nowMs - aboveSinceMs >= SUSTAINED_MS;
        if (sustained && poolSpare) {
            line = Math.min(100.0,
                    line + CLIMB_PER_ITEM * (waiting - WAITING_SETPOINT) * dtSeconds);
        } else if (waiting > WAITING_SETPOINT) {
            line = Math.max(0.0, line - SATURATION_RETREAT * dtSeconds);
        } else {
            line = Math.max(0.0, line - DECAY_PER_ITEM * (WAITING_SETPOINT - waiting) * dtSeconds);
        }
    }

    /** Current release threshold in hit-rate percent; 0 while closed. */
    double line() {
        return line;
    }
}
