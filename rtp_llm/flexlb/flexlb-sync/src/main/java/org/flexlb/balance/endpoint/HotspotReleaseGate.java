package org.flexlb.balance.endpoint;

/**
 * Integral hotspot release line of one Prefill endpoint. The line is the
 * hit-rate percentage below which a pinned cache-affinity request may drift to
 * balanced ranking: it climbs only while the combined router and Engine queue stays above
 * the fleet-relative setpoint for a sustained stretch AND the pool still has an
 * idle worker, decays when pressure falls back, and retreats while the whole
 * pool is saturated, because releasing into a full pool only burns cache hit
 * for compute that cannot complete. The caller computes the setpoint
 * (max(floor, outlierFactor x fleet-median waiting depth)) and passes it per
 * round; routing threads read the published line, and one throttled governor
 * round owned by the strategy performs all updates, so a single writer
 * integrates each sample.
 */
public final class HotspotReleaseGate {

    /** Climb slope in released hit-percent per second per waiting item above the setpoint. */
    private static final double CLIMB_PER_ITEM = 1.2;
    /** Relief decay slope per waiting item below the setpoint; four times slower than the climb. */
    private static final double DECAY_PER_ITEM = 0.3;
    /** A governor round gap beyond this bound restarts the line closed. */
    private static final long STALE_MS = 1_000L;

    private final long sustainedPressureMs;
    private volatile double line;
    private long aboveSinceMs = -1L;
    private long lastGovernMs = -1L;

    HotspotReleaseGate(long sustainedPressureMs) {
        this.sustainedPressureMs = sustainedPressureMs;
    }

    /**
     * Integrate one governor sample against the round's setpoint. Caller
     * serializes governor rounds; routing threads may read {@link #line()}
     * concurrently. A round whose clock moved backwards (NTP step, test
     * timeline restart) is dropped instead of integrated, so the line never
     * runs the wrong direction.
     */
    void govern(long waiting, long setpoint, boolean poolSpare, long nowMs) {
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
        if (waiting > setpoint) {
            if (aboveSinceMs < 0L) {
                aboveSinceMs = nowMs;
            }
        } else {
            aboveSinceMs = -1L;
        }
        boolean sustained = aboveSinceMs >= 0L
                && nowMs - aboveSinceMs >= sustainedPressureMs;
        if (sustained && poolSpare) {
            line = Math.min(100.0,
                    line + CLIMB_PER_ITEM * (waiting - setpoint) * dtSeconds);
        } else if (waiting > setpoint) {
            line = Math.max(0.0, line - DECAY_PER_ITEM * setpoint * dtSeconds);
        } else {
            line = Math.max(0.0,
                    line - DECAY_PER_ITEM * (setpoint - waiting) * dtSeconds);
        }
    }

    /** Current release threshold in hit-rate percent; 0 while closed. */
    double line() {
        return line;
    }
}
