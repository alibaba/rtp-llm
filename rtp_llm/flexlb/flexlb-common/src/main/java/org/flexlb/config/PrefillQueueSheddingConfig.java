package org.flexlb.config;

import lombok.Getter;
import lombok.Setter;

/** Optional overload control for requests waiting for Prefill placement. */
@Getter
@Setter
public final class PrefillQueueSheddingConfig {
    public static final double DEFAULT_MIN_ACCEPTING_WORKER_PERCENT = 10.0;
    /** Actual placement-queue age before soft eviction may start; not predicted work. */
    private long expectedWaitThresholdMs = 30_000L;
    /** Hard request waiting deadline; recovery never extends it. */
    private long maxWaitMs = 60_000L;
    /** Optional relief may reduce actual-age eviction, never trigger it or extend deadlines. */
    private RecoveryConfig recovery;
    /** Gradually withdraw cold-request age exemption after sustained actual queue hard timeouts. */
    private boolean sustainedOverloadProtection = true;
    /** Minimum accepting percentage for cold shedding veto and pool-wide Engine queue relief.
     * Includes capacity-blocked workers in the denominator; does not gate ordinary routing.
     */
    private double minAcceptingWorkerPercent = DEFAULT_MIN_ACCEPTING_WORKER_PERCENT;

    /** Opt-in deadline budgeting with explicitly calibrated uncached throughput.
     * Also preserves maxPendingComputeTokens for cold admission despite low Engine queue reports,
     * keeping excess work globally cancellable during delayed status feedback.
     */
    private DeadlineBudgetConfig deadlineBudget;

    @Getter
    @Setter
    public static final class DeadlineBudgetConfig {
        /** Per-worker aggregate uncached tokens/second; deliberately has no guessed default. */
        private double tokensPerSecond;
        private double capacityFactor = 0.9;
        private double saturatedWorkerPercent = 90.0;
        private long saturationHoldMs = 2_000L;
        /** Leave time for status/dispatch transport and the next control round. */
        private long dispatchSlackMs = 1_000L;
    }

    @Getter
    @Setter
    public static final class RecoveryConfig {
        private long windowMs = 2_000L;
        private long maxGraceMs = 10_000L;
        private long burstProbeMs = 4_000L;
        private double waitMultiplier = 1.5;
        private double maxEvictionFraction = 0.1;
        private double minArrivalDropFraction = 0.2;
        private double minQueueDrainFraction = 0.005;
    }
}
