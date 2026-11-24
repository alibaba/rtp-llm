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
    /** Retry a cold eviction candidate when at least this percentage of workers can accept placement
     * according to their Engine waiting-queue gates. Disabled and unknown gates count as accepting. */
    private double minAcceptingWorkerPercent = DEFAULT_MIN_ACCEPTING_WORKER_PERCENT;

    /** Opt-in deadline budgeting with explicitly calibrated uncached throughput. */
    private DeadlineBudgetConfig deadlineBudget;

    @Getter
    @Setter
    public static final class DeadlineBudgetConfig {
        /** Per-worker aggregate uncached tokens/second; deliberately has no guessed default. */
        private double tokensPerSecond;
        private double capacityFactor = 0.9;
        /** Minimum percentage of the pool with a blocked Engine queue gate to start the saturation timer. */
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
