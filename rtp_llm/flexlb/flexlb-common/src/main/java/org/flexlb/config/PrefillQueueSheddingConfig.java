package org.flexlb.config;

import lombok.Getter;
import lombok.Setter;

/** Optional overload control for requests waiting for Prefill placement. */
@Getter
@Setter
public final class PrefillQueueSheddingConfig {
    /** Actual placement-queue age before soft eviction may start; not predicted work. */
    private long expectedWaitThresholdMs = 30_000L;
    /** Hard request waiting deadline; recovery never extends it. */
    private long maxWaitMs = 60_000L;
    /** Optional relief may reduce actual-age eviction, never trigger it or extend deadlines. */
    private RecoveryConfig recovery;
    /** Gradually withdraw cold-request age exemption after sustained actual queue hard timeouts. */
    private boolean sustainedOverloadProtection = true;

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
