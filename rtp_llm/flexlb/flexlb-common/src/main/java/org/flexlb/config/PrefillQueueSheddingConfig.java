package org.flexlb.config;

import lombok.Getter;
import lombok.Setter;

/** Optional overload control for requests waiting for Prefill placement. */
@Getter
@Setter
public final class PrefillQueueSheddingConfig {
    private long expectedWaitThresholdMs = 30_000L;
    private long maxWaitMs = 60_000L;
    /** Null retains the fixed overload policy. Never changes request deadlines. */
    private RecoveryConfig recovery;

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
