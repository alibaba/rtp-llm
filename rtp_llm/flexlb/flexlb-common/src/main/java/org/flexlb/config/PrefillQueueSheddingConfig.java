package org.flexlb.config;

import lombok.Getter;
import lombok.Setter;

/** Optional overload control for requests waiting for Prefill placement. */
@Getter
@Setter
public final class PrefillQueueSheddingConfig {
    private long expectedWaitThresholdMs = 30_000L;
    private long maxWaitMs = 60_000L;
    /** Arrival/queue adaptation never changes request deadlines. Null is available to programmatic legacy tests. */
    private RecoveryConfig recovery = new RecoveryConfig();

    @Getter
    @Setter
    public static final class RecoveryConfig {
        /** Zero selects automatic observation/budget sizing from the wait SLO. */
        private long windowMs;
        private long maxGraceMs;
        private long burstProbeMs;
        private double waitMultiplier = 1.5;
        private double maxEvictionFraction = 0.1;
        private double minArrivalDropFraction = 0.2;
        private double minQueueDrainFraction = 0.005;
    }
}
