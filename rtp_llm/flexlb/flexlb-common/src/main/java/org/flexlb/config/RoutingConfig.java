package org.flexlb.config;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public final class RoutingConfig {

    private long availabilityHysteresisPercent = 15;
    private volatile TrafficPolicyConfig groupSelector;
    private RolesConfig roles = new RolesConfig();

    @Getter
    @Setter
    public static final class RolesConfig {
        private PrefillConfig prefill = new PrefillConfig();
        private DecodeConfig decode = new DecodeConfig();
    }

    @Getter
    @Setter
    public static final class PrefillConfig {
        private PrefillAvailabilityConfig availability =
                new PrefillAvailabilityConfig();
        private ExecutionTimeEstimatorConfig executionTimeEstimator =
                new ExecutionTimeEstimatorConfig();
        private CandidateChoiceConfig candidateChoice =
                new CandidateChoiceConfig();
        private CacheAffinityConfig cacheAffinity;
    }

    @Getter
    @Setter
    public static final class PrefillAvailabilityConfig {
        private long maxPendingRequests = 64;
    }

    @Getter
    @Setter
    public static final class ExecutionTimeEstimatorConfig {
        private EstimatorType type = EstimatorType.FORMULA;
        private String expression =
                "sum(computeTokens) + 0.3*sum(hitCacheTokens)";
    }

    public enum EstimatorType {
        FORMULA,
        LEARNING
    }

    @Getter
    @Setter
    public static final class CandidateChoiceConfig {
        private CandidateChoiceType type =
                CandidateChoiceType.RANDOM_WITHIN_TOLERANCE;
        private double relativeTolerance = 0.1;
        private long minimumToleranceMs = 20;
        private OutlierRejectionConfig outlierRejection =
                new OutlierRejectionConfig();
        private CandidatePoolConfig pool = new CandidatePoolConfig();
    }

    public enum CandidateChoiceType {
        BEST_ONLY,
        RANDOM_WITHIN_TOLERANCE,
        LEAST_RECENTLY_USED_IN_POOL
    }

    @Getter
    @Setter
    public static final class OutlierRejectionConfig {
        private double maxPendingVsAverageMultiplier = 3.0;
        private double maxProjectedDrainVsAverageMultiplier = 3.0;
    }

    @Getter
    @Setter
    public static final class CandidatePoolConfig {
        private CandidatePoolType type = CandidatePoolType.RATIO;
        private double ratio = 0.3;
        private int minimumWorkers = 1;
        private int workers = 1;
    }

    public enum CandidatePoolType {
        RATIO,
        FIXED
    }

    @Getter
    @Setter
    public static final class CacheAffinityConfig {
        /** Maximum TTFT penalty accepted for choosing a cache leader. */
        private long maxExtraTtftMs;

        /** Minimum reusable-prefix percentage required before affinity applies. */
        private double minPrefixHitPercent = 5;
    }

    @Getter
    @Setter
    public static final class DecodeConfig {
        private DecodeAvailabilityConfig availability =
                new DecodeAvailabilityConfig();
        private KvReservationConfig kvReservation = new KvReservationConfig();
        private double decayPerToken = 0.001;
        private DecodeOutlierRejectionConfig outlierRejection =
                new DecodeOutlierRejectionConfig();
    }

    @Getter
    @Setter
    public static final class DecodeAvailabilityConfig {
        private long maxKvUsagePercent = 90;

        /** Master-side cap for all Engine-facing Decode ownership. */
        private Long maxEngineRequests;
    }

    @Getter
    @Setter
    public static final class KvReservationConfig {
        private Long maxOutputTokensForEstimate = 1000L;
    }

    @Getter
    @Setter
    public static final class DecodeOutlierRejectionConfig {
        private double maxEngineLoadVsAverageMultiplier = 3.0;
        private double maxKvUsedVsAverageMultiplier = 3.0;
    }

}
