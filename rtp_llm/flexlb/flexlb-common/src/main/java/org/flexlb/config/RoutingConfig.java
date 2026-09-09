package org.flexlb.config;

import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public final class RoutingConfig {

    /** Multiplicative scale for all percentage-valued routing settings. */
    public static final double PERCENTAGE_SCALE = 100.0;

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
        private ExecutionTimeEstimatorConfig executionTimeEstimator =
                new ExecutionTimeEstimatorConfig();
        private CacheAffinityConfig cacheAffinity;
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
    }

    @Getter
    @Setter
    public static final class DecodeAvailabilityConfig {
        private long maxKvUsagePercent = 90;

        /** Master-side cap for all Engine-facing Decode ownership. */
        private Long maxEngineRequests;
    }

}
