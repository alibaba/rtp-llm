package org.flexlb.config.routing;

import com.fasterxml.jackson.annotation.JsonIgnore;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.Setter;
import org.flexlb.util.formula.DecodeCostFormula;

/**
 * Groups role-selection policies and the traffic rules used to choose a routing group.
 */
@Getter
@Setter
public final class RoutingConfig {

    /**
     * Multiplicative scale for all percentage-valued routing settings.
     */
    public static final double PERCENTAGE_SCALE = 100.0;

    private volatile TrafficPolicyConfig groupSelector;
    private RolesConfig roles = new RolesConfig();

    /**
     * Holds the independent Prefill and Decode selection policies.
     */
    @Getter
    @Setter
    public static final class RolesConfig {
        private PrefillConfig prefill = new PrefillConfig();
        private DecodeConfig decode = new DecodeConfig();
        private EncoderConfig encoder = new EncoderConfig();
    }

    /**
     * Limits requests registered against each Encoder worker.
     */
    @Getter
    @Setter
    public static final class EncoderConfig {
        private int maxRequests = Integer.MAX_VALUE;
    }

    /**
     * Limits each Prefill worker's compute-token work and configures cache-first selection.
     */
    @Getter
    @Setter
    public static final class PrefillConfig {
        private long maxComputeTokens = 1_048_576L;
        private String costExpression = "computeTokens";
        private CacheAffinityConfig cacheAffinity;
    }

    /**
     * Prefers cache hits among workers whose resulting compute-token work fits the configured cap.
     */
    @Getter
    @Setter
    public static final class CacheAffinityConfig {
        /**
         * Absolute worker work after registration; null inherits Prefill maxComputeTokens.
         */
        private Long maxComputeTokens;

        /**
         * Minimum reusable-prefix percentage required before affinity applies.
         */
        private double minPrefixHitPercent = 5;

        /**
         * Credit applied to remote-source cache matches when estimating effective hit tokens.
         */
        private double remoteDiscount = 0.2;
    }

    /**
     * Combines Decode cost ranking with the independent admission limits.
     */
    @Getter
    @Setter
    public static final class DecodeConfig {
        private DecodeCostEstimatorConfig costEstimator = new DecodeCostEstimatorConfig();
        private DecodeAvailabilityConfig availability = new DecodeAvailabilityConfig();
    }

    /**
     * Caches the compiled Decode cost expression and refreshes it when the expression changes.
     */
    @Getter
    @Setter
    public static final class DecodeCostEstimatorConfig {
        private volatile String expression = "kvcache_used_ratio";

        @JsonIgnore
        @Getter(AccessLevel.NONE)
        @Setter(AccessLevel.NONE)
        private volatile DecodeCostFormula compiledCost;

        @JsonIgnore
        public DecodeCostFormula compiledFormula() {
            String currentExpression = expression;
            DecodeCostFormula cached = compiledCost;
            if (cached != null && cached.expression().equals(currentExpression)) {
                return cached;
            }
            synchronized (this) {
                currentExpression = expression;
                cached = compiledCost;
                if (cached == null || !cached.expression().equals(currentExpression)) {
                    cached = DecodeCostFormula.parse(currentExpression);
                    compiledCost = cached;
                }
                return cached;
            }
        }
    }

    /**
     * Limits Decode KV usage percentage and optional concurrent engine requests.
     */
    @Getter
    @Setter
    public static final class DecodeAvailabilityConfig {
        /**
         * Maximum Decode KV usage percentage, in [1, 100].
         */
        private long maxKvUsagePercent = 90;

        /**
         * Master-side cap for all Engine-facing Decode ownership.
         */
        private Long maxEngineRequests;
    }
}
