package org.flexlb.config;

import com.fasterxml.jackson.annotation.JsonIgnore;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.Setter;
import org.flexlb.balance.prediction.DecodeCostFormula;

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
        private VitConfig vit = new VitConfig();
    }

    @Getter
    @Setter
    public static final class VitConfig {
        /** Default: least media work. When enabled, cache score takes precedence. */
        private boolean cacheAffinityEnabled;
        /** Unknown/expired load must never be interpreted as an idle worker. */
        private long loadMaxAgeMs = 5000;
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

        /**
         * Prefix-hit percentage at or above which the cache leader is preferred
         * regardless of the projected-TTFT cutoff. High-affinity requests carry
         * nearly the whole conversation prefix; losing that prefix on a routing
         * flip costs a full re-prefill, which dominates any modeled TTFT gap.
         * Set to a value above {@link RoutingConfig#PERCENTAGE_SCALE} to disable.
         */
        private double stickyPrefixHitPercent = 90;

        /**
         * Estimated prefix-hit percentage floor at which the endpoint holding
         * the MAXIMUM estimated hit is preferred regardless of the projected-TTFT
         * cutoff. The per-request hit estimate is block-granular and
         * refresh-batched, so a request whose true reusable prefix covers most of
         * its length can still estimate slightly below
         * {@code stickyPrefixHitPercent}; flipping such a request away from the
         * unique max-hit endpoint forfeits the whole cached prefix, which costs
         * far more than any modeled TTFT gap. The floor only recognizes the
         * strict maximum-hit endpoint, never partial-match runners-up.
         * Set to a value above {@link RoutingConfig#PERCENTAGE_SCALE} to disable.
         */
        private double stickyLeaderFloorPercent = 50;
    }

    @Getter
    @Setter
    public static final class DecodeConfig {
        private DecodeCostEstimatorConfig costEstimator = new DecodeCostEstimatorConfig();
        private DecodeAvailabilityConfig availability =
                new DecodeAvailabilityConfig();
    }

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

    @Getter
    @Setter
    public static final class DecodeAvailabilityConfig {
        /** Maximum Decode KV usage percentage, in [1, 100]. */
        private long maxKvUsagePercent = 90;

        /** Master-side cap for all Engine-facing Decode ownership. */
        private Long maxEngineRequests;
    }

}
