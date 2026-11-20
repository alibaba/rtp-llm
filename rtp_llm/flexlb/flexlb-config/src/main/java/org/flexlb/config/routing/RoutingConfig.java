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
     * Separates Prefill ranking, cache estimates and atomic admission limits.
     */
    @Getter
    @Setter
    public static final class PrefillConfig {
        private double remoteDiscount = 0.2;
        private PrefillSelectionConfig selection = new PrefillSelectionConfig();
        private PrefillAdmissionConfig admission = new PrefillAdmissionConfig();

        @JsonIgnore
        public EngineQueueConfig engineQueueAdmission() {
            if (admission.getEngineQueue() != null) return admission.getEngineQueue();
            return selection.getType() == PrefillSelectionType.TOKEN_BALANCE
                    ? new EngineQueueConfig()
                    : null;
        }
    }

    /**
     * Selects the node-ranking algorithm without changing Endpoint resource accounting.
     */
    @Getter
    @Setter
    public static final class PrefillSelectionConfig {
        private PrefillSelectionType type = PrefillSelectionType.COST_BASED;
        private String costExpression = "computeTokens";
        private CacheAffinityConfig cacheAffinity;
        private double minPrefixHitPercent = 5;
    }

    public enum PrefillSelectionType {
        COST_BASED,
        TOKEN_BALANCE
    }

    /**
     * Bounds outstanding Prefill registrations and optionally accounts for Engine waiting pressure.
     */
    @Getter
    @Setter
    public static final class PrefillAdmissionConfig {
        private int maxInflightRequests = 128;
        private EngineQueueConfig engineQueue;
    }

    /**
     * Configures the Engine waiting watermark for TOKEN_BALANCE.
     */
    @Getter
    @Setter
    public static final class EngineQueueConfig {
        private long waitingThreshold = 10;
    }

    /**
     * Prefers reusable prefixes among nodes that pass the shared admission checks.
     */
    @Getter
    @Setter
    public static final class CacheAffinityConfig {
        private double minPrefixHitPercent = 5;
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
        private volatile String expression = "running_size";

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
