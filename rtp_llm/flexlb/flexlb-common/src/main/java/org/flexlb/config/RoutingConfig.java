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
    }

    @Getter
    @Setter
    public static final class PrefillConfig {
        private ExecutionTimeEstimatorConfig executionTimeEstimator =
                new ExecutionTimeEstimatorConfig();
        private CacheAffinityConfig cacheAffinity;
        /** Opt-in cache-first routing; null preserves shortest-TTFT behavior. */
        private TokenBalanceConfig tokenBalance;
    }

    @Getter
    @Setter
    public static final class TokenBalanceConfig {
        /** Above this percentage, the largest reusable prefix wins without a TTFT cap. */
        private double minPrefixHitPercent = 5.0;
        /** Actual Engine waiting-stream watermark; not an inflight or unobserved-dispatch quota.
         * Excludes Master queues and running requests. Null disables Engine queue admission.
         * Unknown telemetry imposes no queue gate; explicit concurrency limits still apply.
         */
        private Long engineQueueAdmissionThreshold = 10L;
        /** Consecutive observed time above the watermark before sustained overload; actual waiting gates dispatch immediately. */
        private long engineQueueOverloadDurationMs = 5_000L;
        /** Maximum hotspot migrations per second per Master. Zero disables rescue. */
        private int hotspotRescue;

        public long hotspotRescueIntervalNanos() {
            return hotspotRescue == 0 ? 0L : Math.ceilDiv(1_000_000_000L, hotspotRescue);
        }
    }

    @Getter
    @Setter
    public static final class ExecutionTimeEstimatorConfig {
        private EstimatorType type = EstimatorType.FORMULA;
        private String expression =
                "sum(computeTokens) + 0.02*sum(hitCacheTokens)";
    }

    public enum EstimatorType {
        FORMULA,
        LEARNING
    }

    @Getter
    @Setter
    public static final class CacheAffinityConfig {
        /** Maximum TTFT penalty accepted for choosing a cache leader. */
        private volatile long maxExtraTtftMs;

        /** Minimum reusable-prefix percentage required before affinity applies. */
        private double minPrefixHitPercent = 5;
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

    @Getter
    @Setter
    public static final class DecodeAvailabilityConfig {
        /** Maximum Decode KV usage percentage, in [1, 100]. */
        private long maxKvUsagePercent = 90;

        /** Master-side cap for all Engine-facing Decode ownership. */
        private Long maxEngineRequests;
    }

}
