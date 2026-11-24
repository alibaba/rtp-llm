package org.flexlb.config;

import com.fasterxml.jackson.annotation.JsonIgnore;
import lombok.AccessLevel;
import lombok.Getter;
import lombok.Setter;
import org.flexlb.balance.prediction.DecodeCostFormula;

/**
 * 角色路由配置：流量分组、各角色成本估计与 Prefill 候选排序策略开关。
 * 所有百分比配置均以 {@link #PERCENTAGE_SCALE} 为满刻度。
 */
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
        /** Select Prefill workers by outstanding uncached token work. */
        private TokenBalanceConfig tokenBalance;
        /**
         * Release pinned cache affinity to balanced ranking once a leader's
         * sustained-congestion line climbs. Applies to both the cache-affinity
         * and the token-balance ranking; disabled leaves both unchanged.
         */
        private HotspotReleaseConfig hotspotRelease = new HotspotReleaseConfig();
    }

    /**
     * Integral hotspot release line tuning. The line's zero point is
     * fleet-relative: a worker only counts as congested once its waiting
     * depth exceeds max(setpointFloor, outlierFactor × fleet median depth)
     * for sustainedPressureMs, so large-batch engines with a deep healthy
     * queue stay pinned while small engines surface a shallow-queue
     * hotspot. Climb/decay slopes stay fixed: their 4:1 asymmetry is the
     * anti-oscillation contract and has no business meaning to tune.
     */
    @Getter
    @Setter
    public static final class HotspotReleaseConfig {
        /** Master switch; balanced fleets keep the line closed on their own. */
        private boolean enabled = true;
        /** Fleet-median multiple above which a worker counts as congested; at least 1.0. */
        private double outlierFactor = 1.5;
        /** Sustained-above-setpoint window before the line may climb, in milliseconds. */
        private long sustainedPressureMs = 5_000L;
        /** Absolute floor of the dynamic setpoint; mirror the fleet's queue-admission baseline. */
        private long setpointFloor = 10L;
    }

    /** Prefill selection by outstanding uncached tokens with hot/cold cache affinity. */
    @Getter
    @Setter
    public static final class TokenBalanceConfig {
        /** Above this percentage, the largest reusable prefix wins without a TTFT cap. */
        private double minPrefixHitPercent = 5.0;
        /** Engine waiting-queue depth that blocks admission and dispatch; null disables the gate. */
        private Long engineQueueAdmissionThreshold = 10L;
        /** Sustained-above-threshold waiting depth before a worker counts as congested. */
        private long engineQueueOverloadDurationMs = 5_000L;
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
        private long maxExtraTtftMs;

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
        /** Set to -1 to disable Master-side KV admission, otherwise 1–100 percent. */
        private long maxKvUsagePercent = 90;

        /** Maximum Engine-facing Decode requests, including outstanding dispatch permits. -1 disables this limit. */
        private Long maxEngineRequests = 64L;

        /** Maximum Worker-reported running Decode streams. -1 disables this limit. */
        private long maxRunningStreamSize = -1L;
    }

}
