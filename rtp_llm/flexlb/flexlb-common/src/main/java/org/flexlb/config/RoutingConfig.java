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
        /** Detect congested cache leaders and pace redirects across this Master's Prefill pools. */
        private HotspotReleaseConfig hotspotRelease = new HotspotReleaseConfig();
    }

    /** Fleet-relative congestion detection and Master-wide hotspot redirect rate. */
    @Getter
    @Setter
    public static final class HotspotReleaseConfig {
        /** Enable redirects from a congested cache leader. */
        private boolean enabled = true;
        /** Ranking among eligible receivers after a hotspot has been detected. */
        private HotspotTargetPolicy targetPolicy = HotspotTargetPolicy.ROUTING_STRATEGY;
        /** Which waiting requests to move from a hot Prefill queue. */
        private QueuedRequestPolicy queuedRequestPolicy = QueuedRequestPolicy.DISABLED;
        /** Maximum queued migration attempts per request. */
        private int maxQueuedMigrationAttempts = 1;
        /** Fleet-median multiple above which a worker counts as congested; at least 1.0. */
        private double outlierFactor = 1.5;
        /** Time above the pool's congestion threshold before declaring a hotspot. */
        private long sustainedPressureMs = 0L;
        /** Maximum source workers selected per pool in a one-second window. Zero includes all hotspots. */
        private int sourceTopK = 5;
        /** Additional uncached Prefill tokens allowed per thousand baseline compute tokens. */
        private int maxExtraComputePermille = 10;
        /** Minimum outstanding-request threshold for the fleet-relative comparison. */
        private long setpointFloor = 10L;
        /** Minimum compute-token threshold, in tokens, independent of the outstanding-request floor. */
        private long computeTokenFloor = 1000L;
    }

    public enum HotspotTargetPolicy {
        ROUTING_STRATEGY,
        MIN_PENDING_COMPUTE,
        MIN_PENDING_COMPUTE_WITHIN_ONE_BLOCK
    }

    public enum QueuedRequestPolicy {
        DISABLED,
        HIGHEST_HIT_RATE_FIRST,
        LOWEST_HIT_RATE_FIRST,
        /** Always choose the minimum source prefix hit ratio among eligible queued requests. */
        MIN_SOURCE_CACHE_HIT_RATE
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
