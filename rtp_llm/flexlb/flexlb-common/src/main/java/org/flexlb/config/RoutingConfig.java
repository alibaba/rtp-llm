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
        private double minPrefixHitPercent = 50.0;
        /** Token backpressure when Engine queue admission is disabled or telemetry is unknown. */
        private long maxPendingComputeTokens = 2_000_000L;
        /** Include recently completed uncached tokens in routing scores. */
        private boolean enableWindow = false;
        private long windowMs = 10_000L;
        /** Actual Engine waiting-stream watermark; not an inflight or unobserved-dispatch quota.
         * Excludes Master queues and running requests. Null retains legacy admission limits.
         */
        private Long engineQueueAdmissionThreshold = 10L;
        /** Consecutive observed time above the watermark before sustained overload; actual waiting gates dispatch immediately. */
        private long engineQueueOverloadDurationMs = 5_000L;
        /** Optional rescue of small, aged requests to lightly loaded cache-eligible workers. */
        private HotspotRescueConfig hotspotRescue;
    }

    @Getter
    @Setter
    public static final class HotspotRescueConfig {
        /** Migrate only when the source Engine reports more waiting streams than this count. */
        private long engineQueueMigrationThreshold = 30L;
        private long minWaitMs = 5_000L;
        /** Total input, not just uncached tokens: a cache miss on the target must stay cheap. */
        private long maxInputTokens = 32_768L;
        /** Target's outstanding uncached tokens (pending + planned), excluding completed history. */
        private long maxIdlePendingTokens = 0L;
        /** Minimum interval between local queue migration rounds; unused rounds never accumulate. */
        private long migrationIntervalMs = 1_000L;
        /** Maximum withdrawals per round, shared by all workers in this model scheduler. */
        private int maxMigrationsPerRound = 1;
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
