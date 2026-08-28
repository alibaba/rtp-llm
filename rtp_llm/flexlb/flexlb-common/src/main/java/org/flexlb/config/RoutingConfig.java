package org.flexlb.config;

import com.fasterxml.jackson.annotation.JsonSubTypes;
import com.fasterxml.jackson.annotation.JsonTypeInfo;
import lombok.Getter;
import lombok.Setter;
import org.flexlb.dao.route.RoleType;

@Getter
@Setter
public final class RoutingConfig {

    private long availabilityHysteresisPercent = 15;
    private volatile TrafficPolicyConfig groupSelector;
    private RolesConfig roles = new RolesConfig();

    public EndpointSelectorConfig selectorFor(RoleType role) {
        return switch (role) {
            case PREFILL, PDFUSION -> roles.getPrefill().getSelector();
            case DECODE -> roles.getDecode().getSelector();
            case VIT -> roles.getVit().getSelector();
            case FRONTEND -> throw new IllegalArgumentException(
                    "FRONTEND has no endpoint selector");
        };
    }

    /** Closed root for every endpoint-selection configuration variant. */
    public sealed interface EndpointSelectorConfig
            permits PrefillSelectorConfig,
                    DecodeSelectorConfig,
                    VitSelectorConfig {
    }

    @Getter
    @Setter
    public static final class RolesConfig {
        private PrefillConfig prefill = new PrefillConfig();
        private DecodeConfig decode = new DecodeConfig();
        private VitConfig vit = new VitConfig();
    }

    @Getter
    @Setter
    public static final class PrefillConfig {
        private PrefillAvailabilityConfig availability = new PrefillAvailabilityConfig();
        private ExecutionTimeEstimatorConfig executionTimeEstimator = new FormulaEstimatorConfig();
        private PrefillSelectorConfig selector = new EstimatedTtftSelectorConfig();
        private CacheAffinityConfig cacheAffinity;

    }

    @Getter
    @Setter
    public static final class PrefillAvailabilityConfig {
        private long maxPendingRequests = 64;
    }

    @JsonTypeInfo(use = JsonTypeInfo.Id.NAME, property = "type")
    @JsonSubTypes({
            @JsonSubTypes.Type(value = FormulaEstimatorConfig.class, name = "FORMULA"),
            @JsonSubTypes.Type(value = LearningEstimatorConfig.class, name = "LEARNING")
    })
    public sealed interface ExecutionTimeEstimatorConfig
            permits FormulaEstimatorConfig, LearningEstimatorConfig {
    }

    @Getter
    @Setter
    public static final class FormulaEstimatorConfig implements ExecutionTimeEstimatorConfig {
        private String expression = "sum(computeTokens) + 0.3*sum(hitCacheTokens)";
    }

    public static final class LearningEstimatorConfig implements ExecutionTimeEstimatorConfig {
    }

    @JsonTypeInfo(use = JsonTypeInfo.Id.NAME, property = "type")
    @JsonSubTypes({
            @JsonSubTypes.Type(value = RandomPrefillSelectorConfig.class, name = "RANDOM"),
            @JsonSubTypes.Type(value = EstimatedTtftSelectorConfig.class, name = "ESTIMATED_TTFT")
    })
    public sealed interface PrefillSelectorConfig
            extends EndpointSelectorConfig
            permits RandomPrefillSelectorConfig, EstimatedTtftSelectorConfig {
    }

    public static final class RandomPrefillSelectorConfig implements PrefillSelectorConfig {
    }

    @Getter
    @Setter
    public static final class EstimatedTtftSelectorConfig implements PrefillSelectorConfig {
        private CandidateChoiceConfig candidateChoice = new RandomWithinToleranceConfig();
    }

    @Getter
    @Setter
    public static final class OutlierRejectionConfig {
        private double maxPendingVsAverageMultiplier = 3.0;
        private double maxProjectedDrainVsAverageMultiplier = 3.0;
    }

    @JsonTypeInfo(use = JsonTypeInfo.Id.NAME, property = "type")
    @JsonSubTypes({
            @JsonSubTypes.Type(value = BestOnlyConfig.class, name = "BEST_ONLY"),
            @JsonSubTypes.Type(value = RandomWithinToleranceConfig.class,
                    name = "RANDOM_WITHIN_TOLERANCE"),
            @JsonSubTypes.Type(value = LeastRecentlyUsedInPoolConfig.class,
                    name = "LEAST_RECENTLY_USED_IN_POOL")
    })
    public sealed interface CandidateChoiceConfig
            permits BestOnlyConfig, RandomWithinToleranceConfig, LeastRecentlyUsedInPoolConfig {
    }

    @Getter
    @Setter
    public static final class BestOnlyConfig implements CandidateChoiceConfig {
        private OutlierRejectionConfig outlierRejection = new OutlierRejectionConfig();
    }

    @Getter
    @Setter
    public static final class RandomWithinToleranceConfig implements CandidateChoiceConfig {
        private double relativeTolerance = 0.1;
        private long minimumToleranceMs = 20;
        private OutlierRejectionConfig outlierRejection = new OutlierRejectionConfig();
    }

    @Getter
    @Setter
    public static final class LeastRecentlyUsedInPoolConfig implements CandidateChoiceConfig {
        private CandidatePoolConfig pool = new RatioCandidatePoolConfig();
    }

    @JsonTypeInfo(use = JsonTypeInfo.Id.NAME, property = "type")
    @JsonSubTypes({
            @JsonSubTypes.Type(value = RatioCandidatePoolConfig.class, name = "RATIO"),
            @JsonSubTypes.Type(value = FixedCandidatePoolConfig.class, name = "FIXED")
    })
    public sealed interface CandidatePoolConfig
            permits RatioCandidatePoolConfig, FixedCandidatePoolConfig {
    }

    @Getter
    @Setter
    public static final class RatioCandidatePoolConfig implements CandidatePoolConfig {
        private double ratio = 0.3;
        private int minimumWorkers = 1;
    }

    @Getter
    @Setter
    public static final class FixedCandidatePoolConfig implements CandidatePoolConfig {
        private int workers = 1;
    }

    @Getter
    @Setter
    public static final class CacheAffinityConfig {
        /**
         * Maximum projected-TTFT penalty accepted for choosing a cache leader.
         * The cutoff is {@code minimumProjectedTtft + maxExtraTtftMs}; cache
         * affinity never bypasses an unavailable or unmodelled endpoint.
         */
        private long maxExtraTtftMs;

        /**
         * Minimum reusable prefix percentage required before affinity applies.
         * This uses predictor-effective cache-hit tokens (the final cache block
         * is intentionally left as compute work), not the raw routing match.
         */
        private double minPrefixHitPercent = 5;
    }

    @Getter
    @Setter
    public static final class DecodeConfig {
        private DecodeAvailabilityConfig availability = new DecodeAvailabilityConfig();
        private KvReservationConfig kvReservation = new KvReservationConfig();
        private DecodeSelectorConfig selector = new KvUsageWeightedRandomConfig();

    }

    @Getter
    @Setter
    public static final class DecodeAvailabilityConfig {
        private long maxKvUsagePercent = 90;
        /**
         * Master-side cap for all Engine-facing Decode ownership, including
         * KV_ALLOCATED, RUNNING, dispatched shadows, and active dispatch
         * permits. This is not the Engine's physical RUNNING concurrency.
         */
        private Long maxEngineRequests;
    }

    @Getter
    @Setter
    public static final class KvReservationConfig {
        private Long maxOutputTokensForEstimate = 1000L;
    }

    @JsonTypeInfo(use = JsonTypeInfo.Id.NAME, property = "type")
    @JsonSubTypes({
            @JsonSubTypes.Type(value = RandomDecodeSelectorConfig.class, name = "RANDOM"),
            @JsonSubTypes.Type(value = KvUsageWeightedRandomConfig.class,
                    name = "KV_USAGE_WEIGHTED_RANDOM")
    })
    public sealed interface DecodeSelectorConfig
            extends EndpointSelectorConfig
            permits RandomDecodeSelectorConfig, KvUsageWeightedRandomConfig {
    }

    public static final class RandomDecodeSelectorConfig implements DecodeSelectorConfig {
    }

    @Getter
    @Setter
    public static final class KvUsageWeightedRandomConfig implements DecodeSelectorConfig {
        private double decayPerToken = 0.001;
        private DecodeOutlierRejectionConfig outlierRejection = new DecodeOutlierRejectionConfig();
    }

    @Getter
    @Setter
    public static final class DecodeOutlierRejectionConfig {
        private double maxEngineLoadVsAverageMultiplier = 3.0;
        private double maxKvUsedVsAverageMultiplier = 3.0;
    }

    @Getter
    @Setter
    public static final class VitConfig {
        private VitSelectorConfig selector = new RandomVitSelectorConfig();
    }

    @JsonTypeInfo(use = JsonTypeInfo.Id.NAME, property = "type")
    @JsonSubTypes({@JsonSubTypes.Type(value = RandomVitSelectorConfig.class, name = "RANDOM")})
    public sealed interface VitSelectorConfig
            extends EndpointSelectorConfig
            permits RandomVitSelectorConfig {
    }

    public static final class RandomVitSelectorConfig implements VitSelectorConfig {
    }
}
