package org.flexlb.config;

import com.fasterxml.jackson.annotation.JsonSubTypes;
import com.fasterxml.jackson.annotation.JsonTypeInfo;
import lombok.Getter;
import lombok.Setter;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.LoadBalanceStrategyEnum;

import static org.flexlb.enums.LoadBalanceStrategyEnum.COST_BASED_DECODE;
import static org.flexlb.enums.LoadBalanceStrategyEnum.COST_BASED_PREFILL;
import static org.flexlb.enums.LoadBalanceStrategyEnum.RANDOM;
import static org.flexlb.enums.LoadBalanceStrategyEnum.SHORTEST_TTFT;

@Getter
@Setter
public final class RoutingConfig {

    private long availabilityHysteresisPercent = 15;
    private volatile TrafficPolicyConfig groupSelector;
    private RolesConfig roles = new RolesConfig();

    public LoadBalanceStrategyEnum strategyFor(RoleType role) {
        return switch (role) {
            case PREFILL, PDFUSION -> roles.getPrefill().strategy();
            case DECODE -> roles.getDecode().strategy();
            case VIT -> RANDOM;
            case FRONTEND -> null;
        };
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
        private SessionAffinityConfig sessionAffinity;

        private LoadBalanceStrategyEnum strategy() {
            if (selector instanceof RandomPrefillSelectorConfig) {
                return RANDOM;
            }
            EstimatedTtftSelectorConfig estimated = (EstimatedTtftSelectorConfig) selector;
            return estimated.getCandidateChoice() instanceof LeastRecentlyUsedInPoolConfig
                    ? SHORTEST_TTFT : COST_BASED_PREFILL;
        }
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
        private double maxWaitVsAverageMultiplier = 3.0;
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
        private long maxExtraTtftMs;
        private double minPrefixHitPercent = 5;
    }

    @Getter
    @Setter
    public static final class SessionAffinityConfig {
        private long ttlMs;
        private long maxExtraTtftMs;
    }

    @Getter
    @Setter
    public static final class DecodeConfig {
        private DecodeAvailabilityConfig availability = new DecodeAvailabilityConfig();
        private KvReservationConfig kvReservation = new KvReservationConfig();
        private DecodeSelectorConfig selector = new KvUsageWeightedRandomConfig();

        private LoadBalanceStrategyEnum strategy() {
            return selector instanceof RandomDecodeSelectorConfig ? RANDOM : COST_BASED_DECODE;
        }
    }

    @Getter
    @Setter
    public static final class DecodeAvailabilityConfig {
        private long maxKvUsagePercent = 90;
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
    public sealed interface VitSelectorConfig permits RandomVitSelectorConfig {
    }

    public static final class RandomVitSelectorConfig implements VitSelectorConfig {
    }
}
