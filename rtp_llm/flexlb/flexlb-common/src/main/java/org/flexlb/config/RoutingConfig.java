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
        /**
         * Production prefill execution-time fit (DSv4), promoted to the code
         * default so an omitted {@code executionTimeEstimator} no longer falls
         * back to the legacy {@code 1 ms/token} sum — which overpredicted a
         * 32k all-miss prefill by ~96x (32.8 s vs the fitted ~342 ms) and
         * poisoned every ledger-driven routing decision built on it.
         *
         * <p>Verbatim reuse is safe in BOTH evaluation contexts:
         * single-request evaluations ({@code FormulaPredictor.estimateMs})
         * bind {@code batchSize = 1}, so every batchSize stair/interaction
         * term vanishes and the fit degenerates to its batch-of-one
         * semantics; batch evaluations ({@code predictBatchMs}) bind the real
         * batch size, matching the production serving context the fit was
         * trained on. All-miss checkpoints: 512 -> ~219 ms, 32768 -> ~342 ms,
         * 49152 -> ~494 ms (predictor ms, before any sleep scaling).
         *
         * <p>Single source of truth: the mock engine resolves its prefill
         * duration from the master-config FORMULA expression, or from this
         * constant when the estimator is absent — never from a silent
         * hard-coded fallback.
         */
        public static final String DEFAULT_EXPRESSION =
                "max(196, -68.612174288157 + 0.993068319341 * (max(0, 287.3980926717 + 2.30134977837751 * batchSize + "
                + "0.158123254797307 * sum(hitCacheTokens / 1024.) + 0.575522710053703 * sum(computeTokens / 1024.) + "
                + "0.0517623430739831 * sum(computeTokens / 1024. * computeTokens / 1024.) + 0.0395308136993267 * "
                + "sum(hitCacheTokens / 1024. * computeTokens / 1024.) + 0.0104363634681015 * sum(hitCacheTokens / 1024. * "
                + "hitCacheTokens / 1024.) + 0.575522710053703 * max(sum(computeTokens / 1024.) - 16, 0) + 2.82077211814514 "
                + "* max(sum(computeTokens / 1024.) - 32, 0) - 0.0254671429192862 * max(sum(computeTokens / 1024.) - 64, 0) "
                + "+ 2.15779213792494 * max(sum(computeTokens / 1024.) - 96, 0) + 0.247806025472364 * "
                + "max(sum(hitCacheTokens / 1024.) - 32, 0) - 0.444522654549492 * max(sum(hitCacheTokens / 1024.) - 64, 0) "
                + "- 0.427317020061895 * max(sum(hitCacheTokens / 1024.) - 128, 0) + 0.347029077528455 * "
                + "max(sum(hitCacheTokens / 1024.) - 256, 0) - 0.298742307762735 * max(sum(hitCacheTokens / 1024.) - 384, "
                + "0) + 2.30134977837751 * max(batchSize - 8, 0) - 3.54884859699154 * max(batchSize - 16, 0) - "
                + "11.3438560779984 * max(batchSize - 24, 0) + 0.879751992138183 * sum(max(computeTokens / 1024. - 2, 0)) + "
                + "0.636364578079591 * sum(max(computeTokens / 1024. - 4, 0)) - 0.0513345988517118 * sum(max(computeTokens "
                + "/ 1024. - 8, 0)) - 0.332584389129357 * sum(max(hitCacheTokens / 1024. - 2, 0)) + 0.305819761192588 * "
                + "sum(max(hitCacheTokens / 1024. - 4, 0)) - 0.287610979974721 * sum(max(hitCacheTokens / 1024. - 8, 0)) + "
                + "0.191310200712013 * sum(max(hitCacheTokens / 1024. - 12, 0)) + 0.0130251644478961 * max(batchSize - 8, "
                + "0) * sum(hitCacheTokens / 1024.) + 0.00981382840761646 * max(batchSize - 16, 0) * sum(hitCacheTokens / "
                + "1024.) - 0.0299132587297009 * max(batchSize - 24, 0) * sum(hitCacheTokens / 1024.) + 0.0447455122487382 "
                + "* max(batchSize - 8, 0) * sum(computeTokens / 1024.) + 0.0104635312001851 * max(batchSize - 16, 0) * "
                + "sum(computeTokens / 1024.) + 0.0542737877321807 * max(batchSize - 24, 0) * sum(computeTokens / 1024.))))";

        private String expression = DEFAULT_EXPRESSION;
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
