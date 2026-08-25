package org.flexlb.config;

import org.flexlb.enums.LoadBalanceStrategyEnum;
import org.junit.jupiter.api.Test;

import static org.assertj.core.api.Assertions.assertThat;

@SuppressWarnings("deprecation")
class FlexlbConfigTest {

    @Test
    void unifiedOutstandingThresholdAppliesToTtftStrategies() {
        FlexlbConfig config = new FlexlbConfig();
        config.setCacheAffinityFirstOutstandingUncachedTokensThreshold(200);
        config.setOutstandingUncachedTokensThreshold(300L);

        assertThat(config.getEffectiveOutstandingUncachedTokensThreshold(
                LoadBalanceStrategyEnum.SHORTEST_TTFT)).isEqualTo(300);
        assertThat(config.getEffectiveOutstandingUncachedTokensThreshold(
                LoadBalanceStrategyEnum.CACHE_AFFINITY_FIRST)).isEqualTo(300);
    }

    @Test
    void zeroUnifiedOutstandingThresholdOverridesLegacyThresholds() {
        FlexlbConfig config = new FlexlbConfig();
        config.setCacheAffinityFirstOutstandingUncachedTokensThreshold(200);
        config.setOutstandingUncachedTokensThreshold(0L);

        assertThat(config.getEffectiveOutstandingUncachedTokensThreshold(
                LoadBalanceStrategyEnum.SHORTEST_TTFT)).isZero();
        assertThat(config.getEffectiveOutstandingUncachedTokensThreshold(
                LoadBalanceStrategyEnum.CACHE_AFFINITY_FIRST)).isZero();
    }

    @Test
    void legacyCacheAffinityOutstandingThresholdDoesNotApplyToShortestTtft() {
        FlexlbConfig config = new FlexlbConfig();
        config.setCacheAffinityFirstOutstandingUncachedTokensThreshold(200);

        assertThat(config.getEffectiveOutstandingUncachedTokensThreshold(
                LoadBalanceStrategyEnum.SHORTEST_TTFT)).isZero();
        assertThat(config.getEffectiveOutstandingUncachedTokensThreshold(
                LoadBalanceStrategyEnum.CACHE_AFFINITY_FIRST)).isEqualTo(200);
    }

    @Test
    void defaultsPrefillQueueSizeThresholdTo1024() {
        assertThat(new FlexlbConfig().getPrefillQueueSizeThreshold()).isEqualTo(1024);
    }

    @Test
    void negativeOutstandingThresholdsAreClampedToDisabled() {
        FlexlbConfig config = new FlexlbConfig();
        config.setCacheAffinityFirstOutstandingUncachedTokensThreshold(-200);
        config.setOutstandingUncachedTokensThreshold(-300L);

        assertThat(config.getEffectiveOutstandingUncachedTokensThreshold(
                LoadBalanceStrategyEnum.SHORTEST_TTFT)).isZero();
        assertThat(config.getEffectiveOutstandingUncachedTokensThreshold(
                LoadBalanceStrategyEnum.CACHE_AFFINITY_FIRST)).isZero();
    }

    @Test
    void cacheAffinityExtraWorkTokensDefaultsToZeroAndCanBeConfigured() {
        FlexlbConfig config = new FlexlbConfig();

        assertThat(config.getCacheAffinityFirstMaxExtraWorkTokens()).isZero();
        config.setCacheAffinityFirstMaxExtraWorkTokens(300);
        assertThat(config.getCacheAffinityFirstMaxExtraWorkTokens()).isEqualTo(300);
    }
}
