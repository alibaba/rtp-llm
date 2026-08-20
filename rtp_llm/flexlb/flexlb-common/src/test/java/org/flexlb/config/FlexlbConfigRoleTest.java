package org.flexlb.config;

import org.flexlb.config.RoutingConfig.LeastRecentlyUsedInPoolConfig;
import org.flexlb.config.RoutingConfig.RandomDecodeSelectorConfig;
import org.flexlb.config.RoutingConfig.RandomPrefillSelectorConfig;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.LoadBalanceStrategyEnum;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class FlexlbConfigRoleTest {

    @Test
    void maps_role_selectors_to_current_router_implementations() {
        FlexlbConfig config = new FlexlbConfig();

        assertEquals(LoadBalanceStrategyEnum.COST_BASED_PREFILL,
                config.strategyFor(RoleType.PREFILL));
        assertEquals(LoadBalanceStrategyEnum.COST_BASED_PREFILL,
                config.strategyFor(RoleType.PDFUSION));
        assertEquals(LoadBalanceStrategyEnum.COST_BASED_DECODE,
                config.strategyFor(RoleType.DECODE));
        assertEquals(LoadBalanceStrategyEnum.RANDOM,
                config.strategyFor(RoleType.VIT));

        config.getRouter().getRoles().getPrefill()
                .setSelector(new RandomPrefillSelectorConfig());
        config.getRouter().getRoles().getDecode()
                .setSelector(new RandomDecodeSelectorConfig());
        assertEquals(LoadBalanceStrategyEnum.RANDOM,
                config.strategyFor(RoleType.PREFILL));
        assertEquals(LoadBalanceStrategyEnum.RANDOM,
                config.strategyFor(RoleType.DECODE));
    }

    @Test
    void pdfusion_reuses_prefill_configuration() {
        RoutingConfig.EstimatedTtftSelectorConfig selector =
                new RoutingConfig.EstimatedTtftSelectorConfig();
        selector.setCandidateChoice(new LeastRecentlyUsedInPoolConfig());
        FlexlbConfig config = new FlexlbConfig();
        config.getRouter().getRoles().getPrefill().setSelector(selector);

        assertEquals(LoadBalanceStrategyEnum.SHORTEST_TTFT,
                config.strategyFor(RoleType.PREFILL));
        assertEquals(LoadBalanceStrategyEnum.SHORTEST_TTFT,
                config.strategyFor(RoleType.PDFUSION));
    }

    @Test
    void output_token_reservation_uses_absence_for_no_cap() {
        FlexlbConfig config = new FlexlbConfig();
        assertEquals(1000L, config.effectiveMaxOutputTokensForReservation(393216L));
        assertEquals(500L, config.effectiveMaxOutputTokensForReservation(500L));

        config.getRouter().getRoles().getDecode().getKvReservation()
                .setMaxOutputTokensForEstimate(8192L);
        assertEquals(8192L, config.effectiveMaxOutputTokensForReservation(393216L));

        config.getRouter().getRoles().getDecode().getKvReservation()
                .setMaxOutputTokensForEstimate(null);
        assertEquals(393216L,
                config.effectiveMaxOutputTokensForReservation(393216L));
    }

    @Test
    void decode_reservation_is_non_negative_saturating_and_capacity_capped() {
        FlexlbConfig config = new FlexlbConfig();

        assertEquals(1500L, config.decodeKvReservationTokens(500L, 10_000L, 0L));
        assertEquals(1200L, config.decodeKvReservationTokens(500L, 10_000L, 1200L));
        assertEquals(0L, config.decodeKvReservationTokens(-1L, -1L, 0L));

        config.getRouter().getRoles().getDecode().getKvReservation()
                .setMaxOutputTokensForEstimate(null);
        assertEquals(Long.MAX_VALUE, config.decodeKvReservationTokens(
                Long.MAX_VALUE - 10L, 100L, 0L));
        assertEquals(4096L, config.decodeKvReservationTokens(
                Long.MAX_VALUE - 10L, 100L, 4096L));
    }
}
