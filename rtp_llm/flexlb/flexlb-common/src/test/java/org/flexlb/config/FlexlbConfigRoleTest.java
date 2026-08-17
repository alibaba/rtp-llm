package org.flexlb.config;

import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.LoadBalanceStrategyEnum;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class FlexlbConfigRoleTest {

    @Test
    void should_define_strategy_for_supported_worker_roles() {
        FlexlbConfig config = new FlexlbConfig();

        assertEquals(LoadBalanceStrategyEnum.COST_BASED_PREFILL,
                config.getStrategyForRoleType(RoleType.PREFILL));
        assertEquals(LoadBalanceStrategyEnum.COST_BASED_DECODE,
                config.getStrategyForRoleType(RoleType.DECODE));
        assertEquals(LoadBalanceStrategyEnum.COST_BASED_PREFILL,
                config.getStrategyForRoleType(RoleType.PDFUSION));
        assertEquals(LoadBalanceStrategyEnum.RANDOM,
                config.getStrategyForRoleType(RoleType.VIT));
    }

    @Test
    void should_not_clamp_max_new_tokens_when_cap_is_zero() {
        FlexlbConfig config = new FlexlbConfig();
        config.setMaxNewTokensCap(0L);
        // cap = 0 means no clamping
        assertEquals(393216L, config.effectiveMaxNewTokensForReservation(393216L));
        assertEquals(0L, config.effectiveMaxNewTokensForReservation(0L));
    }

    @Test
    void should_apply_default_cap_of_1000() {
        FlexlbConfig config = new FlexlbConfig();
        // default cap = 1000 based on actual generation distribution
        assertEquals(1000L, config.effectiveMaxNewTokensForReservation(393216L));
        assertEquals(500L, config.effectiveMaxNewTokensForReservation(500L));
    }

    @Test
    void should_clamp_max_new_tokens_when_cap_is_positive() {
        FlexlbConfig config = new FlexlbConfig();
        config.setMaxNewTokensCap(8192L);
        // declared exceeds cap → clamped to cap
        assertEquals(8192L, config.effectiveMaxNewTokensForReservation(393216L));
        // declared below cap → unchanged
        assertEquals(100L, config.effectiveMaxNewTokensForReservation(100L));
        // declared equals cap → unchanged
        assertEquals(8192L, config.effectiveMaxNewTokensForReservation(8192L));
    }
}
