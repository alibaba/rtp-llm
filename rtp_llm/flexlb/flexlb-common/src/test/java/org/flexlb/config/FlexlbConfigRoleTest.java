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
}
