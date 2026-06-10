package org.flexlb.config;

import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.LoadBalanceStrategyEnum;
import org.flexlb.enums.EngineType;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FlexlbConfigTest {

    @Test
    void llm_engine_accepts_any_strategy() {
        FlexlbConfig config = new FlexlbConfig();

        assertDoesNotThrow(() -> config.validateEngineTypeConfig(List.of(RoleType.values())));
    }

    @Test
    void embedding_engine_defaults_to_round_robin_when_strategy_unset() {
        FlexlbConfig config = new FlexlbConfig();
        config.setEngineType(EngineType.EMBEDDING);

        assertEquals(LoadBalanceStrategyEnum.ROUND_ROBIN, config.getStrategyForRoleType(RoleType.PDFUSION));
        assertDoesNotThrow(() -> config.validateEngineTypeConfig(List.of(RoleType.PDFUSION)));
    }

    @Test
    void llm_engine_defaults_to_shortest_ttft_when_strategy_unset() {
        FlexlbConfig config = new FlexlbConfig();

        assertEquals(LoadBalanceStrategyEnum.SHORTEST_TTFT, config.getStrategyForRoleType(RoleType.PDFUSION));
    }

    @Test
    void embedding_engine_rejects_explicit_load_aware_strategy() {
        FlexlbConfig config = new FlexlbConfig();
        config.setEngineType(EngineType.EMBEDDING);
        config.setLoadBalanceStrategy(LoadBalanceStrategyEnum.SHORTEST_TTFT);

        IllegalStateException e = assertThrows(IllegalStateException.class,
                () -> config.validateEngineTypeConfig(List.of(RoleType.PDFUSION)));
        assertTrue(e.getMessage().contains("SHORTEST_TTFT"));
    }

    @Test
    void embedding_engine_accepts_round_robin() {
        FlexlbConfig config = new FlexlbConfig();
        config.setEngineType(EngineType.EMBEDDING);
        config.setLoadBalanceStrategy(LoadBalanceStrategyEnum.ROUND_ROBIN);

        assertDoesNotThrow(() -> config.validateEngineTypeConfig(List.of(RoleType.PDFUSION)));
    }

    @Test
    void embedding_engine_ignores_undeployed_roles() {
        FlexlbConfig config = new FlexlbConfig();
        config.setEngineType(EngineType.EMBEDDING);
        config.setLoadBalanceStrategy(LoadBalanceStrategyEnum.ROUND_ROBIN);

        // DECODE default is WEIGHTED_CACHE (load-aware), but DECODE is not deployed
        assertDoesNotThrow(() -> config.validateEngineTypeConfig(List.of(RoleType.PDFUSION)));
        assertThrows(IllegalStateException.class,
                () -> config.validateEngineTypeConfig(List.of(RoleType.PDFUSION, RoleType.DECODE)));
    }
}
