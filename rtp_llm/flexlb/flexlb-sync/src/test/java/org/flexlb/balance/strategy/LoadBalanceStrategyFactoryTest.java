package org.flexlb.balance.strategy;

import org.flexlb.enums.LoadBalanceStrategyEnum;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.Mockito.mock;

class LoadBalanceStrategyFactoryTest {

    @AfterEach
    void tearDown() {
        LoadBalanceStrategyFactory.clear();
    }

    @Test
    void clearsRegisteredStrategies() {
        LoadBalanceStrategyFactory.register(
                LoadBalanceStrategyEnum.RANDOM, mock(LoadBalanceStrategy.class));

        LoadBalanceStrategyFactory.clear();

        assertThrows(
                RuntimeException.class,
                () -> LoadBalanceStrategyFactory.getLoadBalanceStrategy(LoadBalanceStrategyEnum.RANDOM));
    }
}
