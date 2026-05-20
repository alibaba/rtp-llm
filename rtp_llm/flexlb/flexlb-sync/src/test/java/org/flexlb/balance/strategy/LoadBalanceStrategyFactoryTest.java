package org.flexlb.balance.strategy;

import org.flexlb.enums.LoadBalanceStrategyEnum;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.lang.reflect.Field;
import java.util.HashMap;
import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

class LoadBalanceStrategyFactoryTest {

    private Map<LoadBalanceStrategyEnum, LoadBalancer> savedFactory;

    @BeforeEach
    @SuppressWarnings("unchecked")
    void snapshotAndClearFactory() throws Exception {
        // The factory is a process-wide static map populated at Spring startup.
        // Save and restore around each test so we can assert on a clean slate
        // (missing-strategy probe) and never leak mocks into other tests.
        Field f = LoadBalanceStrategyFactory.class.getDeclaredField("loadBalancerFactory");
        f.setAccessible(true);
        Map<LoadBalanceStrategyEnum, LoadBalancer> live =
                (Map<LoadBalanceStrategyEnum, LoadBalancer>) f.get(null);
        savedFactory = new HashMap<>(live);
        live.clear();
    }

    @AfterEach
    @SuppressWarnings("unchecked")
    void restoreFactory() throws Exception {
        Field f = LoadBalanceStrategyFactory.class.getDeclaredField("loadBalancerFactory");
        f.setAccessible(true);
        Map<LoadBalanceStrategyEnum, LoadBalancer> live =
                (Map<LoadBalanceStrategyEnum, LoadBalancer>) f.get(null);
        live.clear();
        live.putAll(savedFactory);
    }

    @Test
    void register_then_get_returns_registered_balancer() {
        LoadBalancer balancer = Mockito.mock(LoadBalancer.class);
        LoadBalanceStrategyFactory.register(LoadBalanceStrategyEnum.RANDOM, balancer);

        assertSame(balancer,
                LoadBalanceStrategyFactory.getLoadBalancer(LoadBalanceStrategyEnum.RANDOM));
    }

    @Test
    void get_unknown_strategy_throws_runtime_exception() {
        // RouteService routes by FlexlbConfig.getStrategyForRoleType(...). A
        // strategy name with no bean must surface a hard failure at the
        // gateway rather than silently fall back to whatever was registered
        // last — that would make routing decisions non-deterministic.
        RuntimeException ex = assertThrows(RuntimeException.class,
                () -> LoadBalanceStrategyFactory.getLoadBalancer(LoadBalanceStrategyEnum.RANDOM));
        assertEquals("loadBalanceStrategy not found: " + LoadBalanceStrategyEnum.RANDOM,
                ex.getMessage());
    }

    @Test
    void re_register_overwrites_previous_balancer() {
        LoadBalancer first = Mockito.mock(LoadBalancer.class);
        LoadBalancer second = Mockito.mock(LoadBalancer.class);
        LoadBalanceStrategyFactory.register(LoadBalanceStrategyEnum.WEIGHTED_CACHE, first);
        LoadBalanceStrategyFactory.register(LoadBalanceStrategyEnum.WEIGHTED_CACHE, second);

        LoadBalancer current =
                LoadBalanceStrategyFactory.getLoadBalancer(LoadBalanceStrategyEnum.WEIGHTED_CACHE);
        assertSame(second, current);
        assertNotSame(first, current);
    }

    @Test
    void register_supports_multiple_strategies_independently() {
        LoadBalancer randomBean = Mockito.mock(LoadBalancer.class);
        LoadBalancer ttftBean = Mockito.mock(LoadBalancer.class);
        LoadBalancer weightedBean = Mockito.mock(LoadBalancer.class);
        LoadBalanceStrategyFactory.register(LoadBalanceStrategyEnum.RANDOM, randomBean);
        LoadBalanceStrategyFactory.register(LoadBalanceStrategyEnum.SHORTEST_TTFT, ttftBean);
        LoadBalanceStrategyFactory.register(LoadBalanceStrategyEnum.WEIGHTED_CACHE, weightedBean);

        assertSame(randomBean,
                LoadBalanceStrategyFactory.getLoadBalancer(LoadBalanceStrategyEnum.RANDOM));
        assertSame(ttftBean,
                LoadBalanceStrategyFactory.getLoadBalancer(LoadBalanceStrategyEnum.SHORTEST_TTFT));
        assertSame(weightedBean,
                LoadBalanceStrategyFactory.getLoadBalancer(LoadBalanceStrategyEnum.WEIGHTED_CACHE));
    }
}
