package org.flexlb.balance.strategy;

import org.flexlb.enums.LoadBalanceStrategyEnum;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class LoadBalanceStrategyFactory {

    private static final Map<LoadBalanceStrategyEnum, LoadBalanceStrategy> loadBalanceStrategyMap = new ConcurrentHashMap<>();

    public static void register(LoadBalanceStrategyEnum strategy, LoadBalanceStrategy loadBalanceStrategy) {
        loadBalanceStrategyMap.put(strategy, loadBalanceStrategy);
    }

    public static LoadBalanceStrategy getLoadBalanceStrategy(LoadBalanceStrategyEnum strategy) {
        LoadBalanceStrategy loadBalanceStrategy = loadBalanceStrategyMap.get(strategy);
        if (loadBalanceStrategy == null) {
            throw new RuntimeException("loadBalanceStrategy not found: " + strategy);
        }
        return loadBalanceStrategy;
    }

    /**
     * Test-only: clears the global registry so a test starts from a known-empty state and
     * neither inherits nor leaks strategy registrations across classes (the map is process-wide
     * static, so registration order would otherwise make tests order-dependent).
     *
     * <p>Never call from production code. Strategies register exactly once during Spring
     * initialization; clearing the registry at runtime is unrecoverable — every subsequent
     * {@link #getLoadBalanceStrategy} throws until the process restarts. Public only because test
     * callers live in more than one package.
     */
    public static void resetForTesting() {
        loadBalanceStrategyMap.clear();
    }
}
