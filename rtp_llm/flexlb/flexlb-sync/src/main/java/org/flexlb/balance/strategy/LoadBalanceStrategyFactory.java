package org.flexlb.balance.strategy;

import org.flexlb.enums.LoadBalanceStrategyEnum;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class LoadBalanceStrategyFactory {

    private static final Map<LoadBalanceStrategyEnum, LoadBalancer> loadBalancerFactory = new ConcurrentHashMap<>();

    public static void register(LoadBalanceStrategyEnum strategy, LoadBalancer loadBalancer) {
        loadBalancerFactory.put(strategy, loadBalancer);
    }

    public static LoadBalancer getLoadBalancer(LoadBalanceStrategyEnum strategy) {
        LoadBalancer loadBalancer = loadBalancerFactory.get(strategy);
        if (loadBalancer == null) {
            throw new RuntimeException("loadBalanceStrategy not found: " + strategy);
        }
        return loadBalancer;
    }

    /**
     * Test-only: clears the global registry so a test starts from a known-empty state and
     * neither inherits nor leaks strategy registrations across classes (the map is process-wide
     * static, so registration order would otherwise make tests order-dependent).
     */
    public static void resetForTesting() {
        loadBalancerFactory.clear();
    }
}
