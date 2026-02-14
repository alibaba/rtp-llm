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
}
