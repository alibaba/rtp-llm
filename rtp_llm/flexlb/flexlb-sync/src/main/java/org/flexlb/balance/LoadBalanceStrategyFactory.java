package org.flexlb.balance;

import org.flexlb.balance.strategy.LoadBalancer;
import org.flexlb.enums.LoadBalanceStrategyEnum;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

public class LoadBalanceStrategyFactory {

    private final static Map<LoadBalanceStrategyEnum, LoadBalancer> loadBalancerFactory = new ConcurrentHashMap<>();

    public static void register(LoadBalanceStrategyEnum strategy, LoadBalancer loadBalancer) {
        loadBalancerFactory.put(strategy, loadBalancer);
    }

    public static LoadBalancer getLoadBalanceStrategy(LoadBalanceStrategyEnum strategy) {
        LoadBalancer loadBalancer = loadBalancerFactory.get(strategy);
        if (loadBalancer == null) {
            throw new RuntimeException("loadBalanceStrategy not found: " + strategy);
        }
        return loadBalancer;
    }
}
