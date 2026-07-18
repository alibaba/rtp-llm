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
}
