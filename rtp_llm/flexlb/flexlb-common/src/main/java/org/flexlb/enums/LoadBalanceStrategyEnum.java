package org.flexlb.enums;

import lombok.Getter;

@Getter
public enum LoadBalanceStrategyEnum {

    RANDOM("Random"),  // 随机分配

    SHORTEST_TTFT("ShortestTTFT"),  // 最短TTFT

    WEIGHTED_CACHE("WeightedCache")  // 最低缓存使用策略

    ;
    private final String name;

    LoadBalanceStrategyEnum(String name) {
        this.name = name;
    }

}
