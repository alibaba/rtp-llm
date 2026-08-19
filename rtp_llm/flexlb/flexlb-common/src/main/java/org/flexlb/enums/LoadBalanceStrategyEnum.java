package org.flexlb.enums;

import lombok.Getter;

@Getter
public enum LoadBalanceStrategyEnum {

    RANDOM("Random"),  // Random assignment

    SHORTEST_TTFT("ShortestTTFT"),  // Shortest Time-To-First-Token

    CACHE_AFFINITY_FIRST("CacheAffinityFirst"),  // Cache affinity with a bounded TTFT cost

    WEIGHTED_CACHE("WeightedCache")  // Lowest cache usage strategy

    ;
    private final String name;

    LoadBalanceStrategyEnum(String name) {
        this.name = name;
    }

}
