package org.flexlb.enums;

import lombok.Getter;

@Getter
public enum LoadBalanceStrategyEnum {

    RANDOM("Random"),  // Random assignment

    SHORTEST_TTFT("ShortestTTFT"),  // Shortest Time-To-First-Token

    WEIGHTED_CACHE("WeightedCache"),  // Lowest cache usage strategy

    ROUND_ROBIN("RoundRobin")  // Cursor-based round-robin (cheap, no load awareness)

    ;
    private final String name;

    LoadBalanceStrategyEnum(String name) {
        this.name = name;
    }

}
