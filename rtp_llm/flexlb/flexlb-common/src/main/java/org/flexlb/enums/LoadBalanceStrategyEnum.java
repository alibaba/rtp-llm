package org.flexlb.enums;

import lombok.Getter;

@Getter
public enum LoadBalanceStrategyEnum {

    RANDOM("Random"),  // Random assignment

    SHORTEST_TTFT("ShortestTTFT"),  // Shortest Time-To-First-Token

    WEIGHTED_CACHE("WeightedCache"),  // Lowest cache usage strategy

    COST_BASED_PREFILL("CostBasedPrefill")  // Cost-based prefill worker selection

    ;
    private final String name;

    LoadBalanceStrategyEnum(String name) {
        this.name = name;
    }

}
