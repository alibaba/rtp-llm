package org.flexlb.enums;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonValue;
import lombok.Getter;

@Getter
public enum LoadBalanceStrategyEnum {

    RANDOM("Random"),

    COST_BASED_PREFILL("CostBasedPrefill"),

    COST_BASED_DECODE("CostBasedDecode"),

    SHORTEST_TTFT("ShortestTtft")

    ;
    private final String name;

    LoadBalanceStrategyEnum(String name) {
        this.name = name;
    }

    @JsonValue
    public String getName() {
        return name;
    }

    @JsonCreator
    public static LoadBalanceStrategyEnum fromName(String value) {
        for (LoadBalanceStrategyEnum e : values()) {
            if (e.name.equals(value) || e.name().equals(value)) {
                return e;
            }
        }
        throw new IllegalArgumentException("Unknown strategy: " + value);
    }
}
