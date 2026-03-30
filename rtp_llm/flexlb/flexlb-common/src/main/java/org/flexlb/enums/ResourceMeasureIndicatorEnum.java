package org.flexlb.enums;

import lombok.Getter;

/**
 * Resource measure indicator enumeration
 */
@Getter
public enum ResourceMeasureIndicatorEnum {

    WAIT_TIME("WaitTime", "Wait time"),

    REMAINING_KV_CACHE("RemainingKvCache", "Remaining KV cache"),

    ;

    private final String name;
    private final String description;

    ResourceMeasureIndicatorEnum(String name, String description) {
        this.name = name;
        this.description = description;
    }
}
