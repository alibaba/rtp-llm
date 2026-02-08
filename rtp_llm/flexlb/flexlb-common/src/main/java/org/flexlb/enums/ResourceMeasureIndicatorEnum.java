package org.flexlb.enums;

import lombok.Getter;

/**
 * 资源度量指标枚举
 */
@Getter
public enum ResourceMeasureIndicatorEnum {

    WAIT_TIME("WaitTime", "等待时间"),

    REMAINING_KV_CACHE("RemainingKvCache", "剩余 KvCache"),

    ;

    private final String name;
    private final String description;

    ResourceMeasureIndicatorEnum(String name, String description) {
        this.name = name;
        this.description = description;
    }
}
