package org.flexlb.discovery;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonValue;

import java.util.Arrays;

public enum ServiceDiscoveryType {
    VIPSERVER("vipserver"),
    DASHSCOPE("dashscope"),
    STATIC_ENV("static-env");

    private final String value;

    ServiceDiscoveryType(String value) {
        this.value = value;
    }

    @JsonCreator
    public static ServiceDiscoveryType fromValue(String value) {
        return Arrays.stream(values())
                .filter(type -> type.value.equalsIgnoreCase(value))
                .findFirst()
                .orElseThrow(() -> new IllegalArgumentException(
                        "Unsupported service discovery type: " + value));
    }

    @JsonValue
    public String getValue() {
        return value;
    }
}
