package org.flexlb.enums;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonValue;
import lombok.Getter;

@Getter
public enum TaskPhase {

    PENDING("pending"),
    RUNNING("running"),
    KV_ALLOCATED("kv_allocated");

    @JsonValue
    private final String value;

    TaskPhase(String value) {
        this.value = value;
    }

    @JsonCreator
    public static TaskPhase fromValue(String value) {
        if (value == null) {
            return null;
        }
        for (TaskPhase phase : values()) {
            if (phase.value.equalsIgnoreCase(value)) {
                return phase;
            }
        }
        return null;
    }
}
