package org.flexlb.enums;

import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonValue;

/**
 * Logging level for request
 */
public enum LogLevel {

    @JsonProperty("trace")
    TRACE,
    @JsonProperty("debug")
    DEBUG,
    @JsonProperty("info")
    INFO,
    @JsonProperty("warn")
    WARN,
    @JsonProperty("error")
    ERROR;

    @JsonValue
    public String jsonValue() {
        return name().toLowerCase(java.util.Locale.ROOT);
    }
}
