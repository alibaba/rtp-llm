package org.flexlb.enums;

import com.fasterxml.jackson.annotation.JsonProperty;

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
}
