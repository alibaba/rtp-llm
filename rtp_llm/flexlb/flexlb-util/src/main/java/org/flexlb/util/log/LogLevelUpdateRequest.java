package org.flexlb.util.log;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Getter;
import lombok.Setter;

@Getter
@Setter
public class LogLevelUpdateRequest {

    @JsonProperty("log_level")
    private LogLevel logLevel;
}
