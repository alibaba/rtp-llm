package org.flexlb.dao.loadbalance;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Getter;
import lombok.Setter;
import org.flexlb.enums.LogLevel;

@Getter
@Setter
public class LogLevelUpdateRequest {

    @JsonProperty("log_level")
    private LogLevel logLevel;

}
