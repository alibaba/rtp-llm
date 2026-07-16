package org.flexlb.dao.loadbalance;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

import java.util.List;

@JsonIgnoreProperties(ignoreUnknown = true)
@Data
public class Response {

    @JsonProperty("server_status")
    private List<ServerStatus> serverStatus;

    @JsonProperty("success")
    private boolean success;

    @JsonProperty("code")
    private int code = 200;

    @JsonProperty("error_message")
    private String errorMessage;

    @JsonProperty("real_master_host")
    private String realMasterHost;

    @JsonProperty("queue_length")
    private Integer queueLength;

    public static Response error(StrategyErrorType strategyErrorType) {
        return error(strategyErrorType, strategyErrorType.getErrorMsg());
    }

    /** Variant carrying a caller-supplied message, mirroring {@code BatchScheduleResponse.error}. */
    public static Response error(StrategyErrorType strategyErrorType, String errorMessage) {
        Response result = new Response();
        result.setSuccess(false);
        result.setCode(strategyErrorType.getErrorCode());
        result.setErrorMessage(errorMessage);
        return result;
    }
}
