package org.flexlb.dao.loadbalance;

import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Getter;
import lombok.Setter;
import lombok.ToString;

import java.util.List;

@Getter
@Setter
@ToString
@JsonIgnoreProperties(ignoreUnknown = true)
@JsonInclude(JsonInclude.Include.NON_NULL)
public class BatchScheduleResponse {

    @JsonProperty("success")
    private boolean success;

    @JsonProperty("code")
    private int code = 200;

    @JsonProperty("error_message")
    private String errorMessage;

    @JsonProperty("server_status")
    private List<BatchScheduleTarget> serverStatus;

    @JsonProperty("real_master_host")
    private String realMasterHost;

    public static BatchScheduleResponse success(List<BatchScheduleTarget> targets) {
        BatchScheduleResponse r = new BatchScheduleResponse();
        r.setSuccess(true);
        r.setCode(200);
        r.setServerStatus(targets);
        return r;
    }

    public static BatchScheduleResponse error(StrategyErrorType errorType, String message) {
        BatchScheduleResponse r = new BatchScheduleResponse();
        r.setSuccess(false);
        r.setCode(errorType.getErrorCode());
        r.setErrorMessage(message != null ? message : errorType.getErrorMsg());
        return r;
    }

    public static BatchScheduleResponse error(StrategyErrorType errorType) {
        return error(errorType, errorType.getErrorMsg());
    }
}
