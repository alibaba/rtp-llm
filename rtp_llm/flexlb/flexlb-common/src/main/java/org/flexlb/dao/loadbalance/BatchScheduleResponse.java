package org.flexlb.dao.loadbalance;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.databind.PropertyNamingStrategies;
import com.fasterxml.jackson.databind.annotation.JsonNaming;
import lombok.Getter;
import lombok.Setter;
import lombok.ToString;

import java.util.List;

@Getter
@Setter
@ToString
@JsonIgnoreProperties(ignoreUnknown = true)
@JsonNaming(PropertyNamingStrategies.SnakeCaseStrategy.class)
@JsonInclude(JsonInclude.Include.NON_NULL)
public class BatchScheduleResponse {

    private boolean success;

    private int code = 200;

    private String errorMessage;

    private List<BatchScheduleTarget> serverStatus;

    private String realMasterHost;

    public static BatchScheduleResponse success(List<BatchScheduleTarget> targets) {
        BatchScheduleResponse r = new BatchScheduleResponse();
        r.setSuccess(true);
        r.setServerStatus(targets);
        return r;
    }

    public static BatchScheduleResponse error(StrategyErrorType errorType, String message) {
        BatchScheduleResponse r = new BatchScheduleResponse();
        r.setCode(errorType.getErrorCode());
        r.setErrorMessage(message != null ? message : errorType.getErrorMsg());
        return r;
    }

    public static BatchScheduleResponse error(StrategyErrorType errorType) {
        return error(errorType, null);
    }
}
