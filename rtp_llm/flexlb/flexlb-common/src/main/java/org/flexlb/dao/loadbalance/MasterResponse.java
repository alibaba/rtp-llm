package org.flexlb.dao.loadbalance;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

import java.util.List;

@JsonIgnoreProperties(ignoreUnknown = true)
@Data
public class MasterResponse {

    @JsonProperty("server_status")
    private List<ServerStatus> serverStatus;

    @JsonProperty("inter_request_id")
    private long interRequestId;

    @JsonProperty("success")
    private boolean success;

    @JsonProperty("code")
    private int code = 200;

    @JsonProperty("error_message")
    private String errorMessage;

    @JsonProperty("real_master_host")
    private String realMasterHost;

    public static MasterResponse code(int code) {
        MasterResponse result = new MasterResponse();
        result.setSuccess(false);
        result.setCode(code);
        return result;
    }

    public static MasterResponse error(StrategyErrorType strategyErrorType) {
        MasterResponse result = new MasterResponse();
        result.setSuccess(false);
        result.setCode(strategyErrorType.getErrorCode());
        result.setErrorMessage(strategyErrorType.getErrorMsg());
        return result;
    }
}
