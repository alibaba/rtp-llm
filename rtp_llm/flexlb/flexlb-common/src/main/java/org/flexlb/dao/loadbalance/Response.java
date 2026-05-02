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

    /**
     * V1-α DP batching: true means the request has been asynchronously enqueued at
     * Prefill by Master via Enqueue. Frontend should switch to Decode.FetchResponse
     * for token streaming and skip the legacy GenerateStreamCall to Prefill.
     * <p>
     * Default false keeps old frontends transparent: they will continue to open the
     * stream against Prefill, which (per engine contract) should short-circuit-attach
     * to the already-enqueued task by request_id.
     */
    @JsonProperty("enqueued_by_master")
    private boolean enqueuedByMaster = false;

    public static Response error(StrategyErrorType strategyErrorType) {
        Response result = new Response();
        result.setSuccess(false);
        result.setCode(strategyErrorType.getErrorCode());
        result.setErrorMessage(strategyErrorType.getErrorMsg());
        return result;
    }
}
