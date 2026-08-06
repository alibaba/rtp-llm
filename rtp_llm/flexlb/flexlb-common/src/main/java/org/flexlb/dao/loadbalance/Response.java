package org.flexlb.dao.loadbalance;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

import java.util.List;
import java.util.Map;

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

    @JsonProperty("enqueued_by_master")
    private boolean enqueuedByMaster = false;

    @JsonProperty("worker_summary")
    private Map<String, WorkerRoleSummary> workerSummary;

    @JsonProperty("ready")
    private boolean ready = true;

    public static Response error(StrategyErrorType strategyErrorType) {
        Response result = new Response();
        result.setSuccess(false);
        result.setCode(strategyErrorType.getErrorCode());
        result.setErrorMessage(strategyErrorType.getErrorMsg());
        return result;
    }

    /**
     * Create an error response with a custom error message.
     *
     * @param strategyErrorType the error type (provides the error code)
     * @param message          the error message; if {@code null}, falls back to
     *                         {@code strategyErrorType.getErrorMsg()}
     */
    public static Response error(StrategyErrorType strategyErrorType, String message) {
        Response result = new Response();
        result.setSuccess(false);
        result.setCode(strategyErrorType.getErrorCode());
        result.setErrorMessage(message != null ? message : strategyErrorType.getErrorMsg());
        return result;
    }

    @Data
    public static class WorkerRoleSummary {
        private int discovered;
        private int alive;
        private long maxQueueTokens;
    }
}
