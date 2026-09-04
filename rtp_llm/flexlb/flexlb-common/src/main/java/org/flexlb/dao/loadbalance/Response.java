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

    @JsonProperty("admission_reject_reason")
    private AdmissionRejectReason admissionRejectReason = AdmissionRejectReason.UNSPECIFIED;

    public static Response error(StrategyErrorType strategyErrorType) {
        return error(strategyErrorType, strategyErrorType.getErrorMsg(),
                AdmissionRejectReason.UNSPECIFIED);
    }

    /** Variant carrying a caller-supplied message, mirroring {@code BatchScheduleResponse.error}. */
    public static Response error(StrategyErrorType strategyErrorType, String errorMessage) {
        return error(strategyErrorType, errorMessage, AdmissionRejectReason.UNSPECIFIED);
    }

    public static Response error(StrategyErrorType strategyErrorType,
                                 AdmissionRejectReason admissionRejectReason) {
        return error(strategyErrorType, strategyErrorType.getErrorMsg(), admissionRejectReason);
    }

    private static Response error(StrategyErrorType strategyErrorType,
                                  String errorMessage,
                                  AdmissionRejectReason admissionRejectReason) {
        Response result = new Response();
        result.setSuccess(false);
        result.setCode(strategyErrorType.getErrorCode());
        result.setErrorMessage(strategyErrorType.buildErrorMessage(errorMessage));
        result.setAdmissionRejectReason(admissionRejectReason == null
                ? AdmissionRejectReason.UNSPECIFIED : admissionRejectReason);
        return result;
    }

    @Data
    public static class WorkerRoleSummary {
        private int discovered;
        private int alive;
        private long maxQueueTokens;
    }
}
