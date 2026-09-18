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
        return error(strategyErrorType, AdmissionRejectReason.UNSPECIFIED);
    }

    public static Response error(StrategyErrorType strategyErrorType,
                                 AdmissionRejectReason admissionRejectReason) {
        return error(strategyErrorType, admissionRejectReason, null);
    }

    public static Response error(StrategyErrorType strategyErrorType,
                                 AdmissionRejectReason admissionRejectReason,
                                 String detail) {
        if (admissionRejectReason == null) {
            admissionRejectReason = AdmissionRejectReason.UNSPECIFIED;
        }
        if (!strategyErrorType.acceptsAdmissionRejectReason(admissionRejectReason)) {
            throw new IllegalArgumentException("invalid schedule error code/reason: "
                    + strategyErrorType + "/" + admissionRejectReason);
        }
        Response result = new Response();
        result.setSuccess(false);
        result.setCode(strategyErrorType.getErrorCode());
        detail = switch (strategyErrorType) {
            case PRIORITY_ADMISSION_REJECTED -> admissionRejectReason == AdmissionRejectReason.HIGHER_PRIORITY_AHEAD
                    ? "higher-priority requests are ahead" : "same-priority requests are ahead";
            case ADMISSION_UNAVAILABLE -> "admission unavailable; blocker priority attribution is unavailable";
            case RESOURCE_EXHAUSTED -> {
                String standard = "admission capacity is temporarily exhausted";
                yield detail == null || detail.isBlank() ? standard
                        : detail.startsWith(standard) ? detail : standard + "; trigger=" + detail;
            }
            default -> detail;
        };
        result.setErrorMessage(strategyErrorType.buildErrorMessage(detail));
        result.setAdmissionRejectReason(admissionRejectReason);
        return result;
    }

    @Data
    public static class WorkerRoleSummary {
        private int discovered;
        private int alive;
        private long maxQueueTokens;
    }
}
