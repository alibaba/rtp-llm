package org.flexlb.dao.loadbalance;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;

import java.util.ArrayList;
import java.util.LinkedHashMap;
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

    /** Deep-copy all response data, preserving null collections and entries. */
    public static Response copyOf(Response source) {
        if (source == null) {
            return null;
        }
        Response copy = new Response();
        if (source.serverStatus != null) {
            copy.serverStatus = new ArrayList<>(source.serverStatus.size());
            for (ServerStatus status : source.serverStatus) {
                copy.serverStatus.add(ServerStatus.copyOf(status));
            }
        }
        copy.success = source.success;
        copy.code = source.code;
        copy.errorMessage = source.errorMessage;
        copy.realMasterHost = source.realMasterHost;
        copy.queueLength = source.queueLength;
        copy.enqueuedByMaster = source.enqueuedByMaster;
        if (source.workerSummary != null) {
            copy.workerSummary = new LinkedHashMap<>();
            source.workerSummary.forEach((role, summary) ->
                    copy.workerSummary.put(role, WorkerRoleSummary.copyOf(summary)));
        }
        copy.ready = source.ready;
        copy.admissionRejectReason = source.admissionRejectReason;
        return copy;
    }

    /** Build a successful delivery response without mutating the original route response. */
    public static Response buildSuccessResponse(Response routeResponse, boolean enqueuedByMaster) {
        Response success = copyOf(java.util.Objects.requireNonNull(routeResponse, "routeResponse"));
        success.success = true;
        success.code = 200;
        success.enqueuedByMaster = enqueuedByMaster;
        return success;
    }

    public static Response buildErrorResponse(StrategyErrorType errorType, String message) {
        return error(errorType, errorType == StrategyErrorType.RESOURCE_EXHAUSTED
                ? AdmissionRejectReason.RESOURCE_EXHAUSTED : AdmissionRejectReason.UNSPECIFIED, message);
    }

    public static Response error(StrategyErrorType strategyErrorType) {
        return error(strategyErrorType, strategyErrorType == StrategyErrorType.RESOURCE_EXHAUSTED
                ? AdmissionRejectReason.RESOURCE_EXHAUSTED : AdmissionRejectReason.UNSPECIFIED);
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

        public static WorkerRoleSummary copyOf(WorkerRoleSummary source) {
            if (source == null) {
                return null;
            }
            WorkerRoleSummary copy = new WorkerRoleSummary();
            copy.discovered = source.discovered;
            copy.alive = source.alive;
            copy.maxQueueTokens = source.maxQueueTokens;
            return copy;
        }
    }
}
