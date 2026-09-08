package org.flexlb.dao.pv;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.Data;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.route.RoleType;

import java.util.List;
import java.util.Map;

/** One completed FlexLB scheduling decision written to {@code pv.log}. */
@Data
public class PvLogData {

    // Identifiers may be absent when entry parsing fails.
    @JsonInclude(JsonInclude.Include.NON_NULL)
    private String requestId;
    @JsonInclude(JsonInclude.Include.NON_NULL)
    private Long seqLen;
    @JsonInclude(JsonInclude.Include.NON_NULL)
    private Long requestBodyBytes;
    @JsonInclude(JsonInclude.Include.NON_NULL)
    private Long requestTimeMs;

    private Response response;
    private String error;
    private boolean success;
    private long enqueueTime;
    private long startTime;
    private int code;
    private String admissionRejectReason;
    private String scheduleOrigin;
    private int priority;
    private long requestExpiresAtMs;
    private long latencyMs;
    private long batchId;
    private String requestState;

    private long totalUs;
    private Long arrivalMs;
    private Long reqParseUs;
    @JsonIgnoreProperties({"policy", "dispatcher", "worker"})
    private DecisionGroup decisionGroup;
    @JsonInclude(JsonInclude.Include.NON_EMPTY)
    @JsonIgnoreProperties("prefillPolicy")
    private List<RoutingDecision> routingDecisions;
    private String cacheMatchSource;
    private long cacheMatchUs;
    private int cacheMatchCount;
    @JsonInclude(JsonInclude.Include.NON_EMPTY)
    private List<BalanceContext.CacheMatchSelection> cacheMatchSelections;
    @JsonInclude(JsonInclude.Include.NON_EMPTY)
    private Map<RoleType, String> selectionReasons;

    public PvLogData(BalanceContext ctx) {
        populateCommonFields(ctx);
    }

    public PvLogData(BalanceContext ctx,
                     int code,
                     String admissionRejectReason,
                     String scheduleOrigin,
                     long batchId,
                     String requestState,
                     long completedAtMs) {
        populateCommonFields(ctx);
        this.code = code;
        this.admissionRejectReason = admissionRejectReason;
        this.scheduleOrigin = scheduleOrigin;
        this.priority = ctx.getPriority();
        this.requestExpiresAtMs = ctx.getRequestExpiresAtMs();
        this.latencyMs = Math.max(0, completedAtMs - ctx.getStartTime());
        this.batchId = batchId;
        this.requestState = requestState;
    }

    private void populateCommonFields(BalanceContext ctx) {
        BalanceContext.RoutingTelemetry telemetry = ctx.getRoutingTelemetry();
        Request request = ctx.getRequest();
        if (request != null) {
            this.requestId = request.getRequestId();
            this.seqLen = request.getSeqLen();
            this.requestTimeMs = request.getRequestTimeMs();
        }
        this.requestBodyBytes = ctx.getRequestBodyBytes();
        this.decisionGroup = ctx.getDecisionGroup();
        this.routingDecisions = telemetry.routingDecisions().entrySet().stream()
                .sorted(Map.Entry.comparingByKey()).map(Map.Entry::getValue).toList();
        if (ctx.getResponse() != null) {
            Response source = ctx.getResponse();
            Response projected = new PvResponse();
            projected.setServerStatus(source.getServerStatus());
            projected.setQueueLength(source.getQueueLength());
            projected.setWorkerSummary(source.getWorkerSummary());
            this.response = projected;
        }
        this.error = ctx.getErrorMessage();
        this.success = ctx.isSuccess();
        this.enqueueTime = ctx.getEnqueueTime();
        this.startTime = ctx.getStartTime();
        this.totalUs = ctx.getTotalTimeUs();
        this.arrivalMs = ctx.getRequestArrivalDelayMs();
        this.reqParseUs = ctx.getRequestBodyReadAndDeserializeTimeUs();
        this.cacheMatchSource = telemetry.cacheSource();
        this.cacheMatchUs = telemetry.cacheQueryUs();
        this.cacheMatchCount = telemetry.cacheQueryCount();
        this.cacheMatchSelections = telemetry.cacheSelections().entrySet().stream()
                .sorted(Map.Entry.comparingByKey()).map(Map.Entry::getValue)
                .filter(selection -> !hasRecordedCacheSelection(selection)).toList();
        this.selectionReasons = telemetry.selectionReasons().entrySet().stream()
                .filter(entry -> {
                    RoutingDecision decision = telemetry.routingDecisions().get(entry.getKey());
                    return decision == null || !java.util.Objects.equals(decision.selectionReason(), entry.getValue());
                })
                .collect(java.util.stream.Collectors.toMap(Map.Entry::getKey, Map.Entry::getValue));
    }

    private boolean hasRecordedCacheSelection(BalanceContext.CacheMatchSelection selection) {
        return routingDecisions.stream().filter(decision -> decision.role() == selection.role())
                .flatMap(decision -> decision.candidates().stream())
                .anyMatch(candidate -> candidate.selected()
                        && candidate.endpoint().startsWith(selection.selectedIp() + ":")
                        && java.util.Objects.equals(candidate.routingMatchTokens(), selection.hitCacheTokens()));
    }

    @JsonIgnoreProperties({"success", "code", "error_message", "admission_reject_reason",
            "ready", "enqueued_by_master"})
    private static class PvResponse extends Response {
        @Override
        @JsonProperty("server_status")
        @JsonIgnoreProperties("request_id")
        public List<ServerStatus> getServerStatus() {
            return super.getServerStatus();
        }
    }

}
