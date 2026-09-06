package org.flexlb.dao.pv;

import com.fasterxml.jackson.annotation.JsonInclude;
import lombok.Data;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.route.RoleType;

import java.util.List;
import java.util.Map;

/** One completed FlexLB scheduling decision written to {@code pv.log}. */
@Data
public class PvLogData {

    // Keep identifiers numeric on the wire while allowing pre-parse failures to omit them.
    @JsonInclude(JsonInclude.Include.NON_NULL)
    private Long requestId;
    @JsonInclude(JsonInclude.Include.NON_NULL)
    private Long seqLen;
    @JsonInclude(JsonInclude.Include.NON_NULL)
    private Long inputIdsCount;
    @JsonInclude(JsonInclude.Include.NON_NULL)
    private Long requestBodyBytes;
    @JsonInclude(JsonInclude.Include.NON_NULL)
    private Long requestTimeMs;

    // Historical scheduling fields retained for downstream compatibility.
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
    private String realMasterHost;

    // Request-path observability added by the routing observability batch.
    private long totalUs;
    private long arrivalMs;
    private long reqParseUs;
    private long hashWaitUs;
    private long hashUs;
    private String cacheMatchSource;
    private long cacheMatchUs;
    private int cacheMatchCount;
    @JsonInclude(JsonInclude.Include.NON_EMPTY)
    private List<BalanceContext.CacheMatchSelection> cacheMatchSelections;
    @JsonInclude(JsonInclude.Include.NON_EMPTY)
    private Map<RoleType, String> selectionReasons;
    @JsonInclude(JsonInclude.Include.NON_EMPTY)
    private List<ShortestTtftDecision> shortestTtftDecisions;

    public PvLogData(BalanceContext ctx) {
        populateCommonFields(ctx);
    }

    public PvLogData(BalanceContext ctx,
                     int code,
                     String admissionRejectReason,
                     String scheduleOrigin,
                     long batchId,
                     String requestState,
                     String realMasterHost,
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
        this.realMasterHost = realMasterHost;
    }

    private void populateCommonFields(BalanceContext ctx) {
        Request request = ctx.getRequest();
        if (request != null) {
            this.requestId = request.getRequestId();
            this.seqLen = request.getSeqLen();
            this.requestTimeMs = request.getRequestTimeMs();
        }
        this.inputIdsCount = ctx.getInputIdsCount();
        this.requestBodyBytes = ctx.getRequestBodyBytes();
        this.response = ctx.getResponse();
        this.error = ctx.getErrorMessage();
        this.success = ctx.isSuccess();
        this.enqueueTime = ctx.getEnqueueTime();
        this.startTime = ctx.getStartTime();
        this.totalUs = ctx.getTotalTimeUs();
        this.arrivalMs = ctx.getRequestArrivalDelayMs();
        this.reqParseUs = ctx.getRequestBodyReadAndDeserializeTimeUs();
        this.hashWaitUs = ctx.getBlockHashQueueWaitTimeUs();
        this.hashUs = ctx.getBlockHashExecutionTimeUs();
        this.cacheMatchSource = ctx.getCacheMatchSource();
        this.cacheMatchUs = ctx.getCacheMatchQueryTimeUs();
        this.cacheMatchCount = ctx.getCacheMatchQueryCount();
        this.cacheMatchSelections =
                List.copyOf(ctx.getCacheMatchSelectionByRole().values());
        this.selectionReasons = Map.copyOf(ctx.getSelectionReasonByRole());
        if (!ctx.getShortestTtftDecisionByRole().isEmpty()) {
            this.shortestTtftDecisions =
                    List.copyOf(ctx.getShortestTtftDecisionByRole().values());
        }
    }
}
