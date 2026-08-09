package org.flexlb.dao.pv;

import com.fasterxml.jackson.annotation.JsonInclude;
import lombok.Data;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.route.RoleType;

import java.util.List;
import java.util.Map;

/**
 * PV log data
 */
@Data
public class PvLogData {

    @JsonInclude(JsonInclude.Include.NON_NULL)
    private String requestId;
    @JsonInclude(JsonInclude.Include.NON_NULL)
    private Long seqLen;
    @JsonInclude(JsonInclude.Include.NON_NULL)
    private Long inputIdsCount;
    @JsonInclude(JsonInclude.Include.NON_NULL)
    private Long requestBodyBytes;
    private Response response;
    private String error;
    private boolean success;
    private long enqueueTime;
    private long startTime;
    private long totalUs;
    @JsonInclude(JsonInclude.Include.NON_NULL)
    private Long requestTimeMs;
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

        Request request = ctx.getRequest();
        if (request != null) {
            this.requestId = request.getRequestId();
            this.seqLen = request.getSeqLen();
            this.requestTimeMs = request.getRequestTimeMs();
        }
        this.inputIdsCount = ctx.getInputIdsCount();
        this.requestBodyBytes = ctx.getRequestBodyBytes();
        this.response = ctx.getResponse();
        this.success = ctx.isSuccess();
        this.error = ctx.getErrorMessage();
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
        this.cacheMatchSelections = List.copyOf(ctx.getCacheMatchSelectionByRole().values());
        this.selectionReasons = Map.copyOf(ctx.getSelectionReasonByRole());
        if (!ctx.getShortestTtftDecisionByRole().isEmpty()) {
            this.shortestTtftDecisions = List.copyOf(ctx.getShortestTtftDecisionByRole().values());
        }
    }
}
