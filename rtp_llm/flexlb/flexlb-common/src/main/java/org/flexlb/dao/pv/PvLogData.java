package org.flexlb.dao.pv;

import com.fasterxml.jackson.annotation.JsonInclude;
import lombok.Data;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;

import java.util.List;

/**
 * PV log data
 */
@Data
public class PvLogData {

    private String requestId;
    private long seqLen;
    private Response response;
    private String error;
    private boolean success;
    private long enqueueTime;
    private long startTime;
    private long totalUs;
    private long requestTimeMs;
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
    private List<ShortestTtftDecision> shortestTtftDecisions;

    public PvLogData(BalanceContext ctx) {

        this.requestId = ctx.getRequestId();
        this.seqLen = ctx.getRequest().getSeqLen();
        this.response = ctx.getResponse();
        this.success = ctx.isSuccess();
        this.error = ctx.getErrorMessage();
        this.enqueueTime = ctx.getEnqueueTime();
        this.startTime = ctx.getStartTime();
        this.totalUs = ctx.getTotalTimeUs();
        this.requestTimeMs = ctx.getRequest().getRequestTimeMs();
        this.arrivalMs = ctx.getRequestArrivalDelayMs();
        this.reqParseUs = ctx.getRequestBodyReadAndDeserializeTimeUs();
        this.hashWaitUs = ctx.getBlockHashQueueWaitTimeUs();
        this.hashUs = ctx.getBlockHashExecutionTimeUs();
        this.cacheMatchSource = ctx.getCacheMatchSource();
        this.cacheMatchUs = ctx.getCacheMatchQueryTimeUs();
        this.cacheMatchCount = ctx.getCacheMatchQueryCount();
        this.cacheMatchSelections = List.copyOf(ctx.getCacheMatchSelectionByRole().values());
        if (!ctx.getShortestTtftDecisionByRole().isEmpty()) {
            this.shortestTtftDecisions = List.copyOf(ctx.getShortestTtftDecisionByRole().values());
        }
    }
}
