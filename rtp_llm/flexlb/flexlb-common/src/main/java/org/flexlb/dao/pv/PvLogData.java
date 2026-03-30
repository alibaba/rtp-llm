package org.flexlb.dao.pv;

import lombok.Data;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;

/**
 * PV log data
 */
@Data
public class PvLogData {

    private long requestId;
    private long seqLen;
    private Response response;
    private String error;
    private boolean success;
    private long enqueueTime;
    private long startTime;
    private long requestTimeMs;

    public PvLogData(BalanceContext ctx) {

        this.requestId = ctx.getRequestId();
        this.seqLen = ctx.getRequest().getSeqLen();
        this.response = ctx.getResponse();
        this.success = ctx.isSuccess();
        this.error = ctx.getErrorMessage();
        this.enqueueTime = ctx.getEnqueueTime();
        this.startTime = ctx.getStartTime();
        this.requestTimeMs = ctx.getRequest().getRequestTimeMs();
    }
}