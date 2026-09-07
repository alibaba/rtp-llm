package org.flexlb.dao.pv;

import lombok.Data;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;

/** One completed FlexLB scheduling decision written to {@code pv.log}. */
@Data
public class PvLogData {

    // Keep the historical PV fields for downstream compatibility.
    private long requestId;
    private long seqLen;
    private Response response;
    private String error;
    private boolean success;
    private long enqueueTime;
    private long startTime;
    private long requestTimeMs;

    // Minimal gRPC scheduling fields needed for incident correlation.
    private int code;
    private String admissionRejectReason;
    private String scheduleOrigin;
    private int priority;
    private long requestExpiresAtMs;
    private long latencyMs;
    private long batchId;
    private String requestState;
    private String realMasterHost;
    private String sessionAffinityReason;

    public PvLogData(BalanceContext ctx,
                     int code,
                     String admissionRejectReason,
                     String scheduleOrigin,
                     long batchId,
                     String requestState,
                     String realMasterHost,
                     long completedAtMs) {
        this.requestId = ctx.getRequestId();
        this.seqLen = ctx.getRequest().getSeqLen();
        this.response = ctx.getResponse();
        this.error = ctx.getErrorMessage();
        this.success = ctx.isSuccess();
        this.enqueueTime = ctx.getEnqueueTime();
        this.startTime = ctx.getStartTime();
        this.requestTimeMs = ctx.getRequest().getRequestTimeMs();
        this.code = code;
        this.admissionRejectReason = admissionRejectReason;
        this.scheduleOrigin = scheduleOrigin;
        this.priority = ctx.getPriority();
        this.requestExpiresAtMs = ctx.getRequestExpiresAtMs();
        this.latencyMs = Math.max(0, completedAtMs - ctx.getStartTime());
        this.batchId = batchId;
        this.requestState = requestState;
        this.realMasterHost = realMasterHost;
        this.sessionAffinityReason = ctx.getSessionAffinityReason();
    }
}
