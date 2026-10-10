package org.flexlb.dao.pv;

import lombok.Getter;
import org.flexlb.balance.scheduler.RequestContext;
import org.flexlb.dao.loadbalance.Response;

import java.util.Map;

/** One completed FlexLB scheduling decision written to {@code pv.log}. */
@Getter
public class PvLogData {

    // Keep the historical PV fields for downstream compatibility.
    private final long requestId;
    private final long seqLen;
    private final Response response;
    private final String error;
    private final boolean success;
    private final long enqueueTime;
    private final long startTime;
    private final long requestTimeMs;

    // Minimal gRPC scheduling fields needed for incident correlation.
    private final int code;
    private final String admissionRejectReason;
    private final String scheduleOrigin;
    private final int priority;
    private final long requestExpiresAtMs;
    private final long latencyMs;
    private final long batchId;
    private final String requestState;
    private final String realMasterHost;
    private final Map<String, Object> schedulingDiagnostics;

    public PvLogData(RequestContext ctx,
                     boolean success,
                     String error,
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
        this.error = success ? null : error;
        this.success = success;
        this.schedulingDiagnostics = success ? null : ctx.getSchedulingDiagnostics();
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
    }
}
