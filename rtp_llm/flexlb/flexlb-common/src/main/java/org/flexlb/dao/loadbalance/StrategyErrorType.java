package org.flexlb.dao.loadbalance;

import lombok.Getter;

@Getter
public enum StrategyErrorType {

    CONNECT_FAILED(8202, false),
    CONNECT_TIMEOUT(8203, false),

    // master schedule error
    NO_AVAILABLE_WORKER(8400, true),
    NO_PREFILL_WORKER(8402, true),
    NO_DECODE_WORKER(8403, true),
    NO_PDFUSION_WORKER(8404, true),
    NO_VIT_WORKER(8405, true),
    NO_FRONTEND_WORKER(8407, true),
    INVALID_REQUEST(8406, false),

    // queue error
    QUEUE_FULL(8502, false, "TooManyRequests"),
    QUEUE_TIMEOUT(8503, false, "GatewayTimeout"),
    REQUEST_CANCELLED(8504, false),

    // batch dispatch error
    BATCH_DISPATCH_FAILED(8510, true),
    BATCH_SLO_EXPIRED(8511, false),
    BATCH_BUILD_FAILED(8512, false),
    // worker (decode engine) execution failure — non-retryable to prevent retry storms
    // on persistent errors such as OOM or input-too-long.
    WORKER_EXECUTION_FAILED(8513, false),
    // Auto-TPM: request seq_len can never fit one batch's hard token capacity.
    // Explicit failure instead of a silent batcher drop (design doc 8.3).
    BATCH_TOKEN_CAPACITY_EXCEEDED(8514, false),
    // Auto-TPM: plan retries exhausted purely by optimistic-concurrency conflicts
    // (VERSION_MISMATCH / eviction CONFLICT on every attempt, design doc 16.3).
    // Distinct from NO_AVAILABLE_WORKER, which still covers capacity shortage.
    SCHEDULER_PLAN_CONFLICT(8515, false),
    // Auto-TPM victim terminal: an already-admitted request was cancelled by
    // a strictly higher-priority admission attempt.
    PRIORITY_PREEMPTED(8429, false),
    // Auto-TPM incoming rejection caused by a proven priority blocker.  The
    // typed AdmissionRejectReason distinguishes higher- and same-priority.
    PRIORITY_ADMISSION_REJECTED(8430, false),
    // Auto-TPM incoming rejection when no higher/same-priority queue blocker
    // explains the failure and the selected path cannot provide admission
    // capacity within the request budget. This includes hard KV/slot/token
    // limits and dispatch/engine-admission backpressure; it is not limited to
    // CUDA allocation failure.
    RESOURCE_EXHAUSTED(8431, false),
    // Auto-TPM admission is blocked by occupancy whose priority provenance is
    // unavailable, so the Master cannot truthfully attribute the rejection to
    // higher/same priority or to pure resource exhaustion.
    ADMISSION_UNAVAILABLE(8432, false);

    private final int errorCode;
    private final String errorMsg;
    private final boolean canRetry;
    private final String statusName; // DashScope-compatible status_name (null = not set)

    StrategyErrorType(int errorCode, boolean shouldRetry) {
        this(errorCode, shouldRetry, null);
    }

    StrategyErrorType(int errorCode, boolean shouldRetry, String statusName) {
        this.errorCode = errorCode;
        this.errorMsg = name();
        this.canRetry = shouldRetry;
        this.statusName = statusName;
    }

    /**
     * Build the error_message for this error type. When {@code statusName}
     * is non-null, constructs a JSON string containing {@code status_name}
     * and {@code detail} so downstream consumers (including direct gRPC
     * clients that bypass the dash_sc Python layer) can parse the
     * DashScope-compatible status_name. When {@code statusName} is null,
     * returns the plain detail string (backward compatible).
     *
     * @param detail human-readable detail message (null = use enum name)
     * @return the error_message string
     */
    public String buildErrorMessage(String detail) {
        if (statusName == null) {
            return detail == null ? errorMsg : detail;
        }
        String safeDetail = detail == null ? errorMsg : detail;
        String escaped = safeDetail.replace("\\", "\\\\").replace("\"", "\\\"");
        return "{\"status_name\":\"" + statusName
                + "\",\"detail\":\"" + escaped + "\"}";
    }

    @Override
    public String toString() {
        return name() + "(" + errorCode + ")";
    }
}
