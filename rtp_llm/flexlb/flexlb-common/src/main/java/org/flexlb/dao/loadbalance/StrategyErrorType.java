package org.flexlb.dao.loadbalance;

import lombok.Getter;

import java.util.HashMap;
import java.util.Map;

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
    QUEUE_FULL(8502, false),
    QUEUE_TIMEOUT(8503, false),

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
    // Auto-TPM: queued request preempted by a strictly higher-priority request
    // (429 / Throttling.Aborted semantics, design doc 16.3). Non-retryable
    // against this master — the cluster is saturated with higher-priority work.
    PRIORITY_PREEMPTED(8429, false);

    private final int errorCode;
    private final String errorMsg;
    private final boolean canRetry;

    // Cache for O(1) lookup by error code
    private static final Map<Integer, StrategyErrorType> ERROR_CODE_MAP = new HashMap<>();

    static {
        for (StrategyErrorType type : values()) {
            ERROR_CODE_MAP.put(type.errorCode, type);
        }
    }

    StrategyErrorType(int errorCode, boolean shouldRetry) {
        this.errorCode = errorCode;
        this.errorMsg = name();
        this.canRetry = shouldRetry;
    }

    /**
     * Find StrategyErrorType by error code
     *
     * @param errorCode Error code to search for
     * @return StrategyErrorType if found, null otherwise
     */
    public static StrategyErrorType fromErrorCode(int errorCode) {
        return ERROR_CODE_MAP.get(errorCode);
    }

    @Override
    public String toString() {
        return name() + "(" + errorCode + ")";
    }
}
