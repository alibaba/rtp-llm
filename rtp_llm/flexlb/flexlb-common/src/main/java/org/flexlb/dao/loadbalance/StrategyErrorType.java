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
    // SLO-aware admission rejection — the scheduler predicts the request cannot
    // finish within its SLO even if admitted (est + wait + batcherWait > slo - margin),
    // so it is rejected before occupying any queue slot. Same non-retryable segment
    // style as QUEUE_FULL. NOTE: production frontend retry semantics for this code
    // still need to be aligned before rollout.
    SLO_REJECTED(8504, false),

    // batch dispatch error
    BATCH_DISPATCH_FAILED(8510, true),
    BATCH_SLO_EXPIRED(8511, false),
    BATCH_BUILD_FAILED(8512, false),
    // worker (decode engine) execution failure — non-retryable to prevent retry storms
    // on persistent errors such as OOM or input-too-long.
    WORKER_EXECUTION_FAILED(8513, false);

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
