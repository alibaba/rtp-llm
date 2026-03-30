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
    INVALID_REQUEST(8406, false),

    // queue error
    QUEUE_FULL(8502, false),
    QUEUE_TIMEOUT(8503, false),
    REQUEST_CANCELLED(8504, false);

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
