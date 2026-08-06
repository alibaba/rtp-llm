package org.flexlb.dao.loadbalance;

import lombok.Getter;

import java.util.HashMap;
import java.util.Map;

/**
 * FlexLB strategy error codes.
 *
 * <p>Error code ranges encode retry semantics — no separate {@code canRetry} flag:
 * <ul>
 *   <li><b>4000-4999</b> — non-retryable (invalid request, queue full, internal errors)</li>
 *   <li><b>8000-8999</b> — retryable (no available worker, transient failures)</li>
 * </ul>
 *
 * <p>Partition mirror structure: 42xx/44xx/45xx (non-retryable) ↔ 82xx/84xx/85xx (retryable).
 */
@Getter
public enum StrategyErrorType {

    // Auto-TPM preemption (42xx = non-retryable): the request was a running
    // decode victim cancelled by the engine to make room for a strictly
    // higher-priority request. Maps to HTTP 429 / gRPC RESOURCE_EXHAUSTED
    // (Throttling.Aborted).
    AUTO_TPM_PREEMPTED(4290),

    // connect error (42xx = non-retryable mirror of 82xx)
    CONNECT_FAILED(4202),
    CONNECT_TIMEOUT(4203),

    // master schedule error — retryable (84xx)
    NO_AVAILABLE_WORKER(8400),
    NO_PREFILL_WORKER(8402),
    NO_DECODE_WORKER(8403),
    NO_PDFUSION_WORKER(8404),
    NO_VIT_WORKER(8405),
    NO_FRONTEND_WORKER(8407),
    // master schedule error — non-retryable (44xx = mirror of 84xx)
    INVALID_REQUEST(4406),

    // queue error — non-retryable (45xx = mirror of 85xx)
    QUEUE_FULL(4502),
    QUEUE_TIMEOUT(4503),

    // batch dispatch error — retryable (85xx)
    BATCH_DISPATCH_FAILED(8510),
    // batch dispatch error — non-retryable (45xx = mirror of 85xx)
    BATCH_SLO_EXPIRED(4511),
    BATCH_BUILD_FAILED(4512),
    // worker (decode engine) execution failure — non-retryable to prevent retry storms
    // on persistent errors such as OOM or input-too-long.
    WORKER_EXECUTION_FAILED(4513),
    // global inflight TTL expiry for non-batch (QUEUE/DIRECT) requests
    INFLIGHT_TTL_EXPIRED(4514);

    private final int errorCode;
    private final String errorMsg;

    // Cache for O(1) lookup by error code
    private static final Map<Integer, StrategyErrorType> ERROR_CODE_MAP = new HashMap<>();

    static {
        for (StrategyErrorType type : values()) {
            ERROR_CODE_MAP.put(type.errorCode, type);
        }
    }

    StrategyErrorType(int errorCode) {
        this.errorCode = errorCode;
        this.errorMsg = name();
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

    /**
     * Determine if an error code is retryable based on its numeric range.
     * <ul>
     *   <li>8000-8999 — retryable (transient failures, resource unavailable)</li>
     *   <li>4000-4999 — non-retryable (invalid request, queue full, persistent errors)</li>
     * </ul>
     *
     * @param errorCode the error code to check
     * @return {@code true} if the error is in the retryable range (8000-8999)
     */
    public static boolean isRetryableCode(int errorCode) {
        return errorCode >= 8000 && errorCode <= 8999;
    }

    @Override
    public String toString() {
        return name() + "(" + errorCode + ")";
    }
}
