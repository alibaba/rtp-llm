package org.flexlb.dao.loadbalance;

import lombok.Getter;

@Getter
public enum StrategyErrorType {

    CONNECT_FAILED(8202),
    CONNECT_TIMEOUT(8203),

    // master schedule error
    NO_AVAILABLE_WORKER(8400),
    NO_PREFILL_WORKER(8402),
    NO_DECODE_WORKER(8403),
    NO_PDFUSION_WORKER(8404),
    NO_VIT_WORKER(8405),
    INVALID_REQUEST(8406),

    // queue error
    QUEUE_FULL(8502),
    QUEUE_TIMEOUT(8503),
    REQUEST_CANCELLED(8504);

    private final int errorCode;
    private final String errorMsg;

    StrategyErrorType(int errorCode) {
        this.errorCode = errorCode;
        this.errorMsg = name();
    }

    @Override
    public String toString() {
        return name() + "(" + errorCode + ")";
    }
}
