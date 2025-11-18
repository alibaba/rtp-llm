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
    NO_VIT_WORKER(8405);

    private final int errorCode;
    private final String errorMsg;

    StrategyErrorType(int errorCode) {
        this.errorCode = errorCode;
        this.errorMsg = name();
    }

    public static StrategyErrorType valueOf(int errorCode) {
        for (StrategyErrorType type : StrategyErrorType.values()) {
            if (type.getErrorCode() == errorCode) {
                return type;
            }
        }
        return null;
    }

    @Override
    public String toString() {
        return name() + "(" + errorCode + ")";
    }
}
