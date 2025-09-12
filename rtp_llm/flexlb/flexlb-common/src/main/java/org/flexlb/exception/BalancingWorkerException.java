package org.flexlb.exception;

import org.flexlb.enums.StatusEnum;

public class BalancingWorkerException extends WhaleException {
    public BalancingWorkerException(int code, String name, String message, Throwable cause) {
        super(code, name, message, cause);
    }

    public BalancingWorkerException(int code, String name, String message) {
        super(code, name, message);
    }

    public BalancingWorkerException(StatusEnum statusEnum) {
        super(statusEnum);
    }
}
