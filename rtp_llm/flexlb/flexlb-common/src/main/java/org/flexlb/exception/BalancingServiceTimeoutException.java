package org.flexlb.exception;

import org.flexlb.enums.StatusEnum;

public class BalancingServiceTimeoutException extends WhaleException {

    @Override
    public boolean isSimpleException() {
        return true;
    }

    public BalancingServiceTimeoutException(int code, String name, String message, Throwable cause) {
        super(code, name, message, cause);
    }

    public BalancingServiceTimeoutException(int code, String name, String message) {
        super(code, name, message);
    }

    public BalancingServiceTimeoutException(StatusEnum statusEnum) {
        super(statusEnum);
    }
}
