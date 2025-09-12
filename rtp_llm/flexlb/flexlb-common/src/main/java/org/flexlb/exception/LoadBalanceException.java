package org.flexlb.exception;

import org.flexlb.enums.StatusEnum;

public class LoadBalanceException extends WhaleException {

    @Override
    public boolean isSimpleException() {
        return true;
    }

    public LoadBalanceException(int code, String name, String message, Throwable cause) {
        super(code, name, message, cause);
    }

    public LoadBalanceException(int code, String name, String message) {
        super(code, name, message);
    }

    public LoadBalanceException(StatusEnum statusEnum) {
        super(statusEnum);
    }
}
