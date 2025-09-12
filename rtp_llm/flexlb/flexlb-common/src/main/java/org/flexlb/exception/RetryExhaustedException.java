package org.flexlb.exception;

import org.flexlb.enums.StatusEnum;

public class RetryExhaustedException extends WhaleException {

    @Override
    public boolean isSimpleException() {
        return true;
    }

    @Override
    public boolean isUserException() {
        return true;
    }

    public RetryExhaustedException(int code, String name, String message, Throwable cause) {
        super(code, name, message, cause);
    }

    public RetryExhaustedException(int code, String name, String message) {
        super(code, name, message);
    }

    public RetryExhaustedException(StatusEnum statusEnum) {
        super(statusEnum);
    }
}
