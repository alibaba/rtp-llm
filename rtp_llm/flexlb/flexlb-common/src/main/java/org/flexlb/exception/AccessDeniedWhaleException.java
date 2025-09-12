package org.flexlb.exception;

import org.flexlb.enums.StatusEnum;

public class AccessDeniedWhaleException extends WhaleException {

    @Override
    public boolean isSimpleException() {
        return true;
    }

    public AccessDeniedWhaleException(int code, String name, String message, Throwable cause) {
        super(code, name, message, cause);
    }

    public AccessDeniedWhaleException(int code, String name, String message) {
        super(code, name, message);
    }

    public AccessDeniedWhaleException(StatusEnum statusEnum) {
        super(statusEnum);
    }
}
