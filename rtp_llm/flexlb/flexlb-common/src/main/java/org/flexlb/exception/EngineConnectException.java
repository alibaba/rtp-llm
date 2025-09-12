package org.flexlb.exception;

import org.flexlb.enums.StatusEnum;

public class EngineConnectException extends WhaleException {

    @Override
    public boolean isSimpleException() {
        return true;
    }

    @Override
    public boolean isUserException() {
        return true;
    }

    public EngineConnectException(int code, String name, String message, Throwable cause) {
        super(code, name, message, cause);
    }

    public EngineConnectException(int code, String name, String message) {
        super(code, name, message);
    }

    public EngineConnectException(StatusEnum statusEnum) {
        super(statusEnum);
    }
}
