package org.flexlb.exception;

import org.flexlb.enums.StatusEnum;

@SuppressWarnings("unused")
public class EngineReadTimeoutException extends WhaleException {

    @Override
    public boolean isSimpleException() {
        return true;
    }

    @Override
    public boolean isUserException() {
        return true;
    }

    public EngineReadTimeoutException(int code, String name, String message, Throwable cause) {
        super(code, name, message, cause);
    }

    public EngineReadTimeoutException(int code, String name, String message) {
        super(code, name, message);
    }

    public EngineReadTimeoutException(StatusEnum statusEnum) {
        super(statusEnum);
    }
}
