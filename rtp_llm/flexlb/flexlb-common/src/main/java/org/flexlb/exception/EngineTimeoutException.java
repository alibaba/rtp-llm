package org.flexlb.exception;

import org.flexlb.enums.StatusEnum;

@SuppressWarnings("unused")
public class EngineTimeoutException extends WhaleException {
    public EngineTimeoutException(int code, String name, String message, Throwable cause) {
        super(code, name, message, cause);
    }

    public EngineTimeoutException(int code, String name, String message) {
        super(code, name, message);
    }

    public EngineTimeoutException(StatusEnum statusEnum) {
        super(statusEnum);
    }
}
