package org.flexlb.exception;

import org.flexlb.enums.StatusEnum;

@SuppressWarnings("unused")
public class EngineResponseTooLargeException extends WhaleException {
    public EngineResponseTooLargeException(int code, String name, String message, Throwable cause) {
        super(code, name, message, cause);
    }

    public EngineResponseTooLargeException(int code, String name, String message) {
        super(code, name, message);
    }

    public EngineResponseTooLargeException(StatusEnum statusEnum) {
        super(statusEnum);
    }
}
