package org.flexlb.exception;

import org.flexlb.enums.StatusEnum;

public class EngineServiceResponseException extends WhaleException {
    public EngineServiceResponseException(int code, String name, String message, Throwable cause) {
        super(code, name, message, cause);
    }

    public EngineServiceResponseException(int code, String name, String message) {
        super(code, name, message);
    }

    public EngineServiceResponseException(StatusEnum statusEnum) {
        super(statusEnum);
    }
}
