package org.flexlb.exception;

import org.flexlb.enums.StatusEnum;

public class EngineInternalException extends WhaleException {

    @Override
    public boolean isSimpleException() {
        return true;
    }

    @Override
    public boolean isUserException() {
        return true;
    }

    public EngineInternalException(int code, String name, String message, Throwable cause) {
        super(code, name, message, cause);
    }

    public EngineInternalException(int code, String name, String message) {
        super(code, name, message);
    }

    public EngineInternalException(StatusEnum statusEnum) {
        super(statusEnum);
    }
}
