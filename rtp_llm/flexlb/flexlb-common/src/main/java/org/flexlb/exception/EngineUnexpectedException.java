package org.flexlb.exception;

import org.flexlb.enums.StatusEnum;

public class EngineUnexpectedException extends WhaleException {

    @Override
    public boolean isSimpleException() {
        return true;
    }

    public EngineUnexpectedException(int code, String name, String message, Throwable cause) {
        super(code, name, message, cause);
    }

    public EngineUnexpectedException(int code, String name, String message) {
        super(code, name, message);
    }

    public EngineUnexpectedException(StatusEnum statusEnum) {
        super(statusEnum);
    }
}
