package org.flexlb.exception;

import org.flexlb.enums.StatusEnum;

public class EngineOthersException extends WhaleException {

    @Override
    public boolean isSimpleException() {
        return true;
    }

    public EngineOthersException(int code, String name, String message, Throwable cause) {
        super(code, name, message, cause);
    }

    public EngineOthersException(int code, String name, String message) {
        super(code, name, message);
    }

    public EngineOthersException(StatusEnum statusEnum) {
        super(statusEnum);
    }
}
