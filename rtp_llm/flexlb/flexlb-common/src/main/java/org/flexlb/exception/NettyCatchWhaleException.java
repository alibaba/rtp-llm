package org.flexlb.exception;

import org.flexlb.enums.StatusEnum;

public class NettyCatchWhaleException extends WhaleException {
    public NettyCatchWhaleException(int code, String name, String message, Throwable cause) {
        super(code, name, message, cause);
    }

    public NettyCatchWhaleException(int code, String name, String message) {
        super(code, name, message);
    }

    public NettyCatchWhaleException(StatusEnum statusEnum) {
        super(statusEnum);
    }
}
