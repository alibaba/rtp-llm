package org.flexlb.exception;

import org.flexlb.enums.StatusEnum;

public class NettyCatchException extends FlexLBException {

    public NettyCatchException(int code, String name, String message, Throwable cause) {
        super(code, name, message, cause);
    }

    public NettyCatchException(int code, String name, String message) {
        super(code, name, message);
    }

    public NettyCatchException(StatusEnum statusEnum) {
        super(statusEnum);
    }
}
