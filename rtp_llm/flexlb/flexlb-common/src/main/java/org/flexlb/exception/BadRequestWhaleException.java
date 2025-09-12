package org.flexlb.exception;

import org.flexlb.enums.StatusEnum;

public class BadRequestWhaleException extends WhaleException {

    @Override
    public boolean isSimpleException() {
        return true;
    }


    public BadRequestWhaleException(int code, String name, String message, Throwable cause) {
        super(code, name, message, cause);
    }

    public BadRequestWhaleException(int code, String name, String message) {
        super(code, name, message);
    }

    public BadRequestWhaleException(StatusEnum statusEnum) {
        super(statusEnum);
    }
}
