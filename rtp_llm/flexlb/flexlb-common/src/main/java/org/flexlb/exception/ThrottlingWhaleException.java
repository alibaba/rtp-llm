package org.flexlb.exception;

import org.flexlb.enums.StatusEnum;

public class ThrottlingWhaleException extends WhaleException {

    @Override
    public boolean isSimpleException() {
        return true;
    }

    @Override
    public boolean isUserException() {
        return true;
    }

    public ThrottlingWhaleException(int code, String name, String message, Throwable cause) {
        super(code, name, message, cause);
    }

    public ThrottlingWhaleException(int code, String name, String message) {
        super(code, name, message);
    }

    public ThrottlingWhaleException(StatusEnum statusEnum) {
        super(statusEnum);
    }
}
