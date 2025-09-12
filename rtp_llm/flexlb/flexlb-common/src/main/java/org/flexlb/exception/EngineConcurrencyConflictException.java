package org.flexlb.exception;


import org.flexlb.enums.StatusEnum;

public class EngineConcurrencyConflictException extends WhaleException {

    @Override
    public boolean isSimpleException() {
        return true;
    }

    @Override
    public boolean isUserException() {
        return true;
    }

    public EngineConcurrencyConflictException(int code, String name, String message, Throwable cause) {
        super(code, name, message, cause);
    }

    public EngineConcurrencyConflictException(int code, String name, String message) {
        super(code, name, message);
    }

    public EngineConcurrencyConflictException(StatusEnum statusEnum) {
        super(statusEnum);
    }
}
