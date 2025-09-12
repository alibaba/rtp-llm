package org.flexlb.exception;

import org.flexlb.enums.StatusEnum;

public class PromptExceedException extends WhaleException {

    @Override
    public boolean isSimpleException() {
        return true;
    }

    @Override
    public boolean isUserException() {
        return true;
    }

    public PromptExceedException(int code, String name, String message, Throwable cause) {
        super(code, name, message, cause);
    }

    public PromptExceedException(int code, String name, String message) {
        super(code, name, message);
    }

    public PromptExceedException(StatusEnum statusEnum) {
        super(statusEnum);
    }
}
