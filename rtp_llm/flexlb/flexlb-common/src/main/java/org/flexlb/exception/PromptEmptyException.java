package org.flexlb.exception;

import org.flexlb.enums.StatusEnum;

public class PromptEmptyException extends WhaleException {

    @Override
    public boolean isSimpleException() {
        return true;
    }

    @Override
    public boolean isUserException() {
        return true;
    }

    public PromptEmptyException(int code, String name, String message, Throwable cause) {
        super(code, name, message, cause);
    }

    public PromptEmptyException(int code, String name, String message) {
        super(code, name, message);
    }

    public PromptEmptyException(StatusEnum statusEnum) {
        super(statusEnum);
    }
}
