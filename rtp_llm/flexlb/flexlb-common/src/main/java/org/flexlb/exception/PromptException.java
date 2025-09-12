package org.flexlb.exception;

import org.flexlb.enums.StatusEnum;

public class PromptException extends WhaleException {

    @Override
    public boolean isSimpleException() {
        return true;
    }

    public PromptException(int code, String name, String message, Throwable cause) {
        super(code, name, message, cause);
    }

    public PromptException(int code, String name, String message) {
        super(code, name, message);
    }

    public PromptException(StatusEnum statusEnum) {
        super(statusEnum);
    }
}
