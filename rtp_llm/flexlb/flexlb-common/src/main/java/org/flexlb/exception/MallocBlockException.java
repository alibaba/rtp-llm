package org.flexlb.exception;

import org.flexlb.enums.StatusEnum;

public class MallocBlockException extends WhaleException {

    @Override
    public boolean isSimpleException() {
        return true;
    }

    public MallocBlockException(int code, String name, String message, Throwable cause) {
        super(code, name, message, cause);
    }

    public MallocBlockException(int code, String name, String message) {
        super(code, name, message);
    }

    public MallocBlockException(StatusEnum statusEnum) {
        super(statusEnum);
    }
}
