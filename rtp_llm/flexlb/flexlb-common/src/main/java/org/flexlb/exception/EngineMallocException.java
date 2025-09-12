package org.flexlb.exception;

import org.flexlb.enums.StatusEnum;

@SuppressWarnings("unused")
public class EngineMallocException extends WhaleException {
    public EngineMallocException(int code, String name, String message, Throwable cause) {
        super(code, name, message, cause);
    }

    public EngineMallocException(int code, String name, String message) {
        super(code, name, message);
    }

    public EngineMallocException(StatusEnum statusEnum) {
        super(statusEnum);
    }
}
