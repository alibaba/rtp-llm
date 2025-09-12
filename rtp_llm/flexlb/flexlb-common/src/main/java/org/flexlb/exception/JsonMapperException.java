package org.flexlb.exception;

import org.flexlb.enums.StatusEnum;

public class JsonMapperException extends WhaleException {
    public JsonMapperException(int code, String name, String message, Throwable cause) {
        super(code, name, message, cause);
    }

    public JsonMapperException(int code, String name, String message) {
        super(code, name, message);
    }

    public JsonMapperException(StatusEnum statusEnum) {
        super(statusEnum);
    }
}
