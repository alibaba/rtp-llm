package org.flexlb.exception;

import org.flexlb.enums.StatusEnum;

public class RedisWhaleException extends WhaleException {
    public RedisWhaleException(int code, String name, String message, Throwable cause) {
        super(code, name, message, cause);
    }

    public RedisWhaleException(int code, String name, String message) {
        super(code, name, message);
    }

    public RedisWhaleException(StatusEnum statusEnum) {
        super(statusEnum);
    }
}
