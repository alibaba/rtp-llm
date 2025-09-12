package org.flexlb.exception;

import org.flexlb.enums.StatusEnum;

public class ConfigException extends WhaleException {
    public ConfigException(int code, String name, String message, Throwable cause) {
        super(code, name, message, cause);
    }

    public ConfigException(int code, String name, String message) {
        super(code, name, message);
    }

    public ConfigException(StatusEnum statusEnum) {
        super(statusEnum);
    }
}
