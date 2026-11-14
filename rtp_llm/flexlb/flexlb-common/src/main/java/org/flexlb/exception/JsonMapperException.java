package org.flexlb.exception;

import org.flexlb.enums.StatusEnum;

/**
 * Reports a payload that cannot be mapped to the expected JSON model.
 */
public class JsonMapperException extends FlexLBException {

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
