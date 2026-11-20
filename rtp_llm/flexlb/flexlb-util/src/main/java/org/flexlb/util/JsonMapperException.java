package org.flexlb.util;

/**
 * Carries the error code and cause of a JSON conversion failure.
 */
public class JsonMapperException extends FlexLBException {

    public JsonMapperException(int code, String name, String message, Throwable cause) {
        super(code, name, message, cause);
    }

    public JsonMapperException(int code, String name, String message) {
        super(code, name, message);
    }
}
