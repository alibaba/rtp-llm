package org.flexlb.exception;

public class JsonMapperException extends FlexLBException {

    public JsonMapperException(int code, String name, String message, Throwable cause) {
        super(code, name, message, cause);
    }

    public JsonMapperException(int code, String name, String message) {
        super(code, name, message);
    }
}
