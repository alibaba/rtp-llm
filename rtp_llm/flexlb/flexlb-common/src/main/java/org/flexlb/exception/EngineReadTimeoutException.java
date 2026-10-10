package org.flexlb.exception;

public class EngineReadTimeoutException extends FlexLBException {

    public EngineReadTimeoutException(int code, String name, String message, Throwable cause) {
        super(code, name, message, cause);
    }

    public EngineReadTimeoutException(int code, String name, String message) {
        super(code, name, message);
    }
}
