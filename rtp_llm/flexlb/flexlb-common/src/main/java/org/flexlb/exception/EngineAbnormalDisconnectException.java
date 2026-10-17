package org.flexlb.exception;

public class EngineAbnormalDisconnectException extends FlexLBException {

    public EngineAbnormalDisconnectException(int code, String name, String message, Throwable cause) {
        super(code, name, message, cause);
    }

    public EngineAbnormalDisconnectException(int code, String name, String message) {
        super(code, name, message);
    }
}
