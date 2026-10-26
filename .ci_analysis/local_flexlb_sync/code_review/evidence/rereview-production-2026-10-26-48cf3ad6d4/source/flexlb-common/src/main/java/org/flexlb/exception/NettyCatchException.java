package org.flexlb.exception;

public class NettyCatchException extends FlexLBException {

    public NettyCatchException(int code, String name, String message, Throwable cause) {
        super(code, name, message, cause);
    }

    public NettyCatchException(int code, String name, String message) {
        super(code, name, message);
    }
}
