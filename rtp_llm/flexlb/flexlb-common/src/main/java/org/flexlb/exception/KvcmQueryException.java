package org.flexlb.exception;

public class KvcmQueryException extends RuntimeException {

    public KvcmQueryException(String message) {
        super(message);
    }

    public KvcmQueryException(String message, Throwable cause) {
        super(message, cause);
    }
}
