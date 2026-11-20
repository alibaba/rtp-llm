package org.flexlb.util.network;

import org.flexlb.util.FlexLBException;

/**
 * Reports a timed-out remote response with its error code and cause.
 */
public class ResponseTimeoutException extends FlexLBException {

    public ResponseTimeoutException(int code, String name, String message, Throwable cause) {
        super(code, name, message, cause);
    }

    public ResponseTimeoutException(int code, String name, String message) {
        super(code, name, message);
    }
}
