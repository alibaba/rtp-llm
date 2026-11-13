package org.flexlb.util.network;

import org.flexlb.util.FlexLBException;

/**
 * Reports an unexpected connection closure with its original cause.
 */
public class ConnectionClosedException extends FlexLBException {

    public ConnectionClosedException(int code, String name, String message, Throwable cause) {
        super(code, name, message, cause);
    }

    public ConnectionClosedException(int code, String name, String message) {
        super(code, name, message);
    }
}
