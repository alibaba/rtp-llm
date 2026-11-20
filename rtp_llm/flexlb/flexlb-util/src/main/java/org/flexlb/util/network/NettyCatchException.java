package org.flexlb.util.network;

import org.flexlb.util.FlexLBException;

/**
 * Wraps a Netty client failure with the FlexLB error code and original cause.
 */
public class NettyCatchException extends FlexLBException {

    public NettyCatchException(int code, String name, String message, Throwable cause) {
        super(code, name, message, cause);
    }

    public NettyCatchException(int code, String name, String message) {
        super(code, name, message);
    }
}
