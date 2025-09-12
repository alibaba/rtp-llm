package org.flexlb.exception;

import org.flexlb.enums.StatusEnum;

/**
 * Generic external service exception for AI model service errors
 */
public class ExternalServiceException extends WhaleException {

    @Override
    public boolean isSimpleException() {
        return true;
    }

    public ExternalServiceException(int code, String name, String message, Throwable cause) {
        super(code, name, message, cause);
    }

    public ExternalServiceException(int code, String name, String message) {
        super(code, name, message);
    }

    public ExternalServiceException(StatusEnum statusEnum) {
        super(statusEnum);
    }
}
