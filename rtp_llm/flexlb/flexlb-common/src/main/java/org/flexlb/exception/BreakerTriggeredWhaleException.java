package org.flexlb.exception;

import org.flexlb.enums.StatusEnum;

@SuppressWarnings("unused")
public class BreakerTriggeredWhaleException extends WhaleException {

    @Override
    public boolean isSimpleException() {
        return true;
    }

    @Override
    public boolean isUserException() {
        return true;
    }

    public BreakerTriggeredWhaleException(int code, String name, String message, Throwable cause) {
        super(code, name, message, cause);
    }

    public BreakerTriggeredWhaleException(int code, String name, String message) {
        super(code, name, message);
    }

    public BreakerTriggeredWhaleException(StatusEnum statusEnum) {
        super(statusEnum);
    }
}
