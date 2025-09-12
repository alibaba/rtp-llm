package org.flexlb.exception;

import org.flexlb.enums.StatusEnum;

public class ClientDisconnectWhaleException extends WhaleException {

    @Override
    public boolean isSimpleException() {
        return true;
    }

    public ClientDisconnectWhaleException(int code, String name, String message, Throwable cause) {
        super(code, name, message, cause);
    }

    public ClientDisconnectWhaleException(int code, String name, String message) {
        super(code, name, message);
    }

    public ClientDisconnectWhaleException(StatusEnum statusEnum) {
        super(statusEnum);
    }
}
