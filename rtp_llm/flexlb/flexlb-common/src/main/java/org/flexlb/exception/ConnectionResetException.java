package org.flexlb.exception;


import org.flexlb.enums.StatusEnum;

@SuppressWarnings("unused")
public class ConnectionResetException extends WhaleException {

    @Override
    public boolean isSimpleException() {
        return true;
    }

    public ConnectionResetException(int code, String name, String message, Throwable cause) {
        super(code, name, message, cause);
    }

    public ConnectionResetException(int code, String name, String message) {
        super(code, name, message);
    }

    public ConnectionResetException(StatusEnum statusEnum) {
        super(statusEnum);
    }
}
