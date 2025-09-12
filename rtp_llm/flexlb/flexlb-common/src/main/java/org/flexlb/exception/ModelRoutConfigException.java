package org.flexlb.exception;

import org.flexlb.enums.StatusEnum;

public class ModelRoutConfigException extends WhaleException {

    @Override
    public boolean isSimpleException() {
        return true;
    }

    public ModelRoutConfigException(int code, String name, String message, Throwable cause) {
        super(code, name, message, cause);
    }

    public ModelRoutConfigException(int code, String name, String message) {
        super(code, name, message);
    }

    public ModelRoutConfigException(StatusEnum statusEnum) {
        super(statusEnum);
    }
}
