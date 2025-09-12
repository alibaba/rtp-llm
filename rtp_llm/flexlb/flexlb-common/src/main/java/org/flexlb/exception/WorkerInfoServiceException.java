package org.flexlb.exception;

import org.flexlb.enums.StatusEnum;

public class WorkerInfoServiceException extends WhaleException {

    @Override
    public boolean isSimpleException() {
        return true;
    }

    public WorkerInfoServiceException(int code, String name, String message, Throwable cause) {
        super(code, name, message, cause);
    }

    public WorkerInfoServiceException(int code, String name, String message) {
        super(code, name, message);
    }

    public WorkerInfoServiceException(StatusEnum statusEnum) {
        super(statusEnum);
    }
}
