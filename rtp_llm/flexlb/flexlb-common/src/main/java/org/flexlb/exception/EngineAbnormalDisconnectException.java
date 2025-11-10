package org.flexlb.exception;

import org.flexlb.enums.StatusEnum;

public class EngineAbnormalDisconnectException extends FlexLBException {

    public EngineAbnormalDisconnectException(int code, String name, String message, Throwable cause) {
        super(code, name, message, cause);
    }

    public EngineAbnormalDisconnectException(int code, String name, String message) {
        super(code, name, message);
    }

    public EngineAbnormalDisconnectException(StatusEnum statusEnum) {
        super(statusEnum);
    }
}
