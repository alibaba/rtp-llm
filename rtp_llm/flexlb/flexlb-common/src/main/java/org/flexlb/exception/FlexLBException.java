package org.flexlb.exception;

import lombok.Getter;

public class FlexLBException extends RuntimeException {

    @Getter
    private final int code;

    @Getter
    private final String name;

    public FlexLBException(int code, String name, String message, Throwable cause) {
        super(message, cause);
        this.code = code;
        this.name = name;
    }

    public FlexLBException(int code, String name, String message) {
        super(message);
        this.code = code;
        this.name = name;
    }
}
