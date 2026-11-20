package org.flexlb.util;

import lombok.Getter;

/**
 * Carries FlexLB error details across internal service boundaries.
 */
public class FlexLBException extends RuntimeException {

    @Getter private final int code;

    @Getter private final String name;

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
