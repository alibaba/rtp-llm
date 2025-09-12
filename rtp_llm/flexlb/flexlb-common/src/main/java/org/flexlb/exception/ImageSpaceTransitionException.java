package org.flexlb.exception;

import org.flexlb.enums.StatusEnum;

@SuppressWarnings("unused")
public class ImageSpaceTransitionException extends WhaleException {
    public ImageSpaceTransitionException(int code, String name, String message, Throwable cause) {
        super(code, name, message, cause);
    }

    public ImageSpaceTransitionException(int code, String name, String message) {
        super(code, name, message);
    }

    public ImageSpaceTransitionException(StatusEnum statusEnum) {
        super(statusEnum);
    }
}
