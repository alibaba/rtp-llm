package org.flexlb.exception;

import org.flexlb.enums.StatusEnum;

/**
 * Service discovery exception for model service errors
 * Replaces VipServer-specific exceptions with generic service discovery errors
 */
@SuppressWarnings("unused")
public class ServiceDiscoveryException extends WhaleException {

    @Override
    public boolean isSimpleException() {
        return true;
    }

    public ServiceDiscoveryException(int code, String name, String message, Throwable cause) {
        super(code, name, message, cause);
    }

    public ServiceDiscoveryException(int code, String name, String message) {
        super(code, name, message);
    }

    public ServiceDiscoveryException(StatusEnum statusEnum) {
        super(statusEnum);
    }
}
