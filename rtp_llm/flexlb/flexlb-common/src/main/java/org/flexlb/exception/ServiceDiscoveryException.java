package org.flexlb.exception;

import org.flexlb.enums.BalanceStatusEnum;

/**
 * Service discovery lookup failed (error or timeout), as opposed to succeeding with an
 * empty host list. Callers that treat discovery presence as a liveness signal must keep
 * their previous worker state when this is thrown.
 */
public class ServiceDiscoveryException extends FlexLBException {

    public ServiceDiscoveryException(BalanceStatusEnum status, String message, Throwable cause) {
        super(status.getCode(), status.name(), message, cause);
    }
}
