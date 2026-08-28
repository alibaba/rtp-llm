package org.flexlb.balance.endpoint;

/** Delivery cannot start because the selected endpoint generation retired. */
public final class EndpointGenerationRetiredException extends IllegalStateException {

    public EndpointGenerationRetiredException(String message) {
        super(message);
    }
}
