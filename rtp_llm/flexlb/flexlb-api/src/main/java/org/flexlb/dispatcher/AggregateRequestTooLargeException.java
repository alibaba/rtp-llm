package org.flexlb.dispatcher;

public class AggregateRequestTooLargeException extends RuntimeException {

    public AggregateRequestTooLargeException(long limitBytes) {
        super("aggregate FE request payload exceeds " + limitBytes + " bytes");
    }
}
