package org.flexlb.dispatcher;

/** Raised before fanout when repeated request envelopes exceed the request-scoped byte cap. */
public class AggregateRequestTooLargeException extends RuntimeException {

    public AggregateRequestTooLargeException(long limitBytes) {
        super("aggregate FE request payload exceeds " + limitBytes + " bytes");
    }
}
