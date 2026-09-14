package org.flexlb.dispatcher;

final class AggregateResponseTooLargeException extends RuntimeException {

    AggregateResponseTooLargeException(long limitBytes) {
        super("aggregate FE response exceeds " + limitBytes + " bytes");
    }
}
