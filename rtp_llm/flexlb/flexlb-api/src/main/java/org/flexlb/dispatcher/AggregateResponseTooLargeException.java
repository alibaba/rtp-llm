package org.flexlb.dispatcher;

/** Signals that one fanout request crossed its aggregate in-memory response budget. */
final class AggregateResponseTooLargeException extends RuntimeException {

    AggregateResponseTooLargeException(long limitBytes) {
        super("aggregate FE response exceeds " + limitBytes + " bytes");
    }
}
