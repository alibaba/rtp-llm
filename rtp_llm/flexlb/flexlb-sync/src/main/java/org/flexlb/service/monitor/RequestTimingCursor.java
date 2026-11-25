package org.flexlb.service.monitor;

/**
 * Incremental cursor and in-flight RPC state owned by the collection thread.
 */
final class RequestTimingCursor {
    String epoch = "";
    long sequence;
    boolean inFlight;
    int pagesThisTick;
    long lastFailureLogMs;
}
