package org.flexlb.dispatcher;

/** A single FE response or the retained batch responses exceeded their byte budget. */
final class ResponseTooLargeException extends RuntimeException {

    ResponseTooLargeException(long limitBytes) {
        super("FE response byte budget exceeded: " + limitBytes + " bytes");
    }
}
