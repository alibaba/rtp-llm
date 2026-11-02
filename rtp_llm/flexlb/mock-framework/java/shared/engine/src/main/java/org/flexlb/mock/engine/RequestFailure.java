package org.flexlb.mock.engine;

/** Scheduling failures carry facts; protocol adapters choose wire codes. */
public final class RequestFailure extends RuntimeException {
    public enum Reason { CANCELLED, TIMEOUT, CAPACITY, OVERLOADED, PREEMPTED, CLOSED, DUPLICATE, INVALID, INTERNAL }
    private final Reason reason;
    public RequestFailure(Reason reason, String message) { this(reason, message, null); }
    public RequestFailure(Reason reason, String message, Throwable cause) {
        super(message, cause); this.reason = java.util.Objects.requireNonNull(reason);
    }
    public Reason reason() { return reason; }
}
