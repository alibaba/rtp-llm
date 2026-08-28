package org.flexlb.balance.strategy;

/** A request exceeds every worker's known physical capacity. */
public final class StaticCapacityExceededException extends RuntimeException {

    public StaticCapacityExceededException(String message) {
        super(message);
    }
}
