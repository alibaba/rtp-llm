package org.flexlb.balance.strategy;

/**
 * The selected role has workers, but the request exceeds every worker's
 * known physical capacity and cannot become runnable by waiting.
 */
public final class StaticCapacityExceededException extends RuntimeException {

    public StaticCapacityExceededException(String message) {
        super(message);
    }
}
