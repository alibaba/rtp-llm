package org.flexlb.balance.scheduler;

/**
 * Evidence carried by one placement miss.
 *
 * <p>{@link #POOL_WIDE} is reserved for a snapshot proving that no request can
 * use the observed pool. Every endpoint-, route-, or request-specific miss is
 * {@link #LIMITED}; it must not suppress attempts for other waiting requests.
 */
public enum PlacementBlockScope {
    LIMITED,
    POOL_WIDE;

    boolean stopsPoolScan() {
        return this == POOL_WIDE;
    }
}
