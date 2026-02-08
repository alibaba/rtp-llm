package org.flexlb.balance.resource;

import java.util.concurrent.Semaphore;

import org.flexlb.util.Logger;

/**
 * A Semaphore that allows reducing the number of available permits.
 * This is used for dynamic worker capacity control.
 *
 * @author saichen.sm
 * @since 2026/02/02
 */
public class ReducibleSemaphore extends Semaphore {

    public ReducibleSemaphore(int permits) {
        super(permits);
    }

    public ReducibleSemaphore(int permits, boolean fair) {
        super(permits, fair);
    }

    /**
     * Shrinks the number of available permits by the specified reduction.
     * This method is synchronized and safe for concurrent use.
     *
     * @param reduction the number of permits to remove
     */
    public void reducePermits(int reduction) {
        if (reduction <= 0) {
            Logger.warn("Reduction must be positive, got {}", reduction);
            return;
        }
        super.reducePermits(reduction);
    }
}