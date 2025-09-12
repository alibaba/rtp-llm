package org.flexlb.cache.service;

import java.util.concurrent.atomic.AtomicLong;

/**
 * Dynamic cache interval interface for cache status checks.
 * Adjusts the prefillCacheStatusCheckInterval based on cache diff statistics
 * to optimize sync efficiency.
 * 
 * @author FlexLB
 */
public interface DynamicCacheIntervalService {

    AtomicLong currentIntervalMs = new AtomicLong(100); // Default 100ms

    /**
     * Updates diff statistics and adjusts interval if needed
     *
     * @param diffSize the calculated diff size (added + removed)
     */
    void updateDiffStatistics(int diffSize);

    /**
     * Gets the current cache status check interval
     *
     * @return current interval in milliseconds
     */
    static long getCurrentIntervalMs() {
        return currentIntervalMs.get();
    }

}