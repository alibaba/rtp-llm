package org.flexlb.cache.service;

/**
 * Dynamic cache interval interface for cache status checks.
 * Adjusts the prefillCacheStatusCheckInterval based on cache diff statistics
 * to optimize sync efficiency.
 *
 * @author FlexLB
 */
public interface DynamicCacheIntervalService {

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
    long getCurrentIntervalMs();

}
