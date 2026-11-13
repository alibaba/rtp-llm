package org.flexlb.balance.schedule;

import java.util.Map;

/**
 * Verifies that a fused Worker supplies the complete route without requiring separate Decode capacity.
 */
class PdfusionSchedulingTest {

    private static org.flexlb.cache.CacheMatchResult localMatches(Map<String, Integer> matches) {
        return matches == null
                ? null
                : new org.flexlb.cache.CacheMatchResult(
                        org.flexlb.cache.HostCacheMatch.fromLocalMatches(matches),
                        org.flexlb.cache.CacheMatchSource.LOCAL_SYNC,
                        0L,
                        0L);
    }
}
