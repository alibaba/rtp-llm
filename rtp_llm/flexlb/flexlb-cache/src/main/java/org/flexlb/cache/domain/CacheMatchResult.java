package org.flexlb.cache.domain;

import org.flexlb.dao.cache.HostCacheMatch;

import java.util.Collections;
import java.util.Map;

/**
 * Cache matches and the block size used to produce them.
 *
 * <p>Each map key is a logical worker identity in {@code ip:port@engineIndex} format and has
 * one {@link HostCacheMatch} containing the raw local/P2P block counts.
 */
public record CacheMatchResult(
        Map<String, HostCacheMatch> hostMatches,
        CacheMatchSource source,
        long queryTimeUs,
        long blockSize) {

    public static CacheMatchResult empty(CacheMatchSource source) {
        return new CacheMatchResult(Collections.emptyMap(), source, 0, 0);
    }

    public static CacheMatchResult failed(CacheMatchSource source, long queryTimeUs) {
        return new CacheMatchResult(Collections.emptyMap(), source, queryTimeUs, 0);
    }

    public static long matchedTokens(double matchBlocks, long blockSize, long inputTokens) {
        if (matchBlocks <= 0 || blockSize <= 0) {
            return 0;
        }
        long matchedTokens = Math.round(blockSize * matchBlocks);
        return inputTokens > 0 ? Math.min(inputTokens, matchedTokens) : matchedTokens;
    }

    /**
     * Returns the match for a logical worker.
     *
     * @param workerIpPort logical worker identity in {@code ip:port@engineIndex} format; the
     *                     index identifies one independently routable engine behind the physical
     *                     frontend
     */
    public HostCacheMatch hostMatch(String workerIpPort) {
        return hostMatches.get(workerIpPort);
    }

}
