package org.flexlb.cache.domain;

import org.flexlb.dao.cache.HostCacheMatch;

import java.util.Collections;
import java.util.Map;

/**
 * Cache matches and the block size used to produce them.
 *
 * <p>Each worker has one {@link HostCacheMatch} containing the raw local/P2P block counts.
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

    public HostCacheMatch hostMatch(String workerIpPort) {
        return hostMatches.get(workerIpPort);
    }

}
