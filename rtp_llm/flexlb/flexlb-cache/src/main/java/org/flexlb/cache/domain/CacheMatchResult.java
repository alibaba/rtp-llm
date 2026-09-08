package org.flexlb.cache.domain;

import org.flexlb.dao.cache.HostCacheMatch;
import org.flexlb.dao.master.WorkerStatus;

import java.util.Collections;
import java.util.Map;

/**
 * Cache matches and the block size used to produce them.
 *
 * <p>Each map key normally is a logical worker identity in {@code ip:port@engineIndex} format
 * and has one {@link HostCacheMatch} containing the raw local/P2P block counts. KVCM responses
 * from legacy single-engine deployments can use the physical {@code ip:port} identity.
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
     * Returns the match stored under the given key, without any KVCM legacy fallback. Routing
     * paths should use {@link #hostMatch(WorkerStatus)} instead.
     *
     * @param workerIpPort worker identity key, typically logical {@code ip:port@engineIndex}
     */
    public HostCacheMatch exactHostMatch(String workerIpPort) {
        return hostMatches.get(workerIpPort);
    }

    /**
     * Returns the match for a worker, preserving legacy KVCM host identities for one engine.
     *
     * <p>KVCM formerly returned a physical {@code ip:port} host identity. A single-engine
     * worker has an unambiguous logical {@code @0} identity, so it can use that legacy entry
     * when no exact logical match exists. Multi-engine workers require an exact logical match.
     */
    public HostCacheMatch hostMatch(WorkerStatus workerStatus) {
        if (workerStatus == null) {
            return null;
        }
        HostCacheMatch exactMatch = exactHostMatch(workerStatus.getLogicalIpPort());
        if (exactMatch != null
                || source != CacheMatchSource.KVCM
                || workerStatus.getMultiEngineNum() != 1) {
            return exactMatch;
        }
        return exactHostMatch(workerStatus.getPhysicalIpPort());
    }

}
