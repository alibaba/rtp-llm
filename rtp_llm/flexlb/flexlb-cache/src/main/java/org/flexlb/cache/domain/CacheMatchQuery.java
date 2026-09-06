package org.flexlb.cache.domain;

import org.flexlb.dao.route.RoleType;

import java.util.List;

/**
 * Provider-specific cache keys for one routing decision.
 */
public record CacheMatchQuery(
        String requestId,
        List<Long> blockCacheKeys,
        long blockSize,
        List<Long> localStandbyBlockCacheKeys,
        long localStandbyBlockSize,
        RoleType roleType,
        String group) {
}
