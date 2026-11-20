package org.flexlb.cache;

import org.flexlb.config.service.RoleType;

import java.util.List;

/**
 * Client-provided cache keys and block size for one routing decision.
 */
public record CacheMatchQuery(
        String requestId,
        List<Long> blockCacheKeys,
        long blockSize,
        RoleType roleType,
        String group) {}
