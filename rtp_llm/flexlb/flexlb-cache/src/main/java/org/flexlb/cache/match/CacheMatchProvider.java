package org.flexlb.cache.match;

import org.flexlb.cache.domain.CacheMatchSource;
import org.flexlb.dao.route.RoleType;
import reactor.core.publisher.Mono;

import java.util.List;
import java.util.Map;

/**
 * Matches request block hashes against one cache metadata source.
 */
public interface CacheMatchProvider {

    CacheMatchSource source();

    Mono<Map<String, Integer>> findMatchingEngines(
            String requestId,
            List<Long> blockCacheKeys,
            long blockSize,
            RoleType roleType,
            String group);
}
