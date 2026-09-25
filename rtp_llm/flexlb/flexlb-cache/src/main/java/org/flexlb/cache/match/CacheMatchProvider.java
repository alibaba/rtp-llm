package org.flexlb.cache.match;

import org.flexlb.cache.domain.CacheMatchSource;
import org.flexlb.dao.cache.HostCacheMatch;
import org.flexlb.dao.route.RoleType;

import java.util.List;
import java.util.Map;

/**
 * Matches request block hashes against one cache metadata source.
 */
public interface CacheMatchProvider {

    CacheMatchSource source();

    /**
     * Finds cache matches keyed by logical worker identity in
     * {@code ip:port@engineIndex} format. The index identifies one independently routable
     * engine behind the physical frontend.
     */
    Map<String, HostCacheMatch> findMatchingEngines(
            String requestId,
            List<Long> blockCacheKeys,
            long blockSize,
            RoleType roleType,
            String group);
}
