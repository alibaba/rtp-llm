package org.flexlb.cache.match.kvcm;

import org.flexlb.cache.domain.CacheMatchSource;
import org.flexlb.cache.match.CacheMatchProvider;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.client.KvcmGrpcClient;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Map;

/**
 * Adapts the KVCM gRPC client to the cache metadata abstraction.
 */
@Component
public class KvcmCacheMatchProvider implements CacheMatchProvider {

    private final KvcmGrpcClient kvcmGrpcClient;

    public KvcmCacheMatchProvider(KvcmGrpcClient kvcmGrpcClient) {
        this.kvcmGrpcClient = kvcmGrpcClient;
    }

    @Override
    public CacheMatchSource source() {
        return CacheMatchSource.KVCM;
    }

    @Override
    public Map<String, Integer> findMatchingEngines(String requestId, List<Long> blockCacheKeys, long blockSize,
                                                    RoleType roleType, String group) {
        return kvcmGrpcClient.findMatchingEngines(requestId, blockCacheKeys, blockSize, roleType, group);
    }
}
