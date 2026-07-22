package org.flexlb.cache.service.impl;

import lombok.extern.slf4j.Slf4j;
import org.flexlb.cache.domain.WorkerCacheUpdateResult;
import org.flexlb.cache.service.CacheMatchProvider;
import org.flexlb.cache.service.CacheMatchSource;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.client.KvcmGrpcClient;
import org.springframework.stereotype.Component;

import java.util.List;
import java.util.Map;

/**
 * Adapts the KVCM gRPC client to the cache metadata abstraction.
 */
@Slf4j
@Component
public class KvcmCacheMatchProvider implements CacheMatchProvider {

    private static final String UPDATE_NOT_SUPPORTED_MESSAGE =
            "Local cache updates are disabled when KVCM is enabled";

    private final KvcmGrpcClient kvcmGrpcClient;

    public KvcmCacheMatchProvider(KvcmGrpcClient kvcmGrpcClient) {
        this.kvcmGrpcClient = kvcmGrpcClient;
    }

    @Override
    public CacheMatchSource source() {
        return CacheMatchSource.KVCM;
    }

    @Override
    public Map<String, Integer> findMatchingEngines(
            String requestId,
            List<Long> blockCacheKeys,
            long blockSize,
            RoleType roleType,
            String group) {
        return kvcmGrpcClient.findMatchingEngines(
                requestId, blockCacheKeys, blockSize, roleType, group);
    }

    @Override
    public WorkerCacheUpdateResult updateEngineBlockCache(WorkerStatus workerStatus) {
        String engineIpPort = workerStatus == null ? null : workerStatus.getIpPort();
        log.warn("Ignoring engine cache update for {}: {}",
                engineIpPort, UPDATE_NOT_SUPPORTED_MESSAGE);
        return WorkerCacheUpdateResult.builder()
                .success(false)
                .engineIpPort(engineIpPort)
                .errorMessage(UPDATE_NOT_SUPPORTED_MESSAGE)
                .build();
    }
}
