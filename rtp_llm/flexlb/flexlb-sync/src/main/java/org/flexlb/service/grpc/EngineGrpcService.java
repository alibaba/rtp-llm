package org.flexlb.service.grpc;

import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.engine.grpc.EngineRpcService;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Component;

import java.util.concurrent.CompletableFuture;

/** Engine gRPC service for worker and cache status queries. */
@Component
public class EngineGrpcService {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");

    private final EngineGrpcClient engineGrpcClient;

    public EngineGrpcService(EngineGrpcClient engineGrpcClient) {
        this.engineGrpcClient = engineGrpcClient;
    }

    public CompletableFuture<EngineRpcService.WorkerStatusPB> getWorkerStatusAsync(
            String ip, int grpcPort, long finishedTaskVersion, long requestTimeoutMs,
            RoleType roleType) {
        EngineRpcService.StatusVersionPB request = EngineRpcService.StatusVersionPB.newBuilder()
                .setLatestFinishedVersion(finishedTaskVersion)
                .build();

        if (RoleType.VIT.equals(roleType)) {
            return engineGrpcClient.getMultimodalWorkerStatusAsync(
                    ip, grpcPort, request, requestTimeoutMs);
        }
        return engineGrpcClient.getWorkerStatusAsync(ip, grpcPort, request, requestTimeoutMs);
    }

    public CompletableFuture<EngineRpcService.CacheStatusPB> getCacheStatusAsync(
            String ip, int grpcPort, WorkerStatus workerStatus, long cacheVersion,
            long requestTimeoutMs, RoleType roleType) {
        boolean needCacheKeys = workerStatus.getRole() == RoleType.PREFILL
                || workerStatus.getRole() == RoleType.PDFUSION;
        EngineRpcService.CacheVersionPB request = EngineRpcService.CacheVersionPB.newBuilder()
                .setLatestCacheVersion((int) cacheVersion)
                .setNeedCacheKeys(needCacheKeys)
                .build();
        logger.info("Get cache status Request: {}, cacheVersion: {}, needCacheKeys: {}",
                ip, cacheVersion, needCacheKeys);

        if (RoleType.VIT.equals(roleType)) {
            return engineGrpcClient.getMultimodalCacheStatusAsync(
                    ip, grpcPort, request, requestTimeoutMs);
        }
        return engineGrpcClient.getCacheStatusAsync(ip, grpcPort, request, requestTimeoutMs);
    }
}
