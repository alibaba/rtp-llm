package org.flexlb.service.grpc;

import org.flexlb.config.ConfigService;
import org.flexlb.dao.master.PrefillChunkLease;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.client.EngineGrpcClient;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Component;

import java.util.concurrent.CompletableFuture;

/** Engine gRPC service for worker and cache status queries. */
@Component
public class EngineGrpcService {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");

    private final EngineGrpcClient engineGrpcClient;

    private final ConfigService configService;

    public EngineGrpcService(EngineGrpcClient engineGrpcClient) {
        this(engineGrpcClient, null);
    }

    @Autowired
    public EngineGrpcService(EngineGrpcClient engineGrpcClient, ConfigService configService) {
        this.engineGrpcClient = engineGrpcClient;
        this.configService = configService;
    }

    public CompletableFuture<EngineRpcService.WorkerStatusPB> getWorkerStatusAsync(String ip,
                                                                                   int grpcPort,
                                                                                   long finishedTaskVersion,
                                                                                   long requestTimeoutMs,
                                                                                   RoleType roleType,
                                                                                   WorkerStatus worker) {
        PrefillChunkLease.Command command;
        worker.lock.lock();
        try {
            if (configService != null && roleType == RoleType.PREFILL) {
                var config = configService.loadBalanceConfig().getRouter().getRoles().getPrefill().getChunkLease();
                boolean controlled = config.isEnabled() && config.getGroups().contains(worker.getGroup());
                if (controlled && worker.prefillChunkLease().command() == null) {
                    worker.prefillChunkLease().change(true);
                    worker.prefillChunkCapacity().freeze();
                } else if (!controlled && worker.prefillChunkLease().command() != null) {
                    worker.prefillChunkLease().change(true);
                }
            }
            command = worker.prefillChunkLease().confirmed() ? null : worker.prefillChunkLease().command();
        } finally {
            worker.lock.unlock();
        }
        if (command == null) {
            return getWorkerStatusAsync(ip, grpcPort, finishedTaskVersion, requestTimeoutMs, roleType);
        }
        EngineRpcService.StatusVersionPB request = EngineRpcService.StatusVersionPB.newBuilder()
                .setLatestFinishedVersion(finishedTaskVersion)
                .setPrefillChunkLease(EngineRpcService.PrefillChunkLeasePB.newBuilder()
                        .setVersion(command.version())
                        .setEnabled(command.enabled()))
                .build();
        return engineGrpcClient.getWorkerStatusAsync(ip, grpcPort, request, requestTimeoutMs);
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
        boolean needCacheKeys = workerStatus.getRole().requiresCacheKeys();
        EngineRpcService.CacheVersionPB request = EngineRpcService.CacheVersionPB.newBuilder()
                .setLatestCacheVersion((int) cacheVersion)
                .setNeedCacheKeys(needCacheKeys)
                .build();
        logger.debug("Get cache status Request: {}, cacheVersion: {}, needCacheKeys: {}",
                ip, cacheVersion, needCacheKeys);

        if (RoleType.VIT.equals(roleType)) {
            return engineGrpcClient.getMultimodalCacheStatusAsync(
                    ip, grpcPort, request, requestTimeoutMs);
        }
        return engineGrpcClient.getCacheStatusAsync(ip, grpcPort, request, requestTimeoutMs);
    }
}
