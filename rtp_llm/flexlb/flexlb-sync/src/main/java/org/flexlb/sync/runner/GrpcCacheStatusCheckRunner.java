package org.flexlb.sync.runner;

import org.flexlb.cache.domain.WorkerCacheUpdateResult;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.cache.service.DynamicCacheIntervalService;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.enums.BalanceStatusEnum;
import org.flexlb.service.grpc.EngineGrpcService;
import org.flexlb.service.grpc.EngineStatusConverter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.WorkerGenerationFence;
import org.flexlb.util.CommonUtils;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.Executor;
import java.util.concurrent.atomic.LongAdder;

import static org.flexlb.constant.CommonConstants.DEADLINE_EXCEEDED_MESSAGE;

/** Fetches and publishes one WorkerStatus generation's cache locality. */
public class GrpcCacheStatusCheckRunner implements Runnable {

    private enum CacheApplyResult {
        STALE_GENERATION,
        UNCHANGED,
        APPLIED,
        PUBLICATION_FAILED
    }

    private final String ipPort;
    private final String modelName;
    private final RoleType roleType;
    private final WorkerStatus workerStatus;
    private final ConcurrentMap<String, WorkerStatus> workerStatusMap;
    private final EngineHealthReporter engineHealthReporter;
    private final EngineGrpcService engineGrpcService;
    private final CacheAwareService cacheAwareService;
    private final WorkerGenerationFence generationFence;
    private final String ip;
    private final int grpcPort;
    private final long requestTimeoutMs;
    private final LongAdder syncCount;
    private final long syncEngineStatusInterval;
    private final Executor callbackExecutor;

    public GrpcCacheStatusCheckRunner(String modelName, String ipPort, RoleType roleType,
                                      WorkerStatus workerStatus,
                                      EngineHealthReporter engineHealthReporter,
                                      EngineGrpcService engineGrpcService,
                                      CacheAwareService cacheAwareService,
                                      ConcurrentMap<String, WorkerStatus> workerStatusMap,
                                      WorkerGenerationFence generationFence,
                                      long requestTimeoutMs,
                                      LongAdder syncCount,
                                      long syncEngineStatusInterval,
                                      Executor callbackExecutor) {
        this.ipPort = ipPort;
        int separator = ipPort.lastIndexOf(':');
        this.ip = ipPort.substring(0, separator);
        this.grpcPort = CommonUtils.toGrpcPort(
                Integer.parseInt(ipPort.substring(separator + 1)));
        this.roleType = roleType;
        this.modelName = modelName;
        this.workerStatus = workerStatus;
        this.workerStatusMap = workerStatusMap;
        this.engineHealthReporter = engineHealthReporter;
        this.engineGrpcService = engineGrpcService;
        this.cacheAwareService = cacheAwareService;
        this.generationFence = generationFence;
        this.requestTimeoutMs = requestTimeoutMs;
        this.syncCount = syncCount;
        this.syncEngineStatusInterval = syncEngineStatusInterval;
        this.callbackExecutor = callbackExecutor;
    }

    @Override
    public void run() {
        boolean callbackInstalled = false;
        try {
            long cacheIntervalMs = DynamicCacheIntervalService.getCurrentIntervalMs();
            long roundInterval = Math.max(cacheIntervalMs / syncEngineStatusInterval, 1);
            if (isCacheRole() && syncCount.longValue() % roundInterval != 0) {
                return;
            }

            long startTimeUs = System.nanoTime() / 1000;
            CompletableFuture<EngineRpcService.CacheStatusPB> rpc =
                    engineGrpcService.getCacheStatusAsync(
                            ip, grpcPort, workerStatus, currentCacheVersion(),
                            requestTimeoutMs, roleType);
            CompletableFuture<EngineRpcService.CacheStatusPB> callback = rpc.whenCompleteAsync(
                    (response, failure) -> {
                        if (failure != null) {
                            handleFailure(unwrap(failure));
                            return;
                        }
                        handleResponse(response, startTimeUs);
                    }, callbackExecutor);
            callback.whenComplete((ignored, callbackFailure) ->
                    workerStatus.getCacheCheckInProgress().set(false));
            callbackInstalled = true;
        } finally {
            if (!callbackInstalled) {
                workerStatus.getCacheCheckInProgress().set(false);
            }
        }
    }

    private void handleResponse(EngineRpcService.CacheStatusPB response, long startTimeUs) {
        if (response == null) {
            if (isCurrentGeneration()) {
                engineHealthReporter.reportCacheStatusCheckerFail(
                        modelName, BalanceStatusEnum.RESPONSE_NULL, roleType);
            }
            return;
        }
        engineHealthReporter.reportCacheStatusCheckRemoteInfo(
                modelName, roleType.name(), startTimeUs);

        CacheApplyResult result = generationFence.read(ipPort, () -> {
            if (workerStatusMap.get(ipPort) != workerStatus) {
                return CacheApplyResult.STALE_GENERATION;
            }
            CacheStatus current = workerStatus.getCacheStatus();
            if (current != null && response.getVersion() == current.getVersion()) {
                workerStatus.getCacheLastUpdateTime().set(System.nanoTime() / 1000);
                return CacheApplyResult.UNCHANGED;
            }

            if (isCacheRole()) {
                WorkerCacheUpdateResult publication;
                try {
                    publication = cacheAwareService.publishEngineCacheSnapshot(
                            ipPort, roleType, response.getCacheKeysMap().keySet());
                } catch (RuntimeException publicationFailure) {
                    return CacheApplyResult.PUBLICATION_FAILED;
                }
                if (publication == null || !publication.isSuccess()) {
                    return CacheApplyResult.PUBLICATION_FAILED;
                }
            }

            // Cache version is a commit marker: publish locality first, then
            // store lightweight metadata. The protobuf key view is never kept.
            workerStatus.setCacheStatus(
                    EngineStatusConverter.convertToCacheStatusMetadata(response));
            workerStatus.getCacheLastUpdateTime().set(System.nanoTime() / 1000);
            return CacheApplyResult.APPLIED;
        });

        if (result == CacheApplyResult.PUBLICATION_FAILED) {
            reportPublicationFailure();
        } else if (result == CacheApplyResult.APPLIED
                || result == CacheApplyResult.UNCHANGED) {
            engineHealthReporter.reportCacheStatusCheckerSuccess(modelName, workerStatus);
        }
    }

    private void reportPublicationFailure() {
        engineHealthReporter.reportCacheStatusCheckerFail(
                modelName, BalanceStatusEnum.CACHE_UPDATE_FAILED, roleType);
    }

    private void handleFailure(Throwable failure) {
        if (!isCurrentGeneration()) {
            return;
        }
        String message = failure == null ? null : failure.getMessage();
        BalanceStatusEnum status = message != null
                && message.toLowerCase().contains(DEADLINE_EXCEEDED_MESSAGE.toLowerCase())
                ? BalanceStatusEnum.CACHE_GRPC_TIMEOUT
                : BalanceStatusEnum.CACHE_SERVICE_UNAVAILABLE;
        engineHealthReporter.reportCacheStatusCheckerFail(modelName, status, roleType);
    }

    private boolean isCurrentGeneration() {
        return generationFence.read(
                ipPort, () -> workerStatusMap.get(ipPort) == workerStatus);
    }

    private long currentCacheVersion() {
        CacheStatus current = workerStatus.getCacheStatus();
        return current == null ? -1L : current.getVersion();
    }

    private boolean isCacheRole() {
        return roleType == RoleType.PREFILL || roleType == RoleType.PDFUSION;
    }

    private static Throwable unwrap(Throwable failure) {
        return failure instanceof CompletionException && failure.getCause() != null
                ? failure.getCause() : failure;
    }
}
