package org.flexlb.sync.runner;

import org.flexlb.cache.domain.WorkerCacheUpdateResult;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.cache.service.DynamicCacheIntervalService;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.BalanceStatusEnum;
import org.flexlb.service.grpc.EngineGrpcService;
import org.flexlb.service.grpc.EngineStatusConverter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.WorkerDirectory;
import org.flexlb.util.CommonUtils;
import org.flexlb.util.IdUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Optional;
import java.util.concurrent.CompletionException;
import java.util.concurrent.Executor;
import java.util.concurrent.atomic.LongAdder;

import static org.flexlb.constant.CommonConstants.DEADLINE_EXCEEDED_MESSAGE;

public class GrpcCacheStatusCheckRunner implements Runnable {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");

    private final String ipPort;
    private final String modelName;
    private final String site;
    private final RoleType roleType;
    private final WorkerStatus workerStatus;
    private final WorkerDirectory workerDirectory;
    private final WorkerStatus.PollLease pollLease;
    private final long generationId;
    private final EngineHealthReporter engineHealthReporter;
    private final EngineGrpcService engineGrpcService;
    private final CacheAwareService cacheAwareService;
    private final DynamicCacheIntervalService cacheIntervalService;
    private final String ip;
    private final int grpcPort;
    private final long startTime = System.nanoTime() / 1000;
    private final String id = IdUtils.fastUuid();
    private final boolean debug;
    private final long requestTimeoutMs;
    private final LongAdder syncCount;
    private final Long syncEngineStatusInterval;
    private final Executor callbackExecutor;

    public GrpcCacheStatusCheckRunner(String modelName, String ipPort, String site, RoleType roleType,
                                      WorkerStatus workerStatus,
                                      WorkerStatus.PollLease pollLease,
                                      WorkerDirectory workerDirectory,
                                      EngineHealthReporter engineHealthReporter,
                                      EngineGrpcService engineGrpcService,
                                      CacheAwareService cacheAwareService,
                                      DynamicCacheIntervalService cacheIntervalService,
                                      long requestTimeoutMs,
                                      LongAdder syncCount,
                                      Long syncEngineStatusInterval,
                                      boolean fullSnapshotDebugMode,
                                      Executor callbackExecutor) {

        this.ipPort = ipPort;
        String[] split = ipPort.split(":");
        this.ip = split[0];
        this.roleType = roleType;
        this.grpcPort = CommonUtils.toGrpcPort(Integer.parseInt(split[1]));
        this.modelName = modelName;
        this.workerStatus = workerStatus;
        this.workerDirectory = java.util.Objects.requireNonNull(
                workerDirectory, "workerDirectory");
        this.pollLease = java.util.Objects.requireNonNull(
                pollLease, "pollLease");
        workerStatus.requireCachePollLease(pollLease);
        this.generationId = workerStatus.getGenerationId();
        this.site = site;
        this.engineHealthReporter = engineHealthReporter;
        this.engineGrpcService = engineGrpcService;
        this.cacheAwareService = cacheAwareService;
        this.cacheIntervalService = java.util.Objects.requireNonNull(
                cacheIntervalService, "cacheIntervalService");
        this.debug = fullSnapshotDebugMode;
        this.requestTimeoutMs = requestTimeoutMs;
        this.syncCount = syncCount;
        this.syncEngineStatusInterval = syncEngineStatusInterval;
        this.callbackExecutor = callbackExecutor;
    }

    @Override
    public void run() {
        boolean asyncInitiated = false;
        try {
            logger.debug("GrpcCacheStatusCheckRunner run for {}", ipPort);
            long prefillCacheStatusCheckInterval =
                    DynamicCacheIntervalService.getCurrentIntervalMs();
            long roundInterval = prefillCacheStatusCheckInterval / syncEngineStatusInterval;
            roundInterval = Math.max(roundInterval, 1);

            // EngineSyncRunner is invoked on the status-poll cadence. Prefill
            // cache polling uses an integer number of those ticks so a changing
            // interval does not require a second scheduler or timer.
            if ((RoleType.PREFILL.equals(roleType) || RoleType.PDFUSION.equals(roleType))
                        && syncCount.longValue() % roundInterval != 0) {
                logger.debug("Skip prefill cache status check for {} because not in {}ms interval", ipPort, prefillCacheStatusCheckInterval);
                return; // finally will reset the flag
            }

            long startTime = System.nanoTime() / 1000;
            long currentCacheVersion = getCurrentCacheVersion();

            engineGrpcService.getCacheStatusAsync(ip, grpcPort, workerStatus, currentCacheVersion,
                            requestTimeoutMs, roleType)
                    .thenApply(cacheStatusPB -> {
                        logger.debug("gRPC Cache Status Response - handled for {}, role:{}, cache_key_size:{}, cache_version:{}, "
                                        + "available_kv_cache:{}, total_kv_cache:{}, block_size:{}",
                                ipPort, roleType.name(), cacheStatusPB.getCacheKeysMap().size(), cacheStatusPB.getVersion(),
                                cacheStatusPB.getAvailableKvCache(), cacheStatusPB.getTotalKvCache(), cacheStatusPB.getBlockSize());
                        return EngineStatusConverter.convertToCacheStatus(cacheStatusPB);
                    })
                    .handleAsync((cacheStatus, failure) -> {
                        try {
                            Throwable cause = unwrapCompletionFailure(failure);
                            if (cause != null) {
                                handleException(cause);
                                // Return a default CacheStatus with error information
                                CacheStatus errorStatus = CacheStatus.builder()
                                        .version(-1)
                                        .availableKvCache(0)
                                        .totalKvCache(0)
                                        .blockSize(0)
                                        .message("Cache Status gRPC call failed: "
                                                + cause.getMessage())
                                        .build();
                                handleCacheStatusResponse(errorStatus, startTime);
                            } else {
                                handleCacheStatusResponse(cacheStatus, startTime);
                            }
                        } catch (Throwable callbackFailure) {
                            logger.error("Cache status callback failed for {}",
                                    ipPort, callbackFailure);
                        }
                        return null;
                    }, callbackExecutor)
                    .whenComplete((ignored, callbackFailure) -> {
                        pollLease.close();
                        if (callbackFailure != null) {
                            logger.error(
                                    "Cache status callback was not scheduled for {}",
                                    ipPort,
                                    unwrapCompletionFailure(callbackFailure));
                        }
                    });
            asyncInitiated = true;
        } finally {
            if (!asyncInitiated) {
                pollLease.close();
            }
        }
    }

    private static Throwable unwrapCompletionFailure(Throwable failure) {
        return failure instanceof CompletionException
                && failure.getCause() != null
                ? failure.getCause() : failure;
    }

    private void handleCacheStatusResponse(CacheStatus newCacheStatus, long startTime) {

        try {
            logger.debug("gRPC Cache Status - handled for {}, role:{}", ipPort, roleType.name());

            if (newCacheStatus.getMessage() != null) {
                logger.debug("gRPC Cache Status - {}, role:{}, message:{}", ipPort, roleType.name(), newCacheStatus.getMessage());
                return;
            }

            long successfulIntervalUs;
            workerStatus.lock.lock();
            try {
                if (!workerDirectory.isCurrentStatus(
                        roleType, ipPort, workerStatus)
                        || !workerStatus.isActiveGeneration()) {
                    logger.debug(
                            "Ignore stale cache callback for {}#{}, role:{}",
                            ipPort, generationId, roleType);
                    return;
                }
                if (validateCacheStatusResponse(workerStatus, newCacheStatus)) {
                    workerStatus.publishCacheStatus(newCacheStatus);
                    if (isCacheProducingRole()) {
                        // CacheAwareService is keyed by address, not generation.
                        // Keep the generation lock through this in-memory index
                        // update so retirement cannot publish a replacement or
                        // clear the address between validation and publication.
                        updateLocalKvCache();
                    }
                    logCacheStatusUpdate(newCacheStatus, startTime);
                }
                successfulIntervalUs =
                        workerStatus.recordSuccessfulCachePoll();
            } finally {
                workerStatus.lock.unlock();
            }

            engineHealthReporter.reportCacheStatusCheckRemoteInfo(
                    modelName, roleType.name(), startTime);
            engineHealthReporter.reportCacheStatusCheckerSuccess(
                    modelName, workerStatus, successfulIntervalUs);
        } catch (Throwable e) {
            log("engine cache status check via gRPC exception, msg: " + e.getMessage(), e);
            engineHealthReporter.reportCacheStatusCheckerFail(
                    modelName, BalanceStatusEnum.CACHE_SERVICE_UNAVAILABLE, roleType);
        }
    }

    private boolean validateCacheStatusResponse(WorkerStatus workerStatus, CacheStatus newCacheStatus) {
        if (debug) {
            return true;
        }
        CacheStatus currentCacheStatus = workerStatus.getCacheStatus();
        if (currentCacheStatus != null && newCacheStatus.getVersion() <= currentCacheStatus.getVersion()) {
            logger.debug("gRPC Cache Status - {}, role:{}, version not updated, current: {}, response: {}",
                    ipPort, roleType.name(), currentCacheStatus.getVersion(), newCacheStatus.getVersion());
            return false;
        }
        return true;
    }

    private void logCacheStatusUpdate(CacheStatus cacheStatus, long startTime) {

        logger.debug("gRPC Cache Status - {}, role:{}, block_size:{}, version:{}, cacheKeySize:{},"
                        + " available_kv_cache:{}, total_kv_cache:{}, cost:{}, syncIntervalMs:{}",
                ipPort,
                roleType.name(),
                cacheStatus.getBlockSize(),
                cacheStatus.getVersion(),
                cacheStatus.getCacheKeySize(),
                cacheStatus.getAvailableKvCache(),
                cacheStatus.getTotalKvCache(),
                (System.nanoTime() / 1000) - startTime,
                DynamicCacheIntervalService.getCurrentIntervalMs());
    }

    private WorkerCacheUpdateResult updateLocalKvCache() {
        try {
            WorkerCacheUpdateResult result =
                    cacheAwareService.updateEngineBlockCache(workerStatus);
            if (result == null) {
                logger.debug(
                        "Cache service returned no update result for {}#{}",
                        ipPort, generationId);
                engineHealthReporter.reportCacheStatusCheckerFail(
                        modelName, BalanceStatusEnum.CACHE_UPDATE_FAILED, roleType);
                return null;
            }
            if (!result.isSuccess()) {
                logger.debug(
                        "Cache update rejected for {}#{}, error:{}",
                        ipPort,
                        generationId,
                        result.getErrorMessage());
                engineHealthReporter.reportCacheStatusCheckerFail(
                        modelName,
                        BalanceStatusEnum.CACHE_UPDATE_FAILED,
                        roleType);
            }
            return result;
        } catch (Exception e) {
            logger.debug("Exception to update worker cache for {}#{}: {}",
                    ipPort, generationId, e.getMessage());
            engineHealthReporter.reportCacheStatusCheckerFail(
                    modelName, BalanceStatusEnum.CACHE_UPDATE_FAILED, roleType);
            return null;
        }
    }

    private boolean isCacheProducingRole() {
        return RoleType.PREFILL.equals(roleType)
                || RoleType.PDFUSION.equals(roleType);
    }

    private void log(String msg) {
        logger.debug("[gRPC-Cache][{}][{}][{}][{}][{}μs]: {}",
                id,
                site,
                ipPort,
                modelName,
                (System.nanoTime() / 1000) - startTime,
                msg);
    }

    private void log(String msg, Throwable e) {
        logger.debug("[gRPC-Cache][{}][{}][{}][{}][{}μs]: {}",
                id,
                site,
                ipPort,
                modelName,
                (System.nanoTime() / 1000) - startTime,
                msg,
                e);
    }

    private void handleException(Throwable ex) {
        log("gRPC cache status check failed:ipPort:" + ipPort + ", with exception: " + ex.getMessage());
        // Report specific error based on exception type
        if (ex.getMessage() != null && ex.getMessage().toLowerCase().contains(DEADLINE_EXCEEDED_MESSAGE.toLowerCase())) {
            engineHealthReporter.reportCacheStatusCheckerFail(
                    modelName, BalanceStatusEnum.CACHE_GRPC_TIMEOUT, roleType);
        } else {
            engineHealthReporter.reportCacheStatusCheckerFail(
                    modelName, BalanceStatusEnum.CACHE_SERVICE_UNAVAILABLE, roleType);
        }
    }

    private long getCurrentCacheVersion() {
        return debug ? -1L : Optional.ofNullable(workerStatus)
                .map(WorkerStatus::getCacheStatus)
                .map(CacheStatus::getVersion)
                .orElse(-1L);
    }
}
