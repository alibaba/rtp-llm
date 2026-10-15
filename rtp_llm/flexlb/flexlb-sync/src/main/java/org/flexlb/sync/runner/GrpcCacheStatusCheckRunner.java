package org.flexlb.sync.runner;

import org.flexlb.cache.domain.WorkerCacheUpdateResult;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.cache.service.DynamicCacheIntervalService;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.EndpointTrace;
import org.flexlb.dao.master.EndpointTrace.Stage;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.enums.BalanceStatusEnum;
import org.flexlb.service.grpc.EngineGrpcService;
import org.flexlb.service.grpc.EngineStatusConverter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.WorkerDirectory;
import org.flexlb.util.CommonUtils;
import org.flexlb.util.IdUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executor;

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
    private final String ip;
    private final int grpcPort;
    private final long startTime = System.nanoTime() / 1000;
    private final String id = IdUtils.fastUuid();
    private final boolean debug;
    private final long requestTimeoutMs;
    private final long syncCycleId;
    private final Long syncEngineStatusInterval;
    private final Executor callbackExecutor;

    public GrpcCacheStatusCheckRunner(String modelName, String ipPort, String site, RoleType roleType,
                                      WorkerStatus workerStatus,
                                      WorkerStatus.PollLease pollLease,
                                      WorkerDirectory workerDirectory,
                                      EngineHealthReporter engineHealthReporter,
                                      EngineGrpcService engineGrpcService,
                                      CacheAwareService cacheAwareService,
                                      long requestTimeoutMs,
                                      long syncCycleId,
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
        this.debug = fullSnapshotDebugMode;
        this.requestTimeoutMs = requestTimeoutMs;
        this.syncCycleId = syncCycleId;
        this.syncEngineStatusInterval = syncEngineStatusInterval;
        this.callbackExecutor = callbackExecutor;
    }

    @Override
    public void run() {
        boolean asyncInitiated = false;
        var pollTrace = pollLease.pollTrace();
        try {
            pollTrace.record(Stage.RUNNER_STARTED);
            logger.debug("GrpcCacheStatusCheckRunner run for {}", ipPort);
            long prefillCacheStatusCheckInterval =
                    DynamicCacheIntervalService.getCurrentIntervalMs();
            long roundInterval = prefillCacheStatusCheckInterval / syncEngineStatusInterval;
            roundInterval = Math.max(roundInterval, 1);

            // EngineSyncRunner is invoked on the status-poll cadence. Prefill
            // cache polling uses an integer number of those ticks so a changing
            // interval does not require a second scheduler or timer.
            if (roleType.requiresCacheKeys()
                        && syncCycleId % roundInterval != 0) {
                logger.debug("Skip prefill cache status check for {} because not in {}ms interval", ipPort, prefillCacheStatusCheckInterval);
                pollTrace.record(Stage.POLL_SKIPPED, "CACHE_INTERVAL");
                return; // finally will reset the flag
            }

            long startTime = System.nanoTime() / 1000;
            long currentCacheVersion = getCurrentCacheVersion();

            pollTrace.recordRpcStarted(requestTimeoutMs);
            CompletableFuture<EngineRpcService.CacheStatusPB> rpc;
            try {
                rpc = engineGrpcService.getCacheStatusAsync(
                        ip, grpcPort, workerStatus, currentCacheVersion,
                        requestTimeoutMs, roleType);
            } catch (RuntimeException | Error failure) {
                pollTrace.record(Stage.RPC_INITIATION_FAILED, failure);
                throw failure;
            }
            rpc.whenComplete((response, failure) -> pollTrace.record(Stage.RPC_COMPLETED, failure))
                    .thenApply(response -> {
                        pollTrace.record(Stage.CONVERSION_STARTED);
                        return EngineStatusConverter.convertToCacheStatus(response);
                    })
                    // Readiness precedes executor handoff; it is not an internal gRPC queue timestamp.
                    .whenComplete((response, failure) -> pollTrace.record(Stage.CALLBACK_READY, failure))
                    .handleAsync((response, failure) -> {
                        pollTrace.record(Stage.CALLBACK_STARTED);
                        try {
                            Throwable cause = EndpointTrace.unwrap(failure);
                            if (cause != null) {
                                handleException(cause);
                                CacheStatus errorStatus = CacheStatus.builder()
                                        .version(-1)
                                        .availableKvCache(0)
                                        .totalKvCache(0)
                                        .blockSize(0)
                                        .message("Cache Status gRPC call failed: " + cause.getMessage())
                                        .build();
                                handleCacheStatusResponse(errorStatus, startTime);
                            } else {
                                handleCacheStatusResponse(response, startTime);
                            }
                        } catch (Throwable callbackFailure) {
                            pollTrace.record(Stage.CALLBACK_FAILED, callbackFailure);
                            logger.error("Cache status callback failed for {}", ipPort, callbackFailure);
                        } finally {
                            pollTrace.record(Stage.CALLBACK_ENDED);
                        }
                        return null;
                    }, callbackExecutor)
                    .whenComplete((ignored, failure) -> {
                        pollLease.close();
                        if (failure != null) {
                            Throwable cause = EndpointTrace.unwrap(failure);
                            pollTrace.record(Stage.CALLBACK_FAILED, cause);
                            logger.error("Cache status callback failed for {}", ipPort, cause);
                        }
                    });
            asyncInitiated = true;
        } catch (Throwable failure) {
            pollTrace.record(Stage.RUNNER_FAILED, failure);
            throw failure;
        } finally {
            if (!asyncInitiated) {
                pollLease.close();
            }
        }
    }

    private void handleCacheStatusResponse(CacheStatus newCacheStatus, long startTime) {

        try {
            logger.debug("gRPC Cache Status - handled for {}, role:{}", ipPort, roleType.name());

            if (newCacheStatus.getMessage() != null) {
                pollLease.pollTrace().record(Stage.RESPONSE_DISCARDED, newCacheStatus.getMessage());
                logger.debug("gRPC Cache Status - {}, role:{}, message:{}", ipPort, roleType.name(), newCacheStatus.getMessage());
                return;
            }

            long successfulIntervalUs;
            pollLease.pollTrace().record(Stage.CACHE_LOCK_ATTEMPT);
            workerStatus.lock.lock();
            try {
                pollLease.pollTrace().record(Stage.CACHE_LOCK_ACQUIRED);
                if (!workerDirectory.isCurrentStatus(
                        roleType, ipPort, workerStatus)
                        || !workerStatus.isActiveGeneration()) {
                    pollLease.pollTrace().record(Stage.RESPONSE_DISCARDED, "GENERATION_CHANGED_OR_RETIRING");
                    logger.debug(
                            "Ignore stale cache callback for {}#{}, role:{}",
                            ipPort, generationId, roleType);
                    return;
                }
                if (validateCacheStatusResponse(workerStatus, newCacheStatus)) {
                    workerStatus.publishCacheStatus(newCacheStatus);
                    if (roleType.requiresCacheKeys()) {
                        // CacheAwareService is keyed by address, not generation.
                        // Keep the generation lock through this in-memory index
                        // update so retirement cannot publish a replacement or
                        // clear the address between validation and publication.
                        WorkerCacheUpdateResult updateResult = updateLocalKvCache();
                        if (updateResult != null && updateResult.isSuccess()) {
                            workerStatus.publishCacheIndexedVersion(
                                    newCacheStatus.getVersion());
                        }
                    }
                    logCacheStatusUpdate(newCacheStatus, startTime);
                } else {
                    pollLease.pollTrace().record(Stage.CACHE_VERSION_UNCHANGED);
                }
                successfulIntervalUs =
                        workerStatus.recordSuccessfulCachePoll();
            } finally {
                workerStatus.lock.unlock();
                pollLease.pollTrace().record(Stage.CACHE_LOCK_RELEASED);
            }

            engineHealthReporter.reportCacheStatusCheckRemoteInfo(
                    modelName, roleType.name(), startTime);
            engineHealthReporter.reportCacheStatusCheckerSuccess(
                    modelName, workerStatus, successfulIntervalUs);
        } catch (Throwable e) {
            pollLease.pollTrace().record(Stage.CACHE_RESPONSE_FAILED, e);
            log("engine cache status check via gRPC exception, msg: " + e.getMessage(), e);
            engineHealthReporter.reportCacheStatusCheckerFail(
                    modelName, BalanceStatusEnum.CACHE_SERVICE_UNAVAILABLE, roleType);
        }
    }

    private boolean validateCacheStatusResponse(WorkerStatus workerStatus, CacheStatus newCacheStatus) {
        if (debug) {
            return true;
        }
        WorkerStatus.CacheIndexSnapshot current =
                workerStatus.cacheIndexSnapshot();
        CacheStatus currentCacheStatus = current.cacheStatus();
        if (currentCacheStatus == null) {
            return true;
        }
        long currentVersion = currentCacheStatus.getVersion();
        long responseVersion = newCacheStatus.getVersion();
        boolean responseAlreadyIndexed = current.indexInitialized()
                && current.indexedVersion() == responseVersion;
        if (responseVersion < currentVersion
                || responseVersion == currentVersion && responseAlreadyIndexed) {
            logger.debug("gRPC Cache Status - {}, role:{}, version not updated, current: {}, response: {}",
                    ipPort, roleType.name(), currentVersion, responseVersion);
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
            pollLease.pollTrace().record(Stage.CACHE_INDEX_UPDATE_BEGIN);
            WorkerCacheUpdateResult result =
                    cacheAwareService.updateEngineBlockCache(workerStatus, pollLease.pollTrace());
            if (result != null && result.isSuccess()) {
                pollLease.pollTrace().record(Stage.CACHE_INDEX_UPDATE_END);
            } else {
                pollLease.pollTrace().record(Stage.CACHE_INDEX_UPDATE_FAILED, result == null ? "NULL_RESULT"
                                : Objects.toString(result.getErrorMessage(), ""));
            }
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
            pollLease.pollTrace().record(Stage.CACHE_INDEX_UPDATE_FAILED, e);
            logger.debug("Exception to update worker cache for {}#{}: {}",
                    ipPort, generationId, e.getMessage());
            engineHealthReporter.reportCacheStatusCheckerFail(
                    modelName, BalanceStatusEnum.CACHE_UPDATE_FAILED, roleType);
            return null;
        }
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
        if (debug) {
            return -1L;
        }
        WorkerStatus.CacheIndexSnapshot snapshot =
                workerStatus.cacheIndexSnapshot();
        return snapshot.indexInitialized() ? snapshot.indexedVersion() : -1L;
    }
}
