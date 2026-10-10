package org.flexlb.sync.runner;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.cache.domain.WorkerCacheUpdateResult;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.cache.service.DynamicCacheIntervalService;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.BalanceStatusEnum;
import org.flexlb.service.grpc.WorkerStatusRpcClient;
import org.flexlb.service.grpc.EngineStatusConverter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.util.IdUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.Executor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.LongAdder;

import static org.flexlb.constant.CommonConstants.DEADLINE_EXCEEDED_MESSAGE;

public class GrpcCacheStatusCheckRunner implements Runnable {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");

    private final String ipPort;
    private final String modelName;
    private final RoleType roleType;
    private final WorkerStatus workerStatus;
    private final EndpointRegistry endpointRegistry;
    private final WorkerStatus.PollLease pollLease;
    private final EngineHealthReporter engineHealthReporter;
    private final WorkerStatusRpcClient workerStatusRpcClient;
    private final CacheAwareService cacheAwareService;
    private final DynamicCacheIntervalService cacheIntervalService;
    private final long startTime = TimeUnit.NANOSECONDS.toMicros(System.nanoTime());
    private final String id = IdUtils.fastUuid();
    private final boolean debug;
    private final long requestTimeoutMs;
    private final LongAdder syncCount;
    private final Long syncEngineStatusInterval;
    private final Executor callbackExecutor;

    public GrpcCacheStatusCheckRunner(String modelName, WorkerStatus workerStatus,
                                      WorkerStatus.PollLease pollLease,
                                      EndpointRegistry endpointRegistry,
                                      EngineHealthReporter engineHealthReporter,
                                      WorkerStatusRpcClient workerStatusRpcClient,
                                      CacheAwareService cacheAwareService,
                                      DynamicCacheIntervalService cacheIntervalService,
                                      long requestTimeoutMs,
                                      LongAdder syncCount,
                                      Long syncEngineStatusInterval,
                                      boolean fullSnapshotDebugMode,
                                      Executor callbackExecutor) {

        this.ipPort = workerStatus.getIpPort();
        this.roleType = workerStatus.getRole();
        this.modelName = modelName;
        this.workerStatus = workerStatus;
        this.endpointRegistry = java.util.Objects.requireNonNull(
                endpointRegistry, "endpointRegistry");
        this.pollLease = java.util.Objects.requireNonNull(
                pollLease, "pollLease");
        workerStatus.requireCachePollLease(pollLease);
        this.engineHealthReporter = engineHealthReporter;
        this.workerStatusRpcClient = workerStatusRpcClient;
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
                    cacheIntervalService.getCurrentIntervalMs();
            long roundInterval = prefillCacheStatusCheckInterval / syncEngineStatusInterval;
            roundInterval = Math.max(roundInterval, 1);

            // EngineSyncRunner is invoked on the status-poll cadence. Prefill
            // cache polling uses an integer number of those ticks so a changing
            // interval does not require a second scheduler or timer.
            if (roleType.requiresCacheKeys()
                        && syncCount.longValue() % roundInterval != 0) {
                logger.debug("Skip prefill cache status check for {} because not in {}ms interval", ipPort, prefillCacheStatusCheckInterval);
                return; // The synchronous scope still owns the poll lease.
            }

            long startTime = TimeUnit.NANOSECONDS.toMicros(System.nanoTime());
            long currentCacheVersion = getCurrentCacheVersion();

            PollCompletion.registerResultCallback(pollLease, callbackExecutor, "Cache status", ipPort,
                    workerStatusRpcClient.getCacheStatusAsync(workerStatus.getIp(), workerStatus.getGrpcPort(), currentCacheVersion,
                            requestTimeoutMs, roleType)
                    .thenApply(cacheStatusPB -> {
                        logger.debug("gRPC Cache Status Response - handled for {}, role:{}, cache_key_size:{}, cache_version:{}, "
                                        + "available_kv_cache:{}, total_kv_cache:{}, block_size:{}",
                                ipPort, roleType.name(), cacheStatusPB.getCacheKeysMap().size(), cacheStatusPB.getVersion(),
                                cacheStatusPB.getAvailableKvCache(), cacheStatusPB.getTotalKvCache(), cacheStatusPB.getBlockSize());
                        return EngineStatusConverter.convertToCacheStatus(cacheStatusPB);
                    }), (cacheStatus, failure) -> {
                        if (failure != null) {
                            handleException(failure);
                        } else {
                            handleCacheStatusResponse(cacheStatus, startTime);
                        }
                    });
            asyncInitiated = true;
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
                logger.debug("gRPC Cache Status - {}, role:{}, message:{}", ipPort, roleType.name(), newCacheStatus.getMessage());
                return;
            }

            long successfulIntervalUs;
            workerStatus.lock.lock();
            try {
                if (!endpointRegistry.isCurrentStatus(
                        roleType, ipPort, workerStatus)
                        || !workerStatus.isActiveGeneration()) {
                    logger.debug(
                            "Ignore stale cache callback for {}#{}, role:{}",
                            ipPort, workerStatus.getGenerationId(), roleType);
                    return;
                }
                if (validateCacheStatusResponse(workerStatus, newCacheStatus)) {
                    workerStatus.publishCacheStatus(newCacheStatus);
                    if (roleType.requiresCacheKeys()) {
                        // CacheAwareService is keyed by address, not generation.
                        // Keep the generation lock through this in-memory index
                        // update so retirement cannot publish a replacement or
                        // clear the address between validation and publication.
                        if (updateLocalKvCache()) {
                            workerStatus.publishCacheIndexedVersion(
                                    newCacheStatus.getVersion());
                        }
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
                (TimeUnit.NANOSECONDS.toMicros(System.nanoTime())) - startTime,
                cacheIntervalService.getCurrentIntervalMs());
    }

    private boolean updateLocalKvCache() {
        try {
            WorkerCacheUpdateResult result = cacheAwareService.updateEngineBlockCache(workerStatus);
            if (result != null && result.isSuccess()) {
                return true;
            }
            logger.debug("Cache update failed for {}#{}, error:{}",
                    ipPort, workerStatus.getGenerationId(),
                    result == null ? "no update result" : result.getErrorMessage());
            engineHealthReporter.reportCacheStatusCheckerFail(
                    modelName, BalanceStatusEnum.CACHE_UPDATE_FAILED, roleType);
        } catch (Exception e) {
            logger.debug("Exception to update worker cache for {}#{}: {}",
                    ipPort, workerStatus.getGenerationId(), e.getMessage());
            engineHealthReporter.reportCacheStatusCheckerFail(
                    modelName, BalanceStatusEnum.CACHE_UPDATE_FAILED, roleType);
        }
        return false;
    }

    private void log(String msg, Throwable failure) {
        logger.debug("[gRPC-Cache][{}][{}][{}][{}][{}μs]: {}",
                id,
                workerStatus.getSite(),
                ipPort,
                modelName,
                (TimeUnit.NANOSECONDS.toMicros(System.nanoTime())) - startTime,
                msg, failure);
    }

    private void handleException(Throwable ex) {
        log("gRPC cache status check failed:ipPort:" + ipPort + ", with exception: " + ex.getMessage(), ex);
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
