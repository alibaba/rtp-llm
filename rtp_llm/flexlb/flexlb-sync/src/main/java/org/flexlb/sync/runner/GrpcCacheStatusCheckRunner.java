package org.flexlb.sync.runner;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.cache.domain.WorkerCacheUpdateResult;
import org.flexlb.cache.match.CacheAwareService;
import org.flexlb.cache.match.localsync.DynamicCacheIntervalService;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.EndpointTrace.Stage;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.enums.BalanceStatusEnum;
import org.flexlb.service.grpc.EngineStatusConverter;
import org.flexlb.service.grpc.WorkerStatusRpcClient;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.util.IdUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.Executor;
import java.util.concurrent.TimeUnit;

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
    private final Executor callbackExecutor;

    public GrpcCacheStatusCheckRunner(
            String modelName,
            WorkerStatus workerStatus,
            WorkerStatus.PollLease pollLease,
            EndpointRegistry endpointRegistry,
            EngineHealthReporter engineHealthReporter,
            WorkerStatusRpcClient workerStatusRpcClient,
            CacheAwareService cacheAwareService,
            DynamicCacheIntervalService cacheIntervalService,
            long requestTimeoutMs,
            boolean fullSnapshotDebugMode,
            Executor callbackExecutor) {

        this.ipPort = workerStatus.getLogicalIpPort();
        this.roleType = workerStatus.getRole();
        this.modelName = modelName;
        this.workerStatus = workerStatus;
        this.endpointRegistry =
                java.util.Objects.requireNonNull(endpointRegistry, "endpointRegistry");
        this.pollLease = java.util.Objects.requireNonNull(pollLease, "pollLease");
        workerStatus.requireCachePollLease(pollLease);
        this.engineHealthReporter = engineHealthReporter;
        this.workerStatusRpcClient = workerStatusRpcClient;
        this.cacheAwareService = cacheAwareService;
        this.cacheIntervalService =
                java.util.Objects.requireNonNull(cacheIntervalService, "cacheIntervalService");
        this.debug = fullSnapshotDebugMode;
        this.requestTimeoutMs = requestTimeoutMs;
        this.callbackExecutor = callbackExecutor;
    }

    @Override
    public void run() {
        pollLease.trace().record(Stage.RUNNER_STARTED);
        boolean asyncInitiated = false;
        try {
            logger.debug("GrpcCacheStatusCheckRunner run for {}", ipPort);
            long startTime = TimeUnit.NANOSECONDS.toMicros(System.nanoTime());
            long currentCacheVersion = getCurrentCacheVersion();

            pollLease.trace().recordRpcStarted(requestTimeoutMs);
            PollCompletion.registerResultCallback(
                    pollLease,
                    callbackExecutor,
                    "Cache status",
                    ipPort,
                    workerStatusRpcClient.getCacheStatusAsync(
                            workerStatus.getIp(),
                            workerStatus.getWorkerStatusPort(),
                            currentCacheVersion,
                            requestTimeoutMs,
                            roleType),
                    (cacheStatus, failure) -> {
                        if (failure != null) {
                            handleException(failure);
                        } else {
                            handleCacheStatusResponse(cacheStatus, startTime);
                        }
                    });
            asyncInitiated = true;
        } catch (RuntimeException | Error failure) {
            pollLease.trace().record(Stage.RPC_INITIATION_FAILED, failure);
            throw failure;
        } finally {
            if (!asyncInitiated) {
                pollLease.close();
            }
        }
    }

    private void handleCacheStatusResponse(
            EngineRpcService.CacheStatusPB response, long startTime) {

        try {
            logger.debug("gRPC Cache Status - handled for {}, role:{}", ipPort, roleType.name());

            long successfulIntervalUs;
            pollLease.trace().record(Stage.CACHE_MUTATION_LOCK_ATTEMPT);
            workerStatus.cacheMutationLock.lock();
            pollLease.trace().record(Stage.CACHE_MUTATION_LOCK_ACQUIRED);
            try {
                if (!endpointRegistry.isCurrentStatus(roleType, ipPort, workerStatus)
                        || !workerStatus.isActiveGeneration()) {
                    logger.debug(
                            "Ignore stale cache callback for {}#{}, role:{}",
                            ipPort,
                            workerStatus.getGenerationId(),
                            roleType);
                    return;
                }
                WorkerStatus.CacheIndexSnapshot snapshot = workerStatus.cacheIndexSnapshot();
                boolean reuseKeys =
                        !debug
                                && snapshot.indexInitialized()
                                && snapshot.cacheStatus() != null
                                && snapshot.indexedVersion() == response.getVersion()
                                && snapshot.cacheStatus().getVersion() == response.getVersion();
                pollLease.trace().record(Stage.CONVERSION_STARTED);
                CacheStatus newCacheStatus =
                        EngineStatusConverter.convertToCacheStatus(
                                response,
                                reuseKeys ? snapshot.cacheStatus().getCachedKeys() : null);
                pollLease.trace().record(Stage.CONVERSION_COMPLETED);
                if (reuseKeys) {
                    pollLease.trace().record(Stage.CACHE_VERSION_UNCHANGED);
                }
                if (validateCacheStatusResponse(workerStatus, newCacheStatus)) {
                    workerStatus.publishCacheStatus(newCacheStatus);
                    if (!reuseKeys
                            && workerStatus.isActiveGeneration()
                            && (!roleType.requiresCacheKeys() || updateLocalKvCache())) {
                        if (workerStatus.isActiveGeneration()) {
                            workerStatus.publishCacheIndexedVersion(newCacheStatus.getVersion());
                        }
                    }
                    logCacheStatusUpdate(newCacheStatus, startTime);
                }
                successfulIntervalUs = workerStatus.recordSuccessfulCachePoll();
            } finally {
                workerStatus.cacheMutationLock.unlock();
                pollLease.trace().record(Stage.CACHE_MUTATION_LOCK_RELEASED);
            }

            engineHealthReporter.reportCacheStatusCheckRemoteInfo(workerStatus, startTime);
            engineHealthReporter.reportCacheStatusCheckerSuccess(
                    workerStatus, successfulIntervalUs);
        } catch (Throwable e) {
            pollLease.trace().record(Stage.CACHE_RESPONSE_FAILED, e);
            log("engine cache status check via gRPC exception, msg: " + e.getMessage(), e);
            org.flexlb.metric.SafeFlexMonitor.run(
                    "cacheStatusCheckFailure",
                    () ->
                            engineHealthReporter.reportCacheStatusCheckerFail(
                                    workerStatus, BalanceStatusEnum.CACHE_SERVICE_UNAVAILABLE));
        }
    }

    private boolean validateCacheStatusResponse(
            WorkerStatus workerStatus, CacheStatus newCacheStatus) {
        if (debug) {
            return true;
        }
        WorkerStatus.CacheIndexSnapshot current = workerStatus.cacheIndexSnapshot();
        CacheStatus currentCacheStatus = current.cacheStatus();
        if (currentCacheStatus == null) {
            return true;
        }
        long currentVersion = currentCacheStatus.getVersion();
        long responseVersion = newCacheStatus.getVersion();
        if (responseVersion < currentVersion) {
            logger.debug(
                    "gRPC Cache Status - {}, role:{}, version not updated, current: {}, response: {}",
                    ipPort,
                    roleType.name(),
                    currentVersion,
                    responseVersion);
            return false;
        }
        return true;
    }

    private void logCacheStatusUpdate(CacheStatus cacheStatus, long startTime) {

        logger.debug(
                "gRPC Cache Status - {}, role:{}, block_size:{}, version:{}, cacheKeySize:{},"
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
            pollLease.trace().record(Stage.CACHE_INDEX_UPDATE_BEGIN);
            WorkerCacheUpdateResult result = cacheAwareService.updateFromWorkerStatus(workerStatus);
            if (result != null && result.isSuccess()) {
                pollLease.trace().record(Stage.CACHE_INDEX_UPDATE_END);
                return true;
            }
            pollLease.trace().record(Stage.CACHE_INDEX_UPDATE_FAILED, "UPDATE_FAILED");
            logger.debug(
                    "Cache update failed for {}#{}, error:{}",
                    ipPort,
                    workerStatus.getGenerationId(),
                    result == null ? "no update result" : result.getErrorMessage());
            org.flexlb.metric.SafeFlexMonitor.run(
                    "cacheStatusCheckFailure",
                    () ->
                            engineHealthReporter.reportCacheStatusCheckerFail(
                                    workerStatus, BalanceStatusEnum.CACHE_UPDATE_FAILED));
        } catch (Exception e) {
            pollLease.trace().record(Stage.CACHE_INDEX_UPDATE_FAILED, e);
            logger.debug(
                    "Exception to update worker cache for {}#{}: {}",
                    ipPort,
                    workerStatus.getGenerationId(),
                    e.getMessage());
            org.flexlb.metric.SafeFlexMonitor.run(
                    "cacheStatusCheckFailure",
                    () ->
                            engineHealthReporter.reportCacheStatusCheckerFail(
                                    workerStatus, BalanceStatusEnum.CACHE_UPDATE_FAILED));
        }
        return false;
    }

    private void log(String msg, Throwable failure) {
        logger.debug(
                "[gRPC-Cache][{}][{}][{}][{}][{}μs]: {}",
                id,
                workerStatus.getSite(),
                ipPort,
                modelName,
                (TimeUnit.NANOSECONDS.toMicros(System.nanoTime())) - startTime,
                msg,
                failure);
    }

    private void handleException(Throwable ex) {
        if (logger.isDebugEnabled()) {
            logger.debug(
                    "Cache poll failure: endpoint={} generation={} trace={}",
                    ipPort,
                    workerStatus.getGenerationId(),
                    workerStatus.trace().snapshot().compact());
        }
        log(
                "gRPC cache status check failed:ipPort:"
                        + ipPort
                        + ", with exception: "
                        + ex.getMessage(),
                ex);
        // Report specific error based on exception type
        if (io.grpc.Status.fromThrowable(ex).getCode() == io.grpc.Status.Code.DEADLINE_EXCEEDED) {
            org.flexlb.metric.SafeFlexMonitor.run(
                    "cacheStatusCheckFailure",
                    () ->
                            engineHealthReporter.reportCacheStatusCheckerFail(
                                    workerStatus, BalanceStatusEnum.CACHE_GRPC_TIMEOUT));
        } else {
            org.flexlb.metric.SafeFlexMonitor.run(
                    "cacheStatusCheckFailure",
                    () ->
                            engineHealthReporter.reportCacheStatusCheckerFail(
                                    workerStatus, BalanceStatusEnum.CACHE_SERVICE_UNAVAILABLE));
        }
    }

    private long getCurrentCacheVersion() {
        if (debug) {
            return -1L;
        }
        WorkerStatus.CacheIndexSnapshot snapshot = workerStatus.cacheIndexSnapshot();
        return snapshot.indexInitialized() ? snapshot.indexedVersion() : -1L;
    }
}
