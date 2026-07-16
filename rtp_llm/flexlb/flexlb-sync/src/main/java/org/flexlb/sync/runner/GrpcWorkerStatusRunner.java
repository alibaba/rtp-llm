package org.flexlb.sync.runner;

import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.pv.CacheHitComparisonPvLog;
import org.flexlb.dao.route.RoleType;
import org.flexlb.domain.worker.WorkerStatusResponse;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.enums.BalanceStatusEnum;
import org.flexlb.service.grpc.EngineGrpcService;
import org.flexlb.service.grpc.EngineStatusConverter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.util.IdUtils;
import org.flexlb.util.JsonUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicLong;

import static org.flexlb.constant.CommonConstants.DEADLINE_EXCEEDED_MESSAGE;

public class GrpcWorkerStatusRunner implements Runnable {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");
    private static final Logger pvLogger = LoggerFactory.getLogger("pvLogger");

    private final String ipPort;
    private final String modelName;
    private final String site;
    private final RoleType roleType;
    private final String group;
    private final WorkerStatus workerStatus;
    private final EngineHealthReporter engineHealthReporter;
    private final EngineGrpcService engineGrpcService;
    private final String ip;
    private final int workerStatusPort;
    private final long createTimeUs = System.nanoTime() / 1000;
    private final String id = IdUtils.fastUuid();
    private final long syncRequestTimeoutMs;

    public GrpcWorkerStatusRunner(String modelName, WorkerHost host, RoleType roleType,
                                  WorkerStatus workerStatus,
                                  EngineHealthReporter engineHealthReporter,
                                  EngineGrpcService engineGrpcService,
                                  long syncRequestTimeoutMs) {
        this.ipPort = host.getIpPort();
        this.ip = host.getIp();
        this.workerStatusPort = host.getWorkerStatusPort();
        this.modelName = modelName;
        this.workerStatus = workerStatus;
        this.site = host.getSite();
        this.roleType = roleType;
        this.group = host.getGroup();
        this.engineHealthReporter = engineHealthReporter;
        this.engineGrpcService = engineGrpcService;
        this.syncRequestTimeoutMs = syncRequestTimeoutMs;
    }

    @Override
    public void run() {
        try {
            logger.debug("GrpcWorkerStatusRunner run for {}", ipPort);
            long startTime = System.nanoTime() / 1000;

            long latestFinishedTaskVersion = workerStatus.getLatestFinishedTaskVersion().get();

            WorkerStatusResponse response = launchGrpcStatusCheck(ip, workerStatusPort, latestFinishedTaskVersion);
            handleStatusResponse(response, startTime);
        } finally {
            workerStatus.getStatusCheckInProgress().set(false);
        }
    }

    private WorkerStatusResponse launchGrpcStatusCheck(String ip, int grpcPort, long latestFinishedTaskVersion) {
        try {
            EngineRpcService.WorkerStatusPB workerStatusPB = engineGrpcService.getWorkerStatus(ip, grpcPort, latestFinishedTaskVersion, syncRequestTimeoutMs, roleType);
            return EngineStatusConverter.convertToWorkerStatusResponse(workerStatusPB);
        } catch (Throwable throwable) {
            handleException(throwable);
            WorkerStatusResponse errorResponse = new WorkerStatusResponse();
            errorResponse.setMessage("Worker status gRPC call failed: " + throwable.getMessage());
            return errorResponse;
        }
    }

    private void handleStatusResponse(WorkerStatusResponse newWorkerStatus, long startTime) {
        try {
            if (newWorkerStatus == null) {
                logger.info("query engine worker status via gRPC, response body is null");
                engineHealthReporter.reportStatusCheckerFail(modelName, BalanceStatusEnum.RESPONSE_NULL, ip, roleType);
                return;
            }

            if (newWorkerStatus.getMessage() != null) {
                workerStatus.setAlive(false);
                logger.error("query engine worker status via gRPC, msg={}", newWorkerStatus.getMessage());
                return;
            }

            // Only report success worker status check info
            engineHealthReporter.reportStatusCheckRemoteInfo(modelName, ipPort, newWorkerStatus.getRole(), startTime);

            Long responseVersion = newWorkerStatus.getStatusVersion();
            if (responseVersion == 0L) {
                logger.info("workerStatuses.get(ip) is null for gRPC call");
                return;
            }

            workerStatus.setSite(site);
            workerStatus.setGroup(group);
            workerStatus.setRole(newWorkerStatus.getRole());
            workerStatus.setBlockHashLookaheadTokens(newWorkerStatus.getBlockHashLookaheadTokens());
            updateCacheStatus(newWorkerStatus.getCacheStatus());

            long currentVersion = workerStatus.getStatusVersion().get();
            if (currentVersion >= responseVersion) {
                logger.debug("query engine worker status via gRPC, version is not updated, currentVersion: {}, responseVersion: {}",
                        currentVersion, responseVersion);
                // Update basic worker status even when version is not updated
                workerStatus.setAlive(newWorkerStatus.isAlive());
                workerStatus.setDpSize(newWorkerStatus.getDpSize());
                workerStatus.setTpSize(newWorkerStatus.getTpSize());

                // Update status timestamp and record actual sync interval
                long nowUs = System.nanoTime() / 1000;
                long prevUpdateTime = workerStatus.getStatusLastUpdateTime().get();
                if (prevUpdateTime > 0) {
                    workerStatus.getStatusUpdateIntervalUs().set(nowUs - prevUpdateTime);
                }
                workerStatus.getStatusLastUpdateTime().set(nowUs);

                // Update task state
                Map<String, TaskInfo> waitingTaskInfo = newWorkerStatus.getWaitingTaskInfo();
                Map<String, TaskInfo> runningTaskInfo = newWorkerStatus.getRunningTaskInfo();
                Map<String, TaskInfo> finishedTaskInfo = newWorkerStatus.getFinishedTaskInfo();
                logCacheHitComparisons(workerStatus.updateTaskStates(
                        waitingTaskInfo, runningTaskInfo, finishedTaskInfo));

                // Report success even when version is not updated
                engineHealthReporter.reportStatusCheckerSuccess(modelName, workerStatus,
                    Optional.ofNullable(runningTaskInfo).map(Map::size).orElse(0),
                    Optional.ofNullable(finishedTaskInfo).map(Map::size).orElse(0));

                logWorkerStatusUpdate(startTime, workerStatus);
                return;
            }

            // Update worker status from gRPC response
            workerStatus.setAvailableConcurrency(newWorkerStatus.getAvailableConcurrency());
            workerStatus.setStepLatencyMs(newWorkerStatus.getStepLatencyMs());
            workerStatus.setIterateCount(newWorkerStatus.getIterateCount());
            workerStatus.setDpSize(newWorkerStatus.getDpSize());
            workerStatus.setTpSize(newWorkerStatus.getTpSize());
            workerStatus.setAlive(newWorkerStatus.isAlive());
            workerStatus.getStatusVersion().set(responseVersion);
            workerStatus.getLatestFinishedTaskVersion().set(newWorkerStatus.getLatestFinishedVersion() != null ? newWorkerStatus.getLatestFinishedVersion() : -1L);

            Map<String, TaskInfo> waitingTaskInfo = newWorkerStatus.getWaitingTaskInfo();
            Map<String, TaskInfo> runningTaskInfo = newWorkerStatus.getRunningTaskInfo();
            Map<String, TaskInfo> finishedTaskInfo = newWorkerStatus.getFinishedTaskInfo();
            workerStatus.setWaitingTaskList(waitingTaskInfo);
            workerStatus.setRunningTaskList(runningTaskInfo);

            // Update local task state (including checking lost, updating running, and cleaning completed)
            logCacheHitComparisons(workerStatus.updateTaskStates(
                    waitingTaskInfo, runningTaskInfo, finishedTaskInfo));

            // Correct running queue total wait time
            workerStatus.updateRunningQueueTime();

            engineHealthReporter.reportStatusCheckerSuccess(modelName, workerStatus,
                    Optional.ofNullable(runningTaskInfo).map(Map::size).orElse(0),
                    Optional.ofNullable(finishedTaskInfo).map(Map::size).orElse(0));

            // Update status timestamp and record actual sync interval
            long nowUs = System.nanoTime() / 1000;
            long prevUpdateTime = workerStatus.getStatusLastUpdateTime().get();
            if (prevUpdateTime > 0) {
                workerStatus.getStatusUpdateIntervalUs().set(nowUs - prevUpdateTime);
            }
            workerStatus.getStatusLastUpdateTime().set(nowUs);
            logWorkerStatusUpdate(startTime, workerStatus);

        } catch (Throwable e) {
            log("engine worker status check via gRPC exception, msg: " + e.getMessage());
            engineHealthReporter.reportStatusCheckerFail(modelName, BalanceStatusEnum.UNKNOWN_ERROR, ip, roleType);
        }
    }

    private void logWorkerStatusUpdate(long startTime, WorkerStatus workerStatus) {
        logger.debug("gRPC Worker Status - {}, role:{}, running_queue_tokens:{}, cost_us:{}",
                ipPort,
                workerStatus.getRole(),
                workerStatus.getRunningQueueTime(),
                System.nanoTime() / 1000 - startTime);
    }

    private void logCacheHitComparisons(List<CacheHitComparisonPvLog> comparisons) {
        for (CacheHitComparisonPvLog comparison : comparisons) {
            String json = JsonUtils.toStringOrEmpty(comparison);
            if (!json.isEmpty()) {
                pvLogger.info(json);
            }
        }
    }

    private void updateCacheStatus(CacheStatus cacheStatus) {
        if (cacheStatus == null) {
            return;
        }
        workerStatus.setCacheStatus(cacheStatus);
        long availableKvCache = cacheStatus.getAvailableKvCache();
        long usedKvCache = Math.max(0L, cacheStatus.getTotalKvCache() - availableKvCache);
        workerStatus.updateKvCacheTokens(usedKvCache, availableKvCache);
    }

    private void handleException(Throwable ex) {
        log("gRPC worker status check failed, msg=" + ex.getMessage());
        // Report specific error based on exception type
        if (ex.getMessage() != null && ex.getMessage().toLowerCase().contains(DEADLINE_EXCEEDED_MESSAGE.toLowerCase())) {
            logger.info("gRPC worker status check timeout, msg={}, ipPort: {}, rt_us: {}", ex.getMessage(), ipPort, System.nanoTime() / 1000 - createTimeUs);
            engineHealthReporter.reportStatusCheckerFail(modelName, BalanceStatusEnum.WORKER_STATUS_GRPC_TIMEOUT, ip, roleType);
        } else {
            engineHealthReporter.reportStatusCheckerFail(modelName, BalanceStatusEnum.WORKER_SERVICE_UNAVAILABLE, ip, roleType);
        }
    }

    private void log(String msg) {
        logger.info("[gRPC][{}][{}][{}][{}][{}μs]: {}",
                id,
                site,
                ipPort,
                modelName,
                System.nanoTime() / 1000 - createTimeUs,
                msg);
    }
}
