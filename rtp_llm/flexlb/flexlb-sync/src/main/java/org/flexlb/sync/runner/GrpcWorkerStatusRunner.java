package org.flexlb.sync.runner;

import org.flexlb.cache.domain.CacheHitComparisonResult;
import org.flexlb.cache.match.CacheAwareService;
import org.flexlb.dao.master.CacheHitFeedback;
import org.flexlb.dao.master.CacheStatus;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.TaskStateUpdateResult;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.domain.worker.WorkerStatusResponse;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.enums.BalanceStatusEnum;
import org.flexlb.enums.KvCacheGroupMode;
import org.flexlb.service.grpc.EngineGrpcService;
import org.flexlb.service.grpc.EngineStatusConverter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.util.IdUtils;
import org.flexlb.util.JsonUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;

import static org.flexlb.constant.CommonConstants.DEADLINE_EXCEEDED_MESSAGE;

public class GrpcWorkerStatusRunner implements Runnable {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");
    private static final Logger pvLogger = LoggerFactory.getLogger("pvLogger");

    private final String logicalIpPort;
    private final String ipIndex;
    private final String modelName;
    private final String site;
    private final RoleType roleType;
    private final String group;
    private final WorkerStatus workerStatus;
    private final EngineHealthReporter engineHealthReporter;
    private final EngineGrpcService engineGrpcService;
    private final String ip;
    /** Per-engine control port shared by GetWorkerStatus and GetCacheStatus. */
    private final int workerStatusPort;
    private final int engineIndex;
    private final long createTimeUs = System.nanoTime() / 1000;
    private final String id = IdUtils.fastUuid();
    private final long syncRequestTimeoutMs;
    private final CacheAwareService cacheAwareService;

    public GrpcWorkerStatusRunner(String modelName, WorkerHost host, RoleType roleType,
                                  WorkerStatus workerStatus,
                                  EngineHealthReporter engineHealthReporter,
                                  EngineGrpcService engineGrpcService,
                                  long syncRequestTimeoutMs,
                                  CacheAwareService cacheAwareService) {
        this.logicalIpPort = host.getLogicalIpPort();
        this.ipIndex = host.getIpIndex();
        this.ip = host.getIp();
        this.workerStatusPort = host.getWorkerStatusPort();
        this.engineIndex = host.getEngineIndex();
        this.modelName = modelName;
        this.workerStatus = workerStatus;
        this.site = host.getSite();
        this.roleType = roleType;
        this.group = host.getGroup();
        this.engineHealthReporter = engineHealthReporter;
        this.engineGrpcService = engineGrpcService;
        this.syncRequestTimeoutMs = syncRequestTimeoutMs;
        this.cacheAwareService = cacheAwareService;
    }

    @Override
    public void run() {
        try {
            logger.debug("GrpcWorkerStatusRunner run for {}", logicalIpPort);
            long startTime = System.nanoTime() / 1000;

            long latestFinishedTaskVersion = workerStatus.getLatestFinishedTaskVersion().get();

            WorkerStatusResponse response = launchGrpcStatusCheck(ip, workerStatusPort, latestFinishedTaskVersion, startTime);
            handleStatusResponse(response, startTime);
        } finally {
            workerStatus.getStatusCheckInProgress().set(false);
        }
    }

    private WorkerStatusResponse launchGrpcStatusCheck(String ip,
                                                       int grpcPort,
                                                       long latestFinishedTaskVersion,
                                                       long startTime) {
        try {
            EngineRpcService.WorkerStatusPB workerStatusPB = engineGrpcService.getWorkerStatus(ip, grpcPort, latestFinishedTaskVersion, syncRequestTimeoutMs, roleType);
            return EngineStatusConverter.convertToWorkerStatusResponse(workerStatusPB);
        } catch (Throwable throwable) {
            handleException(throwable, startTime);
            WorkerStatusResponse errorResponse = new WorkerStatusResponse();
            errorResponse.setMessage("Worker status gRPC call failed: " + throwable.getMessage());
            return errorResponse;
        }
    }

    private void handleStatusResponse(WorkerStatusResponse newWorkerStatus, long startTime) {
        try {
            if (newWorkerStatus == null) {
                logger.info("query engine worker status via gRPC, response body is null");
                reportStatusCheckFailure(BalanceStatusEnum.RESPONSE_NULL, startTime);
                return;
            }

            if (newWorkerStatus.getMessage() != null) {
                workerStatus.setAlive(false);
                logger.error("query engine worker status via gRPC, msg={}", newWorkerStatus.getMessage());
                return;
            }

            // Only report success worker status check info
            engineHealthReporter.reportStatusCheckRemoteInfo(
                    modelName, ipIndex, newWorkerStatus.getRole(), startTime);

            Long responseVersion = newWorkerStatus.getStatusVersion();
            if (responseVersion == 0L) {
                logger.info("workerStatuses.get(ip) is null for gRPC call");
                return;
            }

            workerStatus.setSite(site);
            workerStatus.setGroup(group);
            workerStatus.setRole(newWorkerStatus.getRole());
            workerStatus.setBlockHashLookaheadTokens(newWorkerStatus.getBlockHashLookaheadTokens());
            workerStatus.setCacheMatchRollbackBlocks(newWorkerStatus.getCacheMatchRollbackBlocks());
            updateKvCacheGroupMode(newWorkerStatus.getKvCacheGroupMode());
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
                handleTaskStateUpdateResult(workerStatus.updateTaskStates(waitingTaskInfo, runningTaskInfo, finishedTaskInfo));
                reportFinishedPrefillTasks(finishedTaskInfo);

                // Report success even when version is not updated
                engineHealthReporter.reportStatusCheckerSuccess(modelName, workerStatus,
                        Optional.ofNullable(waitingTaskInfo).map(Map::size).orElse(0),
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
            handleTaskStateUpdateResult(workerStatus.updateTaskStates(waitingTaskInfo, runningTaskInfo, finishedTaskInfo));
            reportFinishedPrefillTasks(finishedTaskInfo);

            // Correct running queue total wait time
            workerStatus.updateRunningQueueTime();

            engineHealthReporter.reportStatusCheckerSuccess(modelName, workerStatus,
                    Optional.ofNullable(waitingTaskInfo).map(Map::size).orElse(0),
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
            engineHealthReporter.reportStatusCheckerFail(
                    modelName, BalanceStatusEnum.UNKNOWN_ERROR, ipIndex, roleType);
        }
    }

    private void logWorkerStatusUpdate(long startTime, WorkerStatus workerStatus) {
        logger.debug("gRPC Worker Status - {}, role:{}, running_queue_tokens:{}, cost_us:{}",
                logicalIpPort,
                workerStatus.getRole(),
                workerStatus.getRunningQueueTime(),
                System.nanoTime() / 1000 - startTime);
    }

    private void handleTaskStateUpdateResult(TaskStateUpdateResult updateResult) {
        for (long latencyMs : updateResult.decisionToWaitingObservedLatenciesMs()) {
            engineHealthReporter.reportFlexlbObservedMasterDecisionToWaitingConfirmationLatency(
                    modelName, ipIndex, roleType.getCode(), group, latencyMs);
        }
        for (long latencyMs : updateResult.waitingToRunningObservedLatenciesMs()) {
            engineHealthReporter.reportFlexlbObservedWaitingToRunningLatency(
                    modelName, ipIndex, roleType.getCode(), group, latencyMs);
        }
        for (long latencyMs : updateResult.engineWaitingToRunningLatenciesMs()) {
            engineHealthReporter.reportEngineObservedWaitingToRunningLatency(
                    modelName, ipIndex, roleType.getCode(), group, latencyMs);
        }
        for (long latencyMs : updateResult.engineReceivedToWaitingLatenciesMs()) {
            engineHealthReporter.reportEngineObservedReceivedToWaitingLatency(
                    modelName, ipIndex, roleType.getCode(), group, latencyMs);
        }
        for (CacheHitFeedback feedback : updateResult.cacheHitFeedbacks()) {
            cacheAwareService.buildCacheHitComparison(feedback)
                    .thenAccept(this::reportCacheHitComparison)
                    .exceptionally(error -> {
                        logger.warn("Failed to build cache hit comparison, requestId={}",
                                feedback.requestId(), error);
                        return null;
                    });
        }
    }

    private void reportCacheHitComparison(CacheHitComparisonResult comparison) {
        if (comparison == null) {
            return;
        }
        engineHealthReporter.reportCacheHitComparisonMetrics(modelName, comparison);
        String json = JsonUtils.toStringOrEmpty(comparison);
        if (!json.isEmpty()) {
            pvLogger.info(json);
        }
    }

    private void reportFinishedPrefillTasks(Map<String, TaskInfo> finishedTaskInfo) {
        if ((roleType != RoleType.PREFILL && roleType != RoleType.PDFUSION)
                || finishedTaskInfo == null || finishedTaskInfo.isEmpty()) {
            return;
        }
        for (TaskInfo task : finishedTaskInfo.values()) {
            engineHealthReporter.reportPrefillWorkerStatusTask(
                    modelName, ipIndex, roleType.getCode(), group, task);
            Map<String, Object> event = new LinkedHashMap<>();
            event.put("event", "prefill_worker_status");
            event.put("requestId", task.getRequestId());
            event.put("model", modelName);
            event.put("workerIp", ip);
            event.put("workerPort", workerStatusPort);
            event.put("engineIndex", engineIndex);
            event.put("logicalWorker", logicalIpPort);
            event.put("role", roleType.getCode());
            event.put("group", group);
            event.put("inputQueueEnqueueTimeMs", task.getInputQueueEnqueueTimeMs());
            event.put("inputQueueDrainTimeMs", task.getInputQueueDrainTimeMs());
            event.put("remoteKvWaitMs", task.getRemoteKvWaitMs());
            event.put("firstTokenTimeMs", task.getFirstTokenTimeMs());
            event.put("hbmLocalMatchTokens", task.getHbmLocalMatchTokens());
            event.put("remoteKvAddedMatchTokens", task.getRemoteKvAddedMatchTokens());
            event.put("firstPrefillStepId", task.getFirstPrefillStepId());
            event.put("lastPrefillStepId", task.getLastPrefillStepId());
            event.put("prefillStepCount", task.getPrefillStepCount());
            event.put("prefillNonfinalChunkTokensMin", task.getPrefillNonfinalChunkTokensMin());
            event.put("prefillNonfinalChunkTokensMax", task.getPrefillNonfinalChunkTokensMax());
            event.put("inputQueueWaitMs", duration(task.getInputQueueDrainTimeMs(), task.getInputQueueEnqueueTimeMs()));
            long schedulerToRunningMs = duration(task.getRunningEnteredTimeMs(), task.getWaitingEnteredTimeMs());
            event.put("schedulerToRunningMs", schedulerToRunningMs);
            event.put("schedulerWaitMs", schedulerToRunningMs < 0 ? -1
                    : Math.max(0, schedulerToRunningMs - task.getRemoteKvWaitMs()));
            event.put("runningToFirstTokenMs", duration(task.getFirstTokenTimeMs(), task.getRunningEnteredTimeMs()));
            String json = JsonUtils.toStringOrEmpty(event);
            if (!json.isEmpty()) {
                pvLogger.info(json);
            }
        }
    }

    private long duration(long endTimeMs, long startTimeMs) {
        if (endTimeMs <= 0 || startTimeMs <= 0) {
            return -1;
        }
        return Math.max(0, endTimeMs - startTimeMs);
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

    private void updateKvCacheGroupMode(KvCacheGroupMode mode) {
        if (mode == null || mode == KvCacheGroupMode.UNSPECIFIED
                || mode == workerStatus.getKvCacheGroupMode()) {
            return;
        }
        workerStatus.setKvCacheGroupMode(mode);
    }

    private void handleException(Throwable ex, long startTime) {
        log("gRPC worker status check failed, msg=" + ex.getMessage());
        // Report specific error based on exception type
        if (ex.getMessage() != null && ex.getMessage().toLowerCase().contains(DEADLINE_EXCEEDED_MESSAGE.toLowerCase())) {
            logger.info("gRPC worker status check timeout, msg={}, ipPort: {}, rt_us: {}", ex.getMessage(), logicalIpPort, System.nanoTime() / 1000 - createTimeUs);
            reportStatusCheckFailure(BalanceStatusEnum.WORKER_STATUS_GRPC_TIMEOUT, startTime);
        } else {
            reportStatusCheckFailure(BalanceStatusEnum.WORKER_SERVICE_UNAVAILABLE, startTime);
        }
    }

    private void reportStatusCheckFailure(BalanceStatusEnum errorEnum, long startTime) {
        engineHealthReporter.reportStatusCheckerFail(modelName, errorEnum, ipIndex, roleType);
        engineHealthReporter.reportStatusCheckFailureLatency(
                modelName,
                errorEnum,
                ipIndex,
                roleType,
                System.nanoTime() / 1000 - startTime);
    }

    private void log(String msg) {
        logger.info("[gRPC][{}][{}][{}][{}][{}μs]: {}",
                id,
                site,
                logicalIpPort,
                modelName,
                System.nanoTime() / 1000 - createTimeUs,
                msg);
    }
}
