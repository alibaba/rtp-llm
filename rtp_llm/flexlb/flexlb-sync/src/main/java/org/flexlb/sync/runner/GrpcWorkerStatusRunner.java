package org.flexlb.sync.runner;

import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.domain.worker.WorkerStatusResponse;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.enums.BalanceStatusEnum;
import org.flexlb.service.grpc.EngineGrpcService;
import org.flexlb.service.grpc.EngineStatusConverter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.config.VitStatusConfig;
import org.flexlb.sync.util.GrpcStatusUtils;
import org.flexlb.util.CommonUtils;
import org.flexlb.util.IdUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;
import java.util.Optional;

public class GrpcWorkerStatusRunner implements Runnable {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");
    private final String ipPort;
    private final String modelName;
    private final String site;
    private final RoleType roleType;
    private final String group;
    private final WorkerStatus workerStatus;
    private final EngineHealthReporter engineHealthReporter;
    private final EngineGrpcService engineGrpcService;
    private final String ip;
    private final int grpcPort;
    private final long createTimeUs = System.nanoTime() / 1000;
    private final String id = IdUtils.fastUuid();
    private final long syncRequestTimeoutMs;
    private final boolean retainVitAliveOnTimeout;

    public GrpcWorkerStatusRunner(String modelName, String ipPort, String site, RoleType roleType, String group,
                                  WorkerStatus workerStatus,
                                  EngineHealthReporter engineHealthReporter,
                                  EngineGrpcService engineGrpcService,
                                  long syncRequestTimeoutMs) {
        this(modelName, ipPort, site, roleType, group, workerStatus, engineHealthReporter,
                engineGrpcService, syncRequestTimeoutMs, VitStatusConfig.SYNC_REQUEST_TIMEOUT_MS,
                VitStatusConfig.RETAIN_ALIVE_ON_TIMEOUT);
        if (roleType == RoleType.VIT) {
            VitStatusConfig.warnIfRetentionWindowAtRisk(this.syncRequestTimeoutMs);
        }
    }

    GrpcWorkerStatusRunner(String modelName, String ipPort, String site, RoleType roleType, String group,
                           WorkerStatus workerStatus,
                           EngineHealthReporter engineHealthReporter,
                           EngineGrpcService engineGrpcService,
                           long syncRequestTimeoutMs,
                           long vitSyncRequestTimeoutMs,
                           boolean retainVitAliveOnTimeout) {
        this.ipPort = ipPort;
        String[] split = ipPort.split(":");
        this.ip = split[0];
        this.grpcPort = CommonUtils.toGrpcPort(Integer.parseInt(split[1]));
        this.modelName = modelName;
        this.workerStatus = workerStatus;
        this.site = site;
        this.roleType = roleType;
        this.group = group;
        this.engineHealthReporter = engineHealthReporter;
        this.engineGrpcService = engineGrpcService;
        // The VIT-specific timeout is a lower bound. A larger global sync timeout
        // remains in force instead of being shortened for VIT workers.
        this.syncRequestTimeoutMs = roleType == RoleType.VIT
                ? Math.max(syncRequestTimeoutMs, vitSyncRequestTimeoutMs)
                : syncRequestTimeoutMs;
        this.retainVitAliveOnTimeout = retainVitAliveOnTimeout;
    }

    @Override
    public void run() {
        try {
            logger.info("GrpcWorkerStatusRunner run for {}", ipPort);
            long startTime = System.nanoTime() / 1000;

            long latestFinishedTaskVersion = workerStatus.getLatestFinishedTaskVersion().get();

            StatusCheckResult result = launchGrpcStatusCheck(ip, grpcPort, latestFinishedTaskVersion);
            handleStatusResponse(result, startTime);
        } finally {
            workerStatus.getStatusCheckInProgress().set(false);
        }
    }

    private StatusCheckResult launchGrpcStatusCheck(String ip, int grpcPort, long latestFinishedTaskVersion) {
        try {
            EngineRpcService.WorkerStatusPB workerStatusPB = engineGrpcService.getWorkerStatus(ip, grpcPort, latestFinishedTaskVersion, syncRequestTimeoutMs, roleType);
            return new StatusCheckResult(EngineStatusConverter.convertToWorkerStatusResponse(workerStatusPB), false);
        } catch (Throwable throwable) {
            boolean deadlineExceeded = GrpcStatusUtils.isDeadlineExceeded(throwable);
            handleException(throwable, deadlineExceeded);
            WorkerStatusResponse errorResponse = new WorkerStatusResponse();
            errorResponse.setMessage("Worker status gRPC call failed: " + throwable.getMessage());
            return new StatusCheckResult(errorResponse, deadlineExceeded);
        }
    }

    private void handleStatusResponse(StatusCheckResult result, long startTime) {
        WorkerStatusResponse newWorkerStatus = result.response;
        try {
            if (newWorkerStatus == null) {
                logger.info("query engine worker status via gRPC, response body is null");
                engineHealthReporter.reportStatusCheckerFail(
                        modelName, BalanceStatusEnum.RESPONSE_NULL, roleType);
                return;
            }

            String errorMessage = newWorkerStatus.getMessage();
            if (errorMessage != null) {
                // The proxy owns immediate health for its concrete child workers. FlexLB
                // tracks the aggregate proxy endpoint, so a transient VIT deadline keeps
                // the last state; ExpirationCleaner removes the endpoint after the
                // VIT-specific stale-status window if no later heartbeat succeeds.
                boolean retainLastAliveStatus = shouldRetainVitAlive(result.deadlineExceeded);
                if (!retainLastAliveStatus) {
                    workerStatus.setAlive(false);
                }
                logger.error("query engine worker status via gRPC, msg={}, retainLastAliveStatus={}",
                        errorMessage, retainLastAliveStatus);
                return;
            }

            // Only report success worker status check info
            engineHealthReporter.reportStatusCheckRemoteInfo(
                    modelName, newWorkerStatus.getRole(), startTime);

            Long responseVersion = newWorkerStatus.getStatusVersion();
            if (responseVersion == 0L) {
                logger.info("workerStatuses.get(ip) is null for gRPC call");
                return;
            }

            workerStatus.setSite(site);
            workerStatus.setGroup(group);
            workerStatus.setRole(newWorkerStatus.getRole());

            long currentVersion = workerStatus.getStatusVersion().get();
            if (currentVersion >= responseVersion) {
                logger.info("query engine worker status via gRPC, version is not updated, currentVersion: {}, responseVersion: {}",
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
                workerStatus.setWaitingTaskList(waitingTaskInfo);
                workerStatus.setRunningTaskList(runningTaskInfo);
                workerStatus.updateTaskStates(waitingTaskInfo, runningTaskInfo, finishedTaskInfo);
                workerStatus.updateRunningQueueTime();

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
            workerStatus.getStatusVersion().set(responseVersion != null ? responseVersion : -1L);
            workerStatus.getLatestFinishedTaskVersion().set(newWorkerStatus.getLatestFinishedVersion() != null ? newWorkerStatus.getLatestFinishedVersion() : -1L);

            Map<String, TaskInfo> waitingTaskInfo = newWorkerStatus.getWaitingTaskInfo();
            Map<String, TaskInfo> runningTaskInfo = newWorkerStatus.getRunningTaskInfo();
            Map<String, TaskInfo> finishedTaskInfo = newWorkerStatus.getFinishedTaskInfo();
            workerStatus.setWaitingTaskList(waitingTaskInfo);
            workerStatus.setRunningTaskList(runningTaskInfo);

            // Update local task state (including checking lost, updating running, and cleaning completed)
            workerStatus.updateTaskStates(waitingTaskInfo, runningTaskInfo, finishedTaskInfo);

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
            engineHealthReporter.reportStatusCheckerFail(
                    modelName, BalanceStatusEnum.UNKNOWN_ERROR, roleType);
        }
    }

    private void logWorkerStatusUpdate(long startTime, WorkerStatus workerStatus) {
        logger.info("gRPC Worker Status - {}, role:{}, running_queue_tokens:{}, cost:{}",
                ipPort,
                workerStatus.getRole(),
                workerStatus.getRunningQueueTime(),
                System.nanoTime() / 1000 - startTime);
    }

    private boolean shouldRetainVitAlive(boolean deadlineExceeded) {
        return roleType == RoleType.VIT && deadlineExceeded && retainVitAliveOnTimeout;
    }

    private void handleException(Throwable ex, boolean deadlineExceeded) {
        log("gRPC worker status check failed, msg=" + ex.getMessage());
        // Report specific error based on exception type
        if (deadlineExceeded) {
            logger.info("gRPC worker status check timeout, msg={}, ipPort: {}, rt: {}", ex.getMessage(), ipPort, System.nanoTime() / 1000 - createTimeUs);
            engineHealthReporter.reportStatusCheckerFail(
                    modelName, BalanceStatusEnum.WORKER_STATUS_GRPC_TIMEOUT, roleType);
        } else {
            engineHealthReporter.reportStatusCheckerFail(
                    modelName, BalanceStatusEnum.WORKER_SERVICE_UNAVAILABLE, roleType);
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
    private static class StatusCheckResult {
        private final WorkerStatusResponse response;
        private final boolean deadlineExceeded;

        private StatusCheckResult(WorkerStatusResponse response, boolean deadlineExceeded) {
            this.response = response;
            this.deadlineExceeded = deadlineExceeded;
        }
    }
}
