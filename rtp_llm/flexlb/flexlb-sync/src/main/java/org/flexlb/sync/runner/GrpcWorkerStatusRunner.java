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
import org.flexlb.util.CommonUtils;
import org.flexlb.util.IdUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicLong;

import static org.flexlb.constant.CommonConstants.DEADLINE_EXCEEDED_MESSAGE;

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
    private final long startTime = System.nanoTime() / 1000;
    private final String id = IdUtils.fastUuid();
    private final long syncRequestTimeoutMs;

    public GrpcWorkerStatusRunner(String modelName, String ipPort, String site, RoleType roleType, String group,
                                  WorkerStatus workerStatus,
                                  EngineHealthReporter engineHealthReporter,
                                  EngineGrpcService engineGrpcService,
                                  long syncRequestTimeoutMs) {
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
        this.syncRequestTimeoutMs = syncRequestTimeoutMs;
    }

    @Override
    public void run() {
        try {
            logger.info("GrpcWorkerStatusRunner run for {}", ipPort);
            long startTime = System.nanoTime() / 1000;

            long latestFinishedTaskVersion = Optional.ofNullable(workerStatus.getLatestFinishedTaskVersion())
                    .map(AtomicLong::get)
                    .orElse(-1L);

            WorkerStatusResponse response = launchGrpcStatusCheck(ip, grpcPort, latestFinishedTaskVersion);
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

            Long currentVersion = workerStatus.getStatusVersion();
            if (currentVersion >= responseVersion) {
                logger.info("query engine worker status via gRPC, version is not updated, currentVersion: {}, responseVersion: {}",
                        currentVersion, responseVersion);
                // Update basic worker status even when version is not updated
                workerStatus.setAlive(newWorkerStatus.isAlive());
                workerStatus.setDpSize(newWorkerStatus.getDpSize());
                workerStatus.setTpSize(newWorkerStatus.getTpSize());

                // Set expiration time to 3 seconds from now
                workerStatus.getStatusLastUpdateTime().set(System.nanoTime() / 1000);

                // 更新任务状态
                Map<String, TaskInfo> waitingTaskInfo = newWorkerStatus.getWaitingTaskInfo();
                Map<String, TaskInfo> runningTaskInfo = newWorkerStatus.getRunningTaskInfo();
                Map<String, TaskInfo> finishedTaskInfo = newWorkerStatus.getFinishedTaskInfo();
                workerStatus.updateTaskStates(waitingTaskInfo, runningTaskInfo, finishedTaskInfo);

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
            workerStatus.setVersion(String.valueOf(newWorkerStatus.getVersion()));
            workerStatus.setStatusVersion(responseVersion);

            Map<String, TaskInfo> waitingTaskInfo = newWorkerStatus.getWaitingTaskInfo();
            Map<String, TaskInfo> runningTaskInfo = newWorkerStatus.getRunningTaskInfo();
            Map<String, TaskInfo> finishedTaskInfo = newWorkerStatus.getFinishedTaskInfo();
            workerStatus.setWaitingTaskList(waitingTaskInfo);
            workerStatus.setRunningTaskList(runningTaskInfo);

            // 更新本地任务状态（包含检查丢失、更新运行、清理完成）
            workerStatus.updateTaskStates(waitingTaskInfo, runningTaskInfo, finishedTaskInfo);

            // 纠偏运行队列总排队时间
            workerStatus.updateRunningQueueTime();

            engineHealthReporter.reportStatusCheckerSuccess(modelName, workerStatus,
                    Optional.ofNullable(runningTaskInfo).map(Map::size).orElse(0),
                    Optional.ofNullable(finishedTaskInfo).map(Map::size).orElse(0));

            workerStatus.getStatusLastUpdateTime().set(System.nanoTime() / 1000);
            logWorkerStatusUpdate(startTime, workerStatus);

        } catch (Throwable e) {
            log("engine worker status check via gRPC exception, msg: " + e.getMessage());
            engineHealthReporter.reportStatusCheckerFail(modelName, BalanceStatusEnum.UNKNOWN_ERROR, ip, roleType);
        }
    }

    private void logWorkerStatusUpdate(long startTime, WorkerStatus workerStatus) {
        logger.info("gRPC Worker Status - {}, role:{}, running_queue_tokens:{}, cost:{}",
                ipPort,
                workerStatus.getRole(),
                workerStatus.getRunningQueueTime(),
                System.nanoTime() / 1000 - startTime);
    }

    private void handleException(Throwable ex) {
        log("gRPC worker status check failed, msg=" + ex.getMessage());
        // Report specific error based on exception type
        if (ex.getMessage() != null && ex.getMessage().toLowerCase().contains(DEADLINE_EXCEEDED_MESSAGE.toLowerCase())) {
            logger.info("gRPC worker status check timeout, msg={}, ipPort: {}, rt: {}", ex.getMessage(), ipPort, System.nanoTime() / 1000 - startTime);
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
                System.nanoTime() / 1000 - startTime,
                msg);
    }
}