package org.flexlb.sync.runner;

import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
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

import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.atomic.AtomicLong;

import static org.flexlb.constant.CommonConstants.DEADLINE_EXCEEDED_MESSAGE;

public class GrpcWorkerStatusRunner implements Runnable {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");

    private final String ipPort;
    private final String modelName;
    private final String site;
    private final String group;
    private final Map<String/*ipPort*/, WorkerStatus> workerStatuses;
    private final EngineHealthReporter engineHealthReporter;
    private final EngineGrpcService engineGrpcService;
    private final String ip;
    private final int port;
    private final int grpcPort;
    private final long startTime = System.nanoTime() / 1000;
    private final String id = IdUtils.fastUuid();
    private final long syncRequestTimeoutMs;

    public GrpcWorkerStatusRunner(String modelName, String ipPort, String site, String group,
                                  Map<String/*ip*/, WorkerStatus> workerStatuses,
                                  EngineHealthReporter engineHealthReporter,
                                  EngineGrpcService engineGrpcService,
                                  long syncRequestTimeoutMs) {
        this.ipPort = ipPort;
        String[] split = ipPort.split(":");
        this.ip = split[0];
        this.port = Integer.parseInt(split[1]);
        this.grpcPort = CommonUtils.toGrpcPort(Integer.parseInt(split[1]));
        this.modelName = modelName;
        this.workerStatuses = workerStatuses;
        this.site = site;
        this.group = group;
        this.engineHealthReporter = engineHealthReporter;
        this.engineGrpcService = engineGrpcService;
        this.syncRequestTimeoutMs = syncRequestTimeoutMs;
    }

    @Override
    public void run() {

        logger.info("GrpcWorkerStatusRunner run for {}", ipPort);
        long startTime = System.nanoTime() / 1000;

        WorkerStatus workerStatus = workerStatuses.get(ipPort);
        long latestFinishedTaskVersion = Optional.ofNullable(workerStatus)
                .map(WorkerStatus::getLatestFinishedTaskVersion)
                .map(AtomicLong::get)
                .orElse(-1L);

        WorkerStatusResponse response = launchGrpcStatusCheck(ip, grpcPort, latestFinishedTaskVersion);
        handleStatusResponse(response, startTime);
    }

    private WorkerStatusResponse launchGrpcStatusCheck(String ip, int grpcPort, long latestFinishedTaskVersion) {
        try {
            EngineRpcService.WorkerStatusPB workerStatusPB = engineGrpcService.getWorkerStatus(ip, grpcPort, latestFinishedTaskVersion, syncRequestTimeoutMs);
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
                engineHealthReporter.reportStatusCheckerFail(modelName, BalanceStatusEnum.RESPONSE_NULL, ip);
                return;
            }

            if (newWorkerStatus.getMessage() != null) {
                WorkerStatus workerStatus = getOrCreateWorkerStatus();
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

            WorkerStatus workerStatus = getOrCreateWorkerStatus();

            workerStatus.setSite(site);
            workerStatus.setGroup(group);
            workerStatus.setRole(newWorkerStatus.getRole());

            Long currentVersion = workerStatus.getStatusVersion();
            if (currentVersion >= responseVersion) {
                // 版本相同但是也需要更新 expirationTime
                // Set expiration time to 3 seconds from now
                workerStatus.getStatusLastUpdateTime().set(System.nanoTime() / 1000);
                // 更新任务状态
                List<TaskInfo> runningTaskInfo = newWorkerStatus.getRunningTaskInfo();
                List<TaskInfo> finishedTaskList = newWorkerStatus.getFinishedTaskList();
                workerStatus.updateTaskStates(runningTaskInfo, finishedTaskList);
                logger.info("query engine worker status via gRPC, version is not updated, currentVersion: {}, responseVersion: {}",
                        currentVersion, responseVersion);
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

            List<TaskInfo> runningTaskInfo = newWorkerStatus.getRunningTaskInfo();
            List<TaskInfo> finishedTaskList = newWorkerStatus.getFinishedTaskList();
            workerStatus.setRunningTaskList(runningTaskInfo);

            // 更新本地任务状态（包含检查丢失、更新运行、清理完成）
            workerStatus.updateTaskStates(runningTaskInfo, finishedTaskList);

            // 纠偏运行队列总排队时间
            workerStatus.updateRunningQueueTime();

            engineHealthReporter.reportStatusCheckerSuccess(modelName, workerStatus,
                    Optional.ofNullable(runningTaskInfo).map(List::size).orElse(0),
                    Optional.ofNullable(finishedTaskList).map(List::size).orElse(0));

            workerStatus.getStatusLastUpdateTime().set(System.nanoTime() / 1000);
            logWorkerStatusUpdate(startTime, workerStatus);

        } catch (Throwable e) {
            log("engine worker status check via gRPC exception, msg: " + e.getMessage());
            engineHealthReporter.reportStatusCheckerFail(modelName, BalanceStatusEnum.UNKNOWN_ERROR, ip);
        }
    }

    private void logWorkerStatusUpdate(long startTime, WorkerStatus workerStatus) {
        logger.info("gRPC Worker Status - {}, role:{}, running_queue_tokens:{}, cost:{}",
                ipPort,
                workerStatus.getRole(),
                workerStatus.getRunningQueueTime(),
                System.nanoTime() / 1000 - startTime);
    }

    private WorkerStatus getOrCreateWorkerStatus() {
        WorkerStatus workerStatus = workerStatuses.get(ipPort);
        if (workerStatus == null) {
            logger.info("workerStatuses.get(ipPort) is null for cache status gRPC call, ipPort: {}", ipPort);
            workerStatus = new WorkerStatus();
            workerStatus.setIp(ip);
            workerStatus.setPort(port);
            workerStatuses.put(ipPort, workerStatus);
        }
        return workerStatus;
    }

    private void handleException(Throwable ex) {
        log("gRPC worker status check failed, msg=" + ex.getMessage());
        // Report specific error based on exception type
        if (ex.getMessage() != null && ex.getMessage().toLowerCase().contains(DEADLINE_EXCEEDED_MESSAGE.toLowerCase())) {
            logger.info("gRPC worker status check timeout, msg=" + ex.getMessage() + ", ipPort: " + ipPort + ", rt: " + (System.nanoTime() / 1000 - startTime));
            engineHealthReporter.reportStatusCheckerFail(modelName, BalanceStatusEnum.WORKER_STATUS_GRPC_TIMEOUT, ip);
        } else {
            engineHealthReporter.reportStatusCheckerFail(modelName, BalanceStatusEnum.WORKER_SERVICE_UNAVAILABLE, ip);
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