package org.flexlb.sync.runner;

import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.domain.worker.WorkerStatusResponse;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.enums.BalanceStatusEnum;
import org.flexlb.service.grpc.EngineGrpcService;
import org.flexlb.service.grpc.EngineStatusConverter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.util.IdUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Optional;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

import static org.flexlb.constant.CommonConstants.DEADLINE_EXCEEDED_MESSAGE;
import static org.flexlb.util.CommonUtils.toGrpcPort;

public class GrpcWorkerStatusRunner implements Runnable {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");

    private final String ipPort;
    private final String modelName;
    private final String site;
    private final String group;
    private final ConcurrentHashMap<String/*ipPort*/, WorkerStatus> workerStatuses;
    private final EngineHealthReporter engineHealthReporter;
    private final EngineGrpcService engineGrpcService;
    private final String ip;
    private final int port;
    private final int grpcPort;
    private final long startTime = System.currentTimeMillis();
    private final String id = IdUtils.fastUuid();
    private final long syncRequstTimeoutMs;
    private final RoleType roleType;

    public GrpcWorkerStatusRunner(String modelName, String ipPort, String site, String group,
                                  ConcurrentHashMap<String/*ip*/, WorkerStatus> workerStatuses,
                                  EngineHealthReporter engineHealthReporter,
                                  EngineGrpcService engineGrpcService,
                                  long syncRequstTimeoutMs,
                                  RoleType roleType) {
        this.ipPort = ipPort;
        String[] split = ipPort.split(":");
        this.ip = split[0];
        this.port = Integer.parseInt(split[1]);
        this.grpcPort = toGrpcPort(Integer.parseInt(split[1]));
        this.modelName = modelName;
        this.workerStatuses = workerStatuses;
        this.site = site;
        this.group = group;
        this.engineHealthReporter = engineHealthReporter;
        this.engineGrpcService = engineGrpcService;
        this.syncRequstTimeoutMs = syncRequstTimeoutMs;
        this.roleType = roleType;
    }

    @Override
    public void run() {

        logger.info("GrpcWorkerStatusRunner run for {}", ipPort);
        long startTime = System.currentTimeMillis();

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
            EngineRpcService.WorkerStatusPB workerStatusPB = engineGrpcService.getWorkerStatus(ip, grpcPort, latestFinishedTaskVersion, syncRequstTimeoutMs, roleType);
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
                engineHealthReporter.reportStatusCheckerFail(modelName, BalanceStatusEnum.RESPONSE_NULL);
                return;
            }
            logger.info("gRPC Worker Status - handled for {}, role:{} ", ipPort, newWorkerStatus.getRole());

            if (newWorkerStatus.getMessage() != null) {
                WorkerStatus workerStatus = getOrCreateWorkerStatus();
                workerStatus.setAlive(false);
                logger.info("query engine worker status via gRPC, msg={}", newWorkerStatus.getMessage());
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
                logger.info("query engine worker status via gRPC, version is not updated, currentVersion: {}, responseVersion: {}",
                        currentVersion, responseVersion);
                // Update basic worker status even when version is not updated
                workerStatus.setAlive(newWorkerStatus.isAlive());
                workerStatus.setDpSize(newWorkerStatus.getDpSize());
                workerStatus.setTpSize(newWorkerStatus.getTpSize());

                // Set expiration time to 3 seconds from now
                workerStatus.getExpirationTime().set(System.currentTimeMillis() + 3000);
                workerStatus.getLastUpdateTime().set(System.currentTimeMillis());

                // Report success even when version is not updated
                engineHealthReporter.reportStatusCheckerSuccess(modelName, workerStatus);

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
            workerStatus.setRunningTaskList(newWorkerStatus.getRunningTaskInfo());

            // 更新完结和过期的任务
            workerStatus.clearFinishedTaskAndTimeoutTask(newWorkerStatus.getFinishedTaskList());

            engineHealthReporter.reportStatusCheckerSuccess(modelName, workerStatus);

            // Set expiration time to 3 seconds from now
            workerStatus.getExpirationTime().set(System.currentTimeMillis() + 3000);
            workerStatus.getLastUpdateTime().set(System.currentTimeMillis());
            logWorkerStatusUpdate(startTime, workerStatus);

        } catch (Throwable e) {
            log("engine worker status check via gRPC exception, msg: " + e.getMessage(), e);
            engineHealthReporter.reportStatusCheckerFail(modelName, BalanceStatusEnum.UNKNOWN_ERROR);
        }
    }

    private void logWorkerStatusUpdate(long startTime, WorkerStatus workerStatus) {
        logger.info("gRPC Worker Status - {}, role:{}, running_queue_tokens:{}, cost:{}",
                ipPort,
                workerStatus.getRole(),
                workerStatus.getRunningQueueTime(),
                System.currentTimeMillis() - startTime);
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
        log("gRPC worker status check failed", ex);
        // Report specific error based on exception type
        if (ex.getMessage() != null && ex.getMessage().toLowerCase().contains(DEADLINE_EXCEEDED_MESSAGE.toLowerCase())) {
            engineHealthReporter.reportStatusCheckerFail(modelName, BalanceStatusEnum.WORKER_STATUS_GRPC_TIMEOUT);
        } else {
            engineHealthReporter.reportStatusCheckerFail(modelName, BalanceStatusEnum.WORKER_SERVICE_UNAVAILABLE);
        }
    }

    private void log(String msg, Throwable e) {
        logger.info("[gRPC][{}][{}][{}][{}][{}ms]: {}",
                id,
                site,
                ipPort,
                modelName,
                System.currentTimeMillis() - startTime,
                msg,
                e);
    }
}