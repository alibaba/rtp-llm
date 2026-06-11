package org.flexlb.sync.runner;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.scheduler.FlexlbBatchScheduler;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.domain.worker.WorkerStatusResponse;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.enums.BalanceStatusEnum;
import org.flexlb.enums.TaskPhase;
import org.flexlb.service.grpc.EngineGrpcService;
import org.flexlb.service.grpc.EngineStatusConverter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.util.CommonUtils;
import org.flexlb.util.IdUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;

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
    private final FlexlbBatchScheduler batchScheduler;
    private final String ip;
    private final int grpcPort;
    private final long createTimeUs = System.nanoTime() / 1000;
    private final String id = IdUtils.fastUuid();
    private final long syncRequestTimeoutMs;
    private final EndpointRegistry endpointRegistry;

    public GrpcWorkerStatusRunner(String modelName, String ipPort, String site, RoleType roleType, String group,
                                  WorkerStatus workerStatus,
                                  EngineHealthReporter engineHealthReporter,
                                  EngineGrpcService engineGrpcService,
                                  long syncRequestTimeoutMs,
                                  FlexlbBatchScheduler batchScheduler,
                                  EndpointRegistry endpointRegistry) {
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
        this.batchScheduler = batchScheduler;
        this.endpointRegistry = endpointRegistry;
    }

    @Override
    public void run() {
        try {
            logger.info("GrpcWorkerStatusRunner run for {}", ipPort);
            long startTime = System.nanoTime() / 1000;

            long latestFinishedTaskVersion = workerStatus.getLatestFinishedTaskVersion().get();

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
            workerStatus.setAlive(false);
            return null;
        }
    }

    private void handleStatusResponse(WorkerStatusResponse newWorkerStatus, long startTime) {
        try {
            if (newWorkerStatus == null) {
                logger.info("query engine worker status via gRPC, response body is null");
                engineHealthReporter.reportStatusCheckerFail(modelName, BalanceStatusEnum.RESPONSE_NULL, ip, roleType);
                return;
            }


            // Only report success worker status check info
            engineHealthReporter.reportStatusCheckRemoteInfo(modelName, ipPort, newWorkerStatus.getRole(), startTime);

            Long responseVersion = newWorkerStatus.getStatusVersion();
            if (responseVersion == 0L) {
                logger.info("workerStatuses.get(ip) is null for gRPC call");
                return;
            }

            WorkerEndpoint ep = endpointRegistry != null ? endpointRegistry.get(ipPort) : null;
            if (ep != null) {
                ep.updateFromGrpcResponse(newWorkerStatus);
                // Restore site/group that were set during construction (updateFromGrpcResponse
                // preserves existing values, but ensure they match constructor params)
                ep.setSite(site);
                ep.setGroup(group);
            }

            long currentVersion = workerStatus.getStatusVersion().get();
            if (currentVersion >= responseVersion) {
                logger.info("query engine worker status via gRPC, version is not updated, currentVersion: {}, responseVersion: {}",
                        currentVersion, responseVersion);

                Map<String, TaskInfo> runningTaskInfo = newWorkerStatus.getRunningTaskInfo();
                Map<String, TaskInfo> finishedTaskInfo = newWorkerStatus.getFinishedTaskInfo();
                List<Long> finished = extractFinishedRequestIds(finishedTaskInfo);
                if (batchScheduler != null && !finished.isEmpty()) {
                    batchScheduler.onRequestsFinished(finished);
                }

                if (endpointRegistry != null) {
                    calibrateEndpoint(runningTaskInfo, finishedTaskInfo);
                }

                // Report success even when version is not updated
                engineHealthReporter.reportStatusCheckerSuccess(modelName, workerStatus,
                    Optional.ofNullable(runningTaskInfo).map(Map::size).orElse(0),
                    Optional.ofNullable(finishedTaskInfo).map(Map::size).orElse(0));

                logWorkerStatusUpdate(startTime, workerStatus);
                return;
            }

            Map<String, TaskInfo> runningTaskInfo = newWorkerStatus.getRunningTaskInfo();
            Map<String, TaskInfo> finishedTaskInfo = newWorkerStatus.getFinishedTaskInfo();

            List<Long> finished = extractFinishedRequestIds(finishedTaskInfo);
            if (batchScheduler != null && !finished.isEmpty()) {
                batchScheduler.onRequestsFinished(finished);
            }

            if (endpointRegistry != null) {
                calibrateEndpoint(runningTaskInfo, finishedTaskInfo);
            }

            engineHealthReporter.reportStatusCheckerSuccess(modelName, workerStatus,
                    Optional.ofNullable(runningTaskInfo).map(Map::size).orElse(0),
                    Optional.ofNullable(finishedTaskInfo).map(Map::size).orElse(0));

            logWorkerStatusUpdate(startTime, workerStatus);

        } catch (Throwable e) {
            log("engine worker status check via gRPC exception, msg: " + e.getMessage());
            engineHealthReporter.reportStatusCheckerFail(modelName, BalanceStatusEnum.UNKNOWN_ERROR, ip, roleType);
        }
    }

    private static List<Long> extractFinishedRequestIds(Map<String, TaskInfo> finishedTaskInfo) {
        if (finishedTaskInfo == null) {
            return List.of();
        }
        List<Long> ids = new ArrayList<>(finishedTaskInfo.size());
        for (TaskInfo task : finishedTaskInfo.values()) {
            ids.add(task.getRequestId());
        }
        return ids;
    }

    private void logWorkerStatusUpdate(long startTime, WorkerStatus workerStatus) {
        logger.info("gRPC Worker Status - {}, role:{}, alive:{}, concurrency:{}, "
                        + "step_latency_ms:{}, iterate_count:{}, "
                        + "dp_rank:{}, dp_size:{}, tp_size:{}, "
                        + "avail_kv_tokens:{}, used_kv_tokens:{}, "
                        + "waiting_tasks:{}, running_tasks:{}, "
                        + "version:{}, sync_cost_us:{}",
                ipPort,
                workerStatus.getRole(),
                workerStatus.isAlive(),
                workerStatus.getAvailableConcurrency(),
                workerStatus.getStepLatencyMs(),
                workerStatus.getIterateCount(),
                workerStatus.getDpRank(),
                workerStatus.getDpSize(),
                workerStatus.getTpSize(),
                workerStatus.getAvailableKvCacheTokens(),
                workerStatus.getUsedKvCacheTokens(),
                workerStatus.getRunningTaskList() != null ? workerStatus.getRunningTaskList().values().stream().filter(t -> t.getPhase() != TaskPhase.RUNNING).count() : 0,
                workerStatus.getRunningTaskList() != null ? workerStatus.getRunningTaskList().size() : 0,
                workerStatus.getStatusVersion(),
                System.nanoTime() / 1000 - startTime);
    }

    private void handleException(Throwable ex) {
        log("gRPC worker status check failed, msg=" + ex.getMessage());
        // Report specific error based on exception type
        if (ex.getMessage() != null && ex.getMessage().toLowerCase().contains(DEADLINE_EXCEEDED_MESSAGE.toLowerCase())) {
            logger.info("gRPC worker status check timeout, msg={}, ipPort: {}, rt: {}", ex.getMessage(), ipPort, System.nanoTime() / 1000 - createTimeUs);
            engineHealthReporter.reportStatusCheckerFail(modelName, BalanceStatusEnum.WORKER_STATUS_GRPC_TIMEOUT, ip, roleType);
        } else {
            engineHealthReporter.reportStatusCheckerFail(modelName, BalanceStatusEnum.WORKER_SERVICE_UNAVAILABLE, ip, roleType);
        }
    }

    private void calibrateEndpoint(Map<String, TaskInfo> runningTaskInfo,
                                    Map<String, TaskInfo> finishedTaskInfo) {
        if (roleType == RoleType.PREFILL) {
            PrefillEndpoint ep = endpointRegistry.getPrefill(ipPort);
            if (ep != null) {
                ep.calibrate(finishedTaskInfo, runningTaskInfo);
            }
        } else if (roleType == RoleType.DECODE) {
            DecodeEndpoint ep = endpointRegistry.getDecode(ipPort);
            if (ep != null) {
                ep.calibrate(runningTaskInfo, finishedTaskInfo,
                        ep.getAvailableKvCacheTokens().get());
            }
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
