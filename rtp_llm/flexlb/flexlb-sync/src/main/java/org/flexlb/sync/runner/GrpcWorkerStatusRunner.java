package org.flexlb.sync.runner;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.scheduler.PriorityScheduler;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
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
import java.util.concurrent.CompletionException;
import java.util.concurrent.Executor;

import static org.flexlb.constant.CommonConstants.DEADLINE_EXCEEDED_MESSAGE;

public class GrpcWorkerStatusRunner implements Runnable {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");

    private final String ipPort;
    private final String modelName;
    private final String site;
    private final RoleType roleType;
    private final String group;
    private final WorkerStatus workerStatus;
    private final Map<String, WorkerStatus> workerStatusMap;
    private final EngineHealthReporter engineHealthReporter;
    private final EngineGrpcService engineGrpcService;
    private final PriorityScheduler priorityScheduler;
    private final String ip;
    private final int grpcPort;
    private final long createTimeUs = System.nanoTime() / 1000;
    private final String id = IdUtils.fastUuid();
    private final long syncRequestTimeoutMs;
    private static final int MAX_CONSECUTIVE_FAILURES = 3;
    private final EndpointRegistry endpointRegistry;
    private final Executor callbackExecutor;
    private org.flexlb.sync.status.WorkerRetirementService retirementService;

    public GrpcWorkerStatusRunner withRetirementService(org.flexlb.sync.status.WorkerRetirementService service) {
        this.retirementService = service;
        return this;
    }

    private void retireWorker() {
        if (retirementService != null) retirementService.retire(workerStatusMap, roleType, ipPort, workerStatus);
        else { workerStatus.setAlive(false); if (endpointRegistry != null) endpointRegistry.remove(roleType, ipPort, workerStatus); }
    }

    public GrpcWorkerStatusRunner(String modelName, String ipPort, String site, RoleType roleType, String group,
                                  WorkerStatus workerStatus,
                                  Map<String, WorkerStatus> workerStatusMap,
                                  EngineHealthReporter engineHealthReporter,
                                  EngineGrpcService engineGrpcService,
                                  long syncRequestTimeoutMs,
                                  PriorityScheduler priorityScheduler,
                                  EndpointRegistry endpointRegistry,
                                  Executor callbackExecutor) {
        this.ipPort = ipPort;
        String[] split = ipPort.split(":");
        this.ip = split[0];
        this.grpcPort = CommonUtils.toGrpcPort(Integer.parseInt(split[1]));
        this.modelName = modelName;
        this.workerStatus = workerStatus;
        this.workerStatusMap = workerStatusMap;
        this.site = site;
        this.roleType = roleType;
        this.group = group;
        this.engineHealthReporter = engineHealthReporter;
        this.engineGrpcService = engineGrpcService;
        this.syncRequestTimeoutMs = syncRequestTimeoutMs;
        this.priorityScheduler = priorityScheduler;
        this.endpointRegistry = endpointRegistry;
        this.callbackExecutor = callbackExecutor;
    }

    @Override
    public void run() {
        boolean asyncInitiated = false;
        try {
            logger.debug("GrpcWorkerStatusRunner run for {}", ipPort);
            long startTime = System.nanoTime() / 1000;

            long latestFinishedTaskVersion = workerStatus.getLatestFinishedTaskVersion().get();

            workerStatus.getPollState().rpcStarted(startTime, syncRequestTimeoutMs);
            engineGrpcService.getWorkerStatusAsync(ip, grpcPort, latestFinishedTaskVersion,
                            syncRequestTimeoutMs, roleType)
                    .whenComplete((raw, ex) -> {
                        long receivedUs = System.nanoTime() / 1000;
                        try {
                            if (workerStatus.getPollState().isRetired()) return;
                            if (ex != null) {
                                Throwable failure = ex instanceof CompletionException ? ex.getCause() : ex;
                                handleException(failure);
                                if (workerStatus.getConsecutiveFailures().incrementAndGet() >= MAX_CONSECUTIVE_FAILURES) retireWorker();
                            } else if (raw == null || !raw.getAlive()) {
                                retireWorker();
                            } else {
                                // Publish transport success before conversion or application can queue.
                                workerStatus.getPollState().rpcSucceeded(receivedUs);
                                workerStatus.getConsecutiveFailures().set(0);
                                workerStatus.getPollState().offer(receivedUs, () -> {
                                    workerStatus.lock.lock();
                                    try {
                                        if (workerStatus.getPollState().isRetired() || workerStatusMap.get(ipPort) != workerStatus) return;
                                        try {
                                            var response = EngineStatusConverter.convertToWorkerStatusResponse(raw);
                                            if (handleStatusResponse(response, startTime)) {
                                                workerStatus.getPollState().applied(receivedUs);
                                                workerStatus.getStatusLastUpdateTime().set(receivedUs);
                                            }
                                        } catch (Throwable localFailure) {
                                            logger.error("Local status conversion/application failed, rpc=OK, worker={}", ipPort, localFailure);
                                        }
                                    } finally { workerStatus.lock.unlock(); }
                                }, callbackExecutor);
                            }
                        } finally {
                            workerStatus.getPollState().rpcFinished();
                            workerStatus.getStatusCheckInProgress().set(false);
                        }
                    });
            asyncInitiated = true;
        } finally {
            if (!asyncInitiated) {
                workerStatus.getPollState().rpcFinished();
                workerStatus.getStatusCheckInProgress().set(false);
            }
        }
    }

    private boolean handleStatusResponse(WorkerStatusResponse newWorkerStatus, long startTime) {
        try {
            if (newWorkerStatus == null) {
                logger.debug("query engine worker status via gRPC, response body is null");
                engineHealthReporter.reportStatusCheckerFail(
                        modelName, BalanceStatusEnum.RESPONSE_NULL, roleType);
                return false;
            }
            if (workerStatusMap != null && workerStatusMap.get(ipPort) != workerStatus) {
                logger.debug("Ignore stale worker status callback for {}, role: {}", ipPort, roleType);
                return false;
            }
            // Only report success worker status check info
            engineHealthReporter.reportStatusCheckRemoteInfo(
                    modelName, newWorkerStatus.getRole().name(), startTime);

            Long responseVersion = newWorkerStatus.getStatusVersion();
            if (responseVersion == 0L) {
                logger.debug("workerStatuses.get(ip) is null for gRPC call");
                return false;
            }
            workerStatus.setSite(site);
            workerStatus.setGroup(group);

            long currentVersion = workerStatus.getStatusVersion().get();
            WorkerEndpoint ep = endpointRegistry != null ? endpointRegistry.get(roleType, ipPort) : null;
            boolean versionAdvanced = currentVersion < responseVersion;

            if (versionAdvanced) {
                // 1. WorkerStatusResponse directly updates WorkerStatus
                workerStatus.updateFromResponse(newWorkerStatus);

                if (endpointRegistry != null) {
                    if (workerStatus.isAlive()) {
                        ep = endpointRegistry.ensureEndpoint(roleType, ipPort, workerStatus);
                    } else {
                        endpointRegistry.remove(roleType, ipPort, workerStatus);
                        ep = null;
                    }
                }

                // 2. Notify EP (calibration) — passes both updated status and raw response
                if (ep != null) {
                    ep.onWorkerStatusUpdate(workerStatus, newWorkerStatus);
                }

                // 3. Notify scheduler (cleanup finished requests)
                if (priorityScheduler != null) {
                    priorityScheduler.onWorkerStatusUpdate(newWorkerStatus);
                }

                Long latestFinishedVersion = newWorkerStatus.getLatestFinishedVersion();

                // 4. Advance latestFinishedVersion only after calibrate has processed finished tasks.
                // If this is done outside the version guard, a skipped calibrate (version not
                // advanced) would still consume the incremental version, causing the engine to
                // filter out those finished tasks on the next poll — leaking inflight entries.
                if (latestFinishedVersion != null
                        && latestFinishedVersion > workerStatus.getLatestFinishedTaskVersion().get()) {
                    workerStatus.getLatestFinishedTaskVersion().set(latestFinishedVersion);
                }
            } else {
                workerStatus.refreshStatusHeartbeat(newWorkerStatus.isAlive());
                if (endpointRegistry != null) {
                    if (workerStatus.isAlive()) {
                        ep = endpointRegistry.ensureEndpoint(roleType, ipPort, workerStatus);
                    } else {
                        endpointRegistry.remove(roleType, ipPort, workerStatus);
                        ep = null;
                    }
                }
                if (priorityScheduler != null) {
                    // The running-task payload is a full snapshot even when
                    // status_version is unchanged. Preserve it as liveness
                    // evidence without replaying versioned finished tasks.
                    priorityScheduler.onWorkerStatusHeartbeat(newWorkerStatus);
                }
            }

            engineHealthReporter.reportStatusCheckerSuccess(modelName, workerStatus, ep,
                    Optional.ofNullable(newWorkerStatus.getRunningTaskInfo()).map(Map::size).orElse(0),
                    Optional.ofNullable(newWorkerStatus.getFinishedTaskInfo()).map(Map::size).orElse(0));

            logWorkerStatusUpdate(startTime, workerStatus);
            return true;

        } catch (Throwable e) {
            log("engine worker status check via gRPC exception, msg: " + e.getMessage());
            engineHealthReporter.reportStatusCheckerFail(
                    modelName, BalanceStatusEnum.UNKNOWN_ERROR, roleType);
            return false;
        }
    }

    private void logWorkerStatusUpdate(long startTime, WorkerStatus workerStatus) {
        logger.debug("gRPC Worker Status - {}, role:{}, alive:{}, concurrency:{}, "
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
                workerStatus.getTotalKvCacheTokens().get() - workerStatus.getAvailableKvCacheTokens().get(),
                workerStatus.getRunningTaskList() != null ? workerStatus.getRunningTaskList().values().stream().filter(t -> t.getPhase() != org.flexlb.enums.TaskPhase.RUNNING).count() : 0,
                workerStatus.getRunningTaskList() != null ? workerStatus.getRunningTaskList().size() : 0,
                workerStatus.getStatusVersion(),
                System.nanoTime() / 1000 - startTime);
    }

    private void handleException(Throwable ex) {
        log("gRPC worker status check failed, msg=" + ex.getMessage());
        // Report specific error based on exception type
        if (ex.getMessage() != null && ex.getMessage().toLowerCase().contains(DEADLINE_EXCEEDED_MESSAGE.toLowerCase())) {
            logger.debug("gRPC worker status check timeout, msg={}, ipPort: {}, rt: {}", ex.getMessage(), ipPort, System.nanoTime() / 1000 - createTimeUs);
            engineHealthReporter.reportStatusCheckerFail(
                    modelName, BalanceStatusEnum.WORKER_STATUS_GRPC_TIMEOUT, roleType);
        } else {
            engineHealthReporter.reportStatusCheckerFail(
                    modelName, BalanceStatusEnum.WORKER_SERVICE_UNAVAILABLE, roleType);
        }

    }

    private void log(String msg) {
        logger.debug("[gRPC][{}][{}][{}][{}][{}μs]: {}",
                id,
                site,
                ipPort,
                modelName,
                System.nanoTime() / 1000 - createTimeUs,
                msg);
    }
}
