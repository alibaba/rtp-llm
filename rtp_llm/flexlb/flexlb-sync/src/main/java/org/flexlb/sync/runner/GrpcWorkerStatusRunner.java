package org.flexlb.sync.runner;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.scheduler.FlexlbBatchScheduler;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.dao.master.WorkerStatusResponse;
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
import java.util.concurrent.CompletableFuture;
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
    private final FlexlbBatchScheduler batchScheduler;
    private final String ip;
    private final int grpcPort;
    private final long createTimeUs = System.nanoTime() / 1000;
    private final String id = IdUtils.fastUuid();
    private final long syncRequestTimeoutMs;
    private static final int MAX_CONSECUTIVE_FAILURES = 3;
    private final EndpointRegistry endpointRegistry;
    private final Executor callbackExecutor;

    public GrpcWorkerStatusRunner(String modelName, String ipPort, String site, RoleType roleType, String group,
                                  WorkerStatus workerStatus,
                                  Map<String, WorkerStatus> workerStatusMap,
                                  EngineHealthReporter engineHealthReporter,
                                  EngineGrpcService engineGrpcService,
                                  long syncRequestTimeoutMs,
                                  FlexlbBatchScheduler batchScheduler,
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
        this.batchScheduler = batchScheduler;
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

            engineGrpcService.getWorkerStatusAsync(ip, grpcPort, latestFinishedTaskVersion,
                            syncRequestTimeoutMs, roleType)
                    .thenApply(EngineStatusConverter::convertToWorkerStatusResponse)
                    .whenCompleteAsync((response, ex) -> {
                        try {
                            if (ex != null) {
                                Throwable throwable = ex instanceof CompletionException ? ex.getCause() : ex;
                                handleException(throwable);
                                long failures = workerStatus.getConsecutiveFailures().incrementAndGet();
                                logger.error("gRPC status check failed, consecutiveFailures={}/{}, msg={}",
                                        failures, MAX_CONSECUTIVE_FAILURES, throwable.getMessage());
                                if (failures >= MAX_CONSECUTIVE_FAILURES) {
                                    workerStatus.setAlive(false);
                                    if (endpointRegistry != null) {
                                        endpointRegistry.remove(roleType, ipPort, workerStatus);
                                    }
                                    logger.error("worker {} marked dead after {} consecutive gRPC failures", ipPort, failures);
                                }
                            } else {
                                handleStatusResponse(response, startTime);
                            }
                        } finally {
                            workerStatus.getStatusCheckInProgress().set(false);
                        }
                    }, callbackExecutor);
            asyncInitiated = true;
        } finally {
            if (!asyncInitiated) {
                workerStatus.getStatusCheckInProgress().set(false);
            }
        }
    }

    private void handleStatusResponse(WorkerStatusResponse newWorkerStatus, long startTime) {
        try {
            if (newWorkerStatus == null) {
                logger.info("query engine worker status via gRPC, response body is null");
                engineHealthReporter.reportStatusCheckerFail(modelName, BalanceStatusEnum.RESPONSE_NULL, ip, ipPort, roleType);
                return;
            }
            if (workerStatusMap != null && workerStatusMap.get(ipPort) != workerStatus) {
                logger.info("Ignore stale worker status callback for {}, role: {}", ipPort, roleType);
                return;
            }

            engineHealthReporter.reportStatusCheckRemoteInfo(modelName, ip, ipPort, newWorkerStatus.getRole() != null ? newWorkerStatus.getRole().name() : "UNKNOWN", startTime);

            // Reset consecutive failure counter on successful response
            workerStatus.getConsecutiveFailures().set(0);

            Long responseVersion = newWorkerStatus.getStatusVersion();
            if (responseVersion == 0L) {
                logger.info("workerStatuses.get(ip) is null for gRPC call");
                return;
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
                if (batchScheduler != null) {
                    batchScheduler.onWorkerStatusUpdate(workerStatus, newWorkerStatus);
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
            }

            engineHealthReporter.reportStatusCheckerSuccess(modelName, workerStatus, ep,
                    Optional.ofNullable(newWorkerStatus.getRunningTaskInfo()).map(Map::size).orElse(0),
                    Optional.ofNullable(newWorkerStatus.getFinishedTaskInfo()).map(Map::size).orElse(0));

            logWorkerStatusUpdate(startTime, workerStatus);

        } catch (Throwable e) {
            log("engine worker status check via gRPC exception, msg: " + e.getMessage());
            engineHealthReporter.reportStatusCheckerFail(modelName, BalanceStatusEnum.UNKNOWN_ERROR, ip, ipPort, roleType);
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
            logger.info("gRPC worker status check timeout, msg={}, ipPort: {}, rt: {}", ex.getMessage(), ipPort, System.nanoTime() / 1000 - createTimeUs);
            engineHealthReporter.reportStatusCheckerFail(modelName, BalanceStatusEnum.WORKER_STATUS_GRPC_TIMEOUT, ip, ipPort, roleType);
        } else {
            engineHealthReporter.reportStatusCheckerFail(modelName, BalanceStatusEnum.WORKER_SERVICE_UNAVAILABLE, ip, ipPort, roleType);
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
