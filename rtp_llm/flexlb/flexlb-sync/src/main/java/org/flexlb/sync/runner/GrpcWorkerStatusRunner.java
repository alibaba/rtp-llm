package org.flexlb.sync.runner;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.scheduler.FlexlbBatchScheduler;
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
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.TimeUnit;

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
    /** Minimal delay before re-arming the next long-poll (guards against busy-spin). */
    private static final long LONG_POLL_REARM_DELAY_MS = 1;
    private final EndpointRegistry endpointRegistry;
    private final Executor callbackExecutor;
    private final StatusLongPollConfig longPollConfig;

    public GrpcWorkerStatusRunner(String modelName, String ipPort, String site, RoleType roleType, String group,
                                  WorkerStatus workerStatus,
                                  Map<String, WorkerStatus> workerStatusMap,
                                  EngineHealthReporter engineHealthReporter,
                                  EngineGrpcService engineGrpcService,
                                  long syncRequestTimeoutMs,
                                  FlexlbBatchScheduler batchScheduler,
                                  EndpointRegistry endpointRegistry,
                                  Executor callbackExecutor) {
        this(modelName, ipPort, site, roleType, group, workerStatus, workerStatusMap, engineHealthReporter,
                engineGrpcService, syncRequestTimeoutMs, batchScheduler, endpointRegistry, callbackExecutor, null);
    }

    public GrpcWorkerStatusRunner(String modelName, String ipPort, String site, RoleType roleType, String group,
                                  WorkerStatus workerStatus,
                                  Map<String, WorkerStatus> workerStatusMap,
                                  EngineHealthReporter engineHealthReporter,
                                  EngineGrpcService engineGrpcService,
                                  long syncRequestTimeoutMs,
                                  FlexlbBatchScheduler batchScheduler,
                                  EndpointRegistry endpointRegistry,
                                  Executor callbackExecutor,
                                  StatusLongPollConfig longPollConfig) {
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
        this.longPollConfig = longPollConfig;
    }

    @Override
    public void run() {
        boolean asyncInitiated = false;
        try {
            logger.info("GrpcWorkerStatusRunner run for {}", ipPort);
            long startTime = System.nanoTime() / 1000;

            long latestFinishedTaskVersion = workerStatus.getLatestFinishedTaskVersion().get();

            // Long-poll: ask the engine to hold the request until a new completion
            // event, and widen the gRPC deadline to cover the parked time.
            boolean longPoll = longPollConfig != null && longPollConfig.enabled();
            long waitTimeoutMs = longPoll ? longPollConfig.timeoutMs() : 0;
            long requestTimeoutMs = syncRequestTimeoutMs + waitTimeoutMs;

            engineGrpcService.getWorkerStatusAsync(ip, grpcPort, latestFinishedTaskVersion,
                            requestTimeoutMs, roleType, waitTimeoutMs)
                    .thenApply(EngineStatusConverter::convertToWorkerStatusResponse)
                    .whenCompleteAsync((response, ex) -> {
                        boolean rearmed = false;
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
                                // Long-poll chain: launch the next poll as soon as this
                                // response lands instead of waiting for the fixed sync
                                // tick. On failure/stale/dead workers the chain breaks
                                // and the periodic loop (SYNC_STATUS_INTERVAL) resumes
                                // ownership — that is the retry/backoff path.
                                rearmed = rearmLongPoll();
                            }
                        } finally {
                            if (!rearmed) {
                                workerStatus.getStatusCheckInProgress().set(false);
                            }
                        }
                    }, callbackExecutor);
            asyncInitiated = true;
        } finally {
            if (!asyncInitiated) {
                workerStatus.getStatusCheckInProgress().set(false);
            }
        }
    }

    /**
     * Re-arm the next long-poll while keeping statusCheckInProgress=true across
     * the hand-off, so the periodic EngineSyncRunner loop keeps skipping this
     * worker (no duplicate in-flight polls).
     *
     * @return true when the next poll was scheduled and owns the in-progress flag
     */
    private boolean rearmLongPoll() {
        if (longPollConfig == null || !longPollConfig.enabled()) {
            return false;
        }
        if (workerStatusMap != null && workerStatusMap.get(ipPort) != workerStatus) {
            return false; // stale generation: the periodic loop owns the current one
        }
        if (!workerStatus.isAlive()) {
            return false;
        }
        GrpcWorkerStatusRunner next = new GrpcWorkerStatusRunner(
                modelName, ipPort, site, roleType, group, workerStatus, workerStatusMap,
                engineHealthReporter, engineGrpcService, syncRequestTimeoutMs,
                batchScheduler, endpointRegistry, callbackExecutor, longPollConfig);
        try {
            longPollConfig.rearmScheduler().schedule(
                    next, LONG_POLL_REARM_DELAY_MS, TimeUnit.MILLISECONDS);
            return true;
        } catch (RejectedExecutionException e) {
            logger.warn("long-poll re-arm rejected for worker {}, periodic loop resumes", ipPort);
            return false;
        }
    }

    private void handleStatusResponse(WorkerStatusResponse newWorkerStatus, long startTime) {
        try {
            if (newWorkerStatus == null) {
                logger.info("query engine worker status via gRPC, response body is null");
                engineHealthReporter.reportStatusCheckerFail(
                        modelName, BalanceStatusEnum.RESPONSE_NULL, roleType);
                return;
            }
            if (workerStatusMap != null && workerStatusMap.get(ipPort) != workerStatus) {
                logger.info("Ignore stale worker status callback for {}, role: {}", ipPort, roleType);
                return;
            }

            // Only report success worker status check info
            engineHealthReporter.reportStatusCheckRemoteInfo(
                    modelName, newWorkerStatus.getRole().name(), startTime);

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
                    batchScheduler.onWorkerStatusUpdate(newWorkerStatus);
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
            engineHealthReporter.reportStatusCheckerFail(
                    modelName, BalanceStatusEnum.UNKNOWN_ERROR, roleType);
        }
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
}
