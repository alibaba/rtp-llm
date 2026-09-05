package org.flexlb.sync.runner;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.scheduler.PriorityScheduler;
import org.flexlb.cache.domain.CacheHitComparisonResult;
import org.flexlb.cache.match.CacheAwareService;
import org.flexlb.dao.master.CacheHitFeedback;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.TaskStateUpdateResult;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.BalanceStatusEnum;
import org.flexlb.service.grpc.EngineGrpcService;
import org.flexlb.service.grpc.EngineStatusConverter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.util.CommonUtils;
import org.flexlb.util.IdUtils;
import org.flexlb.util.JsonUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.CompletionException;
import java.util.concurrent.Executor;

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
    private final Map<String, WorkerStatus> workerStatusMap;
    private final EngineHealthReporter engineHealthReporter;
    private final EngineGrpcService engineGrpcService;
    private final PriorityScheduler priorityScheduler;
    private final String ip;
    private final int workerStatusPort;
    private final long createTimeUs = System.nanoTime() / 1000;
    private final String id = IdUtils.fastUuid();
    private final long syncRequestTimeoutMs;
    private static final int MAX_CONSECUTIVE_FAILURES = 3;
    private final EndpointRegistry endpointRegistry;
    private final Executor callbackExecutor;
    private final CacheAwareService cacheAwareService;

    public GrpcWorkerStatusRunner(String modelName, String ipPort, String site, RoleType roleType, String group,
                                  WorkerStatus workerStatus,
                                  Map<String, WorkerStatus> workerStatusMap,
                                  EngineHealthReporter engineHealthReporter,
                                  EngineGrpcService engineGrpcService,
                                  long syncRequestTimeoutMs,
                                  PriorityScheduler priorityScheduler,
                                  EndpointRegistry endpointRegistry,
                                  Executor callbackExecutor) {
        this(modelName, ipPort, defaultWorkerStatusPort(ipPort), site, roleType, group, workerStatus, workerStatusMap,
                engineHealthReporter, engineGrpcService, syncRequestTimeoutMs,
                priorityScheduler, endpointRegistry, callbackExecutor, null);
    }

    public GrpcWorkerStatusRunner(String modelName, String ipPort, int workerStatusPort, String site, RoleType roleType,
                                  String group, WorkerStatus workerStatus,
                                  Map<String, WorkerStatus> workerStatusMap,
                                  EngineHealthReporter engineHealthReporter,
                                  EngineGrpcService engineGrpcService,
                                  long syncRequestTimeoutMs,
                                  PriorityScheduler priorityScheduler,
                                  EndpointRegistry endpointRegistry,
                                  Executor callbackExecutor,
                                  CacheAwareService cacheAwareService) {
        this.ipPort = workerStatus.getLogicalIpPort();
        String[] split = ipPort.split(":");
        this.ip = split[0];
        this.workerStatusPort = workerStatusPort;
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
        this.cacheAwareService = cacheAwareService;
    }

    public GrpcWorkerStatusRunner(String modelName, String ipPort, String site, RoleType roleType,
                                  String group, WorkerStatus workerStatus,
                                  Map<String, WorkerStatus> workerStatusMap,
                                  EngineHealthReporter engineHealthReporter,
                                  EngineGrpcService engineGrpcService,
                                  long syncRequestTimeoutMs,
                                  PriorityScheduler priorityScheduler,
                                  EndpointRegistry endpointRegistry,
                                  Executor callbackExecutor,
                                  CacheAwareService cacheAwareService) {
        this(modelName, ipPort, defaultWorkerStatusPort(ipPort), site, roleType, group, workerStatus,
                workerStatusMap, engineHealthReporter, engineGrpcService, syncRequestTimeoutMs,
                priorityScheduler, endpointRegistry, callbackExecutor, cacheAwareService);
    }

    private static int defaultWorkerStatusPort(String ipPort) {
        String[] split = ipPort.split(":");
        return CommonUtils.toGrpcPort(Integer.parseInt(split[1].split("@")[0]));
    }

    @Override
    public void run() {
        boolean asyncInitiated = false;
        try {
            logger.debug("GrpcWorkerStatusRunner run for {}", ipPort);
            long startTime = System.nanoTime() / 1000;

            long latestFinishedTaskVersion = workerStatus.getLatestFinishedTaskVersion().get();

            engineGrpcService.getWorkerStatusAsync(ip, workerStatusPort, latestFinishedTaskVersion,
                            syncRequestTimeoutMs, roleType)
                    .thenApply(EngineStatusConverter::convertToWorkerStatusResponse)
                    .whenCompleteAsync((response, ex) -> {
                        try {
                            if (ex != null) {
                                Throwable throwable = ex instanceof CompletionException ? ex.getCause() : ex;
                                handleException(throwable);
                                long failures = workerStatus.getConsecutiveFailures().incrementAndGet();
                                logger.debug("gRPC status check failed, consecutiveFailures={}/{}, msg={}",
                                        failures, MAX_CONSECUTIVE_FAILURES, throwable.getMessage());
                                if (failures >= MAX_CONSECUTIVE_FAILURES) {
                                    workerStatus.setAlive(false);
                                    if (endpointRegistry != null) {
                                        endpointRegistry.remove(roleType, ipPort, workerStatus);
                                    }
                                    if (failures == MAX_CONSECUTIVE_FAILURES) {
                                        logger.error("worker {} marked dead after {} consecutive gRPC failures", ipPort, failures);
                                    }
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
                workerStatus.setAlive(false);
                logger.debug("query engine worker status via gRPC, response body is null");
                engineHealthReporter.reportStatusCheckerFail(
                        modelName, BalanceStatusEnum.RESPONSE_NULL, workerStatus.getIpIndex(), roleType);
                return;
            }
            if (workerStatusMap != null && workerStatusMap.get(ipPort) != workerStatus) {
                logger.debug("Ignore stale worker status callback for {}, role: {}", ipPort, roleType);
                return;
            }
            // Only report success worker status check info
            engineHealthReporter.reportStatusCheckRemoteInfo(
                    modelName, workerStatus.getIpIndex(), newWorkerStatus.getRole().name(), startTime);

            Long responseVersion = newWorkerStatus.getStatusVersion();
            if (responseVersion == 0L) {
                logger.debug("workerStatuses.get(ip) is null for gRPC call");
                return;
            }
            workerStatus.getConsecutiveFailures().set(0);

            workerStatus.setSite(site);
            workerStatus.setGroup(group);

            long currentVersion = workerStatus.getStatusVersion().get();
            WorkerEndpoint ep = endpointRegistry != null ? endpointRegistry.get(roleType, ipPort) : null;
            boolean versionAdvanced = currentVersion < responseVersion;

            Map<String, TaskInfo> waitingTaskInfo = newWorkerStatus.getWaitingTaskInfo();
            Map<String, TaskInfo> runningTaskInfo = newWorkerStatus.getRunningTaskInfo();
            Map<String, TaskInfo> lifecycleRunningTaskInfo =
                    runningOnly(runningTaskInfo);
            Map<String, TaskInfo> finishedTaskInfo = newWorkerStatus.getFinishedTaskInfo();

            // Task lifecycle is incremental and may advance even when the coarse
            // status version is unchanged. Reconcile it on every successful poll.
            workerStatus.updateFromResponse(newWorkerStatus);
            workerStatus.setWaitingTaskList(waitingTaskInfo);
            TaskStateUpdateResult taskStateUpdateResult = workerStatus.updateTaskStates(
                    waitingTaskInfo, lifecycleRunningTaskInfo, finishedTaskInfo);
            handleTaskStateUpdateResult(taskStateUpdateResult);
            reportFinishedPrefillTasks(finishedTaskInfo);
            workerStatus.updateRunningQueueTime();

            if (versionAdvanced) {
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

            } else {
                if (endpointRegistry != null) {
                    if (workerStatus.isAlive()) {
                        ep = endpointRegistry.ensureEndpoint(roleType, ipPort, workerStatus);
                    } else {
                        endpointRegistry.remove(roleType, ipPort, workerStatus);
                        ep = null;
                    }
                }
            }

            boolean hasFinishedTasks = finishedTaskInfo != null && !finishedTaskInfo.isEmpty();
            if (priorityScheduler != null && (versionAdvanced || hasFinishedTasks)) {
                priorityScheduler.onWorkerStatusUpdate(newWorkerStatus);
            }

            // Advance the incremental cursor only after both local lifecycle
            // reconciliation and scheduler cleanup consumed the finished tasks.
            Long latestFinishedVersion = newWorkerStatus.getLatestFinishedVersion();
            if (latestFinishedVersion != null
                    && latestFinishedVersion > workerStatus.getLatestFinishedTaskVersion().get()) {
                workerStatus.getLatestFinishedTaskVersion().set(latestFinishedVersion);
            }

            engineHealthReporter.reportStatusCheckerSuccess(modelName, workerStatus, ep,
                    Optional.ofNullable(newWorkerStatus.getRunningTaskInfo()).map(Map::size).orElse(0),
                    Optional.ofNullable(newWorkerStatus.getFinishedTaskInfo()).map(Map::size).orElse(0));

            logWorkerStatusUpdate(startTime, workerStatus);

        } catch (Throwable e) {
            log("engine worker status check via gRPC exception, msg: " + e.getMessage());
            engineHealthReporter.reportStatusCheckerFail(
                    modelName, BalanceStatusEnum.UNKNOWN_ERROR, workerStatus.getIpIndex(), roleType);
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

    private void handleTaskStateUpdateResult(TaskStateUpdateResult updateResult) {
        for (long latencyMs : updateResult.decisionToWaitingObservedLatenciesMs()) {
            engineHealthReporter.reportFlexlbObservedMasterDecisionToWaitingConfirmationLatency(
                    modelName, workerStatus.getIpIndex(), roleType.getCode(), group, latencyMs);
        }
        for (long latencyMs : updateResult.waitingToRunningObservedLatenciesMs()) {
            engineHealthReporter.reportFlexlbObservedWaitingToRunningLatency(
                    modelName, workerStatus.getIpIndex(), roleType.getCode(), group, latencyMs);
        }
        for (long latencyMs : updateResult.engineWaitingToRunningLatenciesMs()) {
            engineHealthReporter.reportEngineObservedWaitingToRunningLatency(
                    modelName, workerStatus.getIpIndex(), roleType.getCode(), group, latencyMs);
        }
        for (long latencyMs : updateResult.engineReceivedToWaitingLatenciesMs()) {
            engineHealthReporter.reportEngineObservedReceivedToWaitingLatency(
                    modelName, workerStatus.getIpIndex(), roleType.getCode(), group, latencyMs);
        }
        if (cacheAwareService == null) {
            return;
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
                    modelName, workerStatus.getIpIndex(), roleType.getCode(), group, task);
            Map<String, Object> event = new LinkedHashMap<>();
            event.put("event", "prefill_worker_status");
            event.put("requestId", task.getRequestId());
            event.put("model", modelName);
            event.put("workerIp", ip);
            event.put("workerPort", workerStatusPort);
            event.put("engineIndex", workerStatus.getEngineIndex());
            event.put("logicalWorker", workerStatus.getLogicalIpPort());
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
            event.put("prefillNonfinalChunkTokensMin",
                    task.getPrefillNonfinalChunkTokensMin());
            event.put("prefillNonfinalChunkTokensMax",
                    task.getPrefillNonfinalChunkTokensMax());
            event.put("inputQueueWaitMs",
                    duration(task.getInputQueueDrainTimeMs(), task.getInputQueueEnqueueTimeMs()));
            long schedulerToRunningMs =
                    duration(task.getRunningEnteredTimeMs(), task.getWaitingEnteredTimeMs());
            event.put("schedulerToRunningMs", schedulerToRunningMs);
            event.put("schedulerWaitMs", schedulerToRunningMs < 0
                    ? -1 : Math.max(0, schedulerToRunningMs - task.getRemoteKvWaitMs()));
            event.put("runningToFirstTokenMs",
                    duration(task.getFirstTokenTimeMs(), task.getRunningEnteredTimeMs()));
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

    private static Map<String, TaskInfo> runningOnly(Map<String, TaskInfo> tasks) {
        if (tasks == null) {
            return null;
        }
        Map<String, TaskInfo> running = new HashMap<>();
        tasks.forEach((requestId, task) -> {
            if (task != null && task.getPhase() == org.flexlb.enums.TaskPhase.RUNNING) {
                running.put(requestId, task);
            }
        });
        return running;
    }

    private void handleException(Throwable ex) {
        log("gRPC worker status check failed, msg=" + ex.getMessage());
        // Report specific error based on exception type
        if (ex.getMessage() != null && ex.getMessage().toLowerCase().contains(DEADLINE_EXCEEDED_MESSAGE.toLowerCase())) {
            logger.debug("gRPC worker status check timeout, msg={}, ipPort: {}, rt: {}", ex.getMessage(), ipPort, System.nanoTime() / 1000 - createTimeUs);
            engineHealthReporter.reportStatusCheckerFail(
                    modelName, BalanceStatusEnum.WORKER_STATUS_GRPC_TIMEOUT, workerStatus.getIpIndex(), roleType);
        } else {
            engineHealthReporter.reportStatusCheckerFail(
                    modelName, BalanceStatusEnum.WORKER_SERVICE_UNAVAILABLE, workerStatus.getIpIndex(), roleType);
        }
        workerStatus.refreshStatusHeartbeat(workerStatus.isAlive());
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
