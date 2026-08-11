package org.flexlb.sync.runner;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.EndpointRetireCause;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.scheduler.FlexlbBatchScheduler;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerLifecycleState;
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
            logger.info("GrpcWorkerStatusRunner run for {}", ipPort);
            long startTime = System.nanoTime() / 1000;

            long latestFinishedTaskVersion = workerStatus.getLatestFinishedTaskVersion().get();

            engineGrpcService.getWorkerStatusAsync(ip, grpcPort, latestFinishedTaskVersion,
                            syncRequestTimeoutMs, roleType)
                    .thenApply(EngineStatusConverter::convertToWorkerStatusResponse)
                    .whenCompleteAsync((response, ex) -> {
                        try {
                            if (ex != null) {
                                Throwable throwable = ex instanceof CompletionException ? ex.getCause() : ex;
                                handleStatusFailure(throwable);
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
        if (!validateStatusResponse(newWorkerStatus)) {
            recordInvalidStatusFailure(newWorkerStatus);
            return;
        }

        WorkerEndpoint ep = null;
        boolean retireGeneration = false;
        workerStatus.lock.lock();
        try {
            if (!isCurrentProbeableGeneration()) {
                logger.info("Ignore stale worker status callback for {}, role: {}", ipPort, roleType);
                return;
            }

            reportSafely("status-check-remote-info", () ->
                    engineHealthReporter.reportStatusCheckRemoteInfo(
                            modelName, newWorkerStatus.getRole().name(), startTime));

            workerStatus.setSite(site);
            workerStatus.setGroup(group);

            Long responseVersion = newWorkerStatus.getStatusVersion();
            long currentVersion = workerStatus.getStatusVersion().get();
            boolean versionAdvanced = currentVersion < responseVersion;
            if (versionAdvanced) {
                workerStatus.updateFromResponse(newWorkerStatus);
            }

            // A syntactically valid RPC is a successful heartbeat even when the
            // Engine explicitly reports itself unavailable. It resets transport
            // failures, but only an alive response may publish a generation.
            workerStatus.recordStatusSuccess();

            if (!newWorkerStatus.isAlive()) {
                retireGeneration = workerStatus.isReady() && workerStatus.tryBeginRetirement();
            } else {
                boolean firstPublication =
                        workerStatus.getLifecycleState() == WorkerLifecycleState.PROBING;
                boolean snapshotApplied = !versionAdvanced;
                if (firstPublication) {
                    validatePublishableEndpointStatus();
                    if (endpointRegistry != null) {
                        ep = endpointRegistry.publishValidatedEndpoint(
                                roleType, ipPort, workerStatus, newWorkerStatus);
                        snapshotApplied = ep != null;
                    } else {
                        // Health-only unit-test mode; production always publishes
                        // through EndpointRegistry's generation slot.
                        workerStatus.tryMarkReady();
                        snapshotApplied = true;
                    }
                } else {
                    ep = endpointRegistry != null ? endpointRegistry.get(roleType, ipPort) : null;
                    if (versionAdvanced && ep != null) {
                        snapshotApplied = ep.tryOnWorkerStatusUpdate(workerStatus, newWorkerStatus);
                    }
                }

                // A publication can be retried after an earlier retirement barrier
                // rejected it. In that case statusVersion is already equal, but the
                // first endpoint still needs scheduler reconciliation and cursor commit.
                if (snapshotApplied && (versionAdvanced || firstPublication)) {
                    if (batchScheduler != null) {
                        batchScheduler.onWorkerStatusUpdate(newWorkerStatus);
                    }

                    Long latestFinishedVersion = newWorkerStatus.getLatestFinishedVersion();
                    if (latestFinishedVersion != null
                            && latestFinishedVersion > workerStatus.getLatestFinishedTaskVersion().get()) {
                        workerStatus.getLatestFinishedTaskVersion().set(latestFinishedVersion);
                    }
                }
                if (!snapshotApplied) {
                    logger.warn("Validated worker snapshot was not applied; generation stays fenced: ipPort={}, role={}, state={}",
                            ipPort, roleType, workerStatus.getLifecycleState());
                }
            }

            WorkerEndpoint reportedEndpoint = ep;
            reportSafely("status-check-success", () ->
                    engineHealthReporter.reportStatusCheckerSuccess(
                            modelName, workerStatus, reportedEndpoint,
                            Optional.ofNullable(newWorkerStatus.getRunningTaskInfo())
                                    .map(Map::size).orElse(0),
                            Optional.ofNullable(newWorkerStatus.getFinishedTaskInfo())
                                    .map(Map::size).orElse(0)));
            logWorkerStatusUpdate(startTime, workerStatus);
        } catch (Throwable e) {
            log("engine worker status check via gRPC exception, msg: " + e.getMessage());
            reportSafely("status-check-unknown-error", () ->
                    engineHealthReporter.reportStatusCheckerFail(
                            modelName, BalanceStatusEnum.UNKNOWN_ERROR, roleType));
        } finally {
            workerStatus.lock.unlock();
        }

        if (retireGeneration) {
            retireCurrentGeneration("engine-reported-not-alive");
        }
    }

    private boolean validateStatusResponse(WorkerStatusResponse response) {
        if (response == null) {
            return false;
        }
        Long version = response.getStatusVersion();
        return response.getRole() == roleType && version != null && version > 0L;
    }

    private void recordInvalidStatusFailure(WorkerStatusResponse response) {
        if (!isCurrentProbeableGeneration()) {
            logger.info("Ignore invalid status from stale worker generation: {}, role: {}", ipPort, roleType);
            return;
        }
        BalanceStatusEnum status = response == null
                ? BalanceStatusEnum.RESPONSE_NULL : BalanceStatusEnum.UNKNOWN_ERROR;
        logger.warn("Invalid worker status response for {}, expectedRole={}, actualRole={}, version={}",
                ipPort, roleType, response == null ? null : response.getRole(),
                response == null ? null : response.getStatusVersion());
        reportSafely("invalid-status", () ->
                engineHealthReporter.reportStatusCheckerFail(modelName, status, roleType));
        recordFailureAndMaybeRetire("invalid-status-response", null);
    }

    private void validatePublishableEndpointStatus() {
        if ((roleType == RoleType.PREFILL || roleType == RoleType.PDFUSION)
                && workerStatus.getDpSize() > 1L) {
            throw new UnsupportedOperationException(String.format(
                    "%s DP group endpoint not yet supported: model=%s, ipPort=%s, dp_size=%d",
                    roleType, modelName, ipPort, workerStatus.getDpSize()));
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

    private void handleStatusFailure(Throwable throwable) {
        if (!isCurrentProbeableGeneration()) {
            logger.info("Ignore stale worker failure callback for {}, role: {}", ipPort, roleType);
            return;
        }
        handleException(throwable);
        recordFailureAndMaybeRetire("grpc-status-failure", throwable);
    }

    private void recordFailureAndMaybeRetire(String cause, Throwable throwable) {
        boolean retireGeneration = false;
        long failures = -1L;
        workerStatus.lock.lock();
        try {
            if (!isCurrentProbeableGeneration()) {
                logger.info("Ignore stale worker failure callback for {}, role: {}", ipPort, roleType);
                return;
            }
            failures = workerStatus.recordStatusFailure();
            logger.error("gRPC status check failed, consecutiveFailures={}/{}, msg={}",
                    failures, MAX_CONSECUTIVE_FAILURES,
                    throwable == null ? cause : throwable.getMessage());
            if (workerStatus.isReady()
                    && failures >= MAX_CONSECUTIVE_FAILURES
                    && workerStatus.tryBeginRetirement()) {
                retireGeneration = true;
            }
        } finally {
            workerStatus.lock.unlock();
        }

        if (retireGeneration) {
            logger.error("worker {} generation retiring after {} consecutive gRPC failures", ipPort, failures);
            retireCurrentGeneration(cause);
        }
    }

    private boolean isCurrentProbeableGeneration() {
        return workerStatus.isProbeable()
                && (workerStatusMap == null || workerStatusMap.get(ipPort) == workerStatus);
    }

    /**
     * Complete retirement for the generation already fenced as RETIRING through
     * EndpointRegistry's publication barrier and unified settlement path.
     */
    private void retireCurrentGeneration(String cause) {
        boolean endpointRemoved = false;
        boolean statusRemoved = false;
        try {
            endpointRemoved = endpointRegistry != null
                    && endpointRegistry.retire(roleType, ipPort, workerStatus,
                    EndpointRetireCause.HEALTH_CHECK_FAILED);
        } finally {
            if (workerStatusMap != null) {
                statusRemoved = workerStatusMap.remove(ipPort, workerStatus);
            }
            workerStatus.markClosed();
        }
        logger.info("Retired worker generation: ipPort={}, role={}, cause={}, statusRemoved={}, endpointRemoved={}",
                ipPort, roleType, cause, statusRemoved, endpointRemoved);
    }

    private void handleException(Throwable ex) {
        log("gRPC worker status check failed, msg=" + ex.getMessage());
        // Report specific error based on exception type
        if (ex.getMessage() != null && ex.getMessage().toLowerCase().contains(DEADLINE_EXCEEDED_MESSAGE.toLowerCase())) {
            logger.info("gRPC worker status check timeout, msg={}, ipPort: {}, rt: {}", ex.getMessage(), ipPort, System.nanoTime() / 1000 - createTimeUs);
            reportSafely("status-check-timeout", () ->
                    engineHealthReporter.reportStatusCheckerFail(
                            modelName, BalanceStatusEnum.WORKER_STATUS_GRPC_TIMEOUT, roleType));
        } else {
            reportSafely("status-check-unavailable", () ->
                    engineHealthReporter.reportStatusCheckerFail(
                            modelName, BalanceStatusEnum.WORKER_SERVICE_UNAVAILABLE, roleType));
        }
    }

    /** Monitoring must never participate in the worker lifecycle transaction. */
    private void reportSafely(String operation, Runnable report) {
        try {
            report.run();
        } catch (Throwable reportingFailure) {
            logger.warn("Ignore worker health telemetry failure: operation={}, ipPort={}, role={}",
                    operation, ipPort, roleType, reportingFailure);
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
