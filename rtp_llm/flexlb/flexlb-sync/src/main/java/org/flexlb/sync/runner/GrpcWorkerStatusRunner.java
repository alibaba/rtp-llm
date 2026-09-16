package org.flexlb.sync.runner;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.BalanceStatusEnum;
import org.flexlb.service.grpc.EngineGrpcService;
import org.flexlb.service.grpc.EngineStatusConverter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.WorkerDirectory;
import org.flexlb.util.CommonUtils;
import org.flexlb.util.IdUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;
import java.util.Objects;
import java.util.concurrent.CompletionException;
import java.util.concurrent.Executor;

import static org.flexlb.constant.CommonConstants.DEADLINE_EXCEEDED_MESSAGE;

public class GrpcWorkerStatusRunner implements Runnable {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");
    private static final Runnable NO_STATUS_PROJECTION = () -> { };

    private final String ipPort;
    private final String modelName;
    private final String site;
    private final RoleType roleType;
    private final WorkerStatus workerStatus;
    private final WorkerStatus.PollLease pollLease;
    private final WorkerDirectory workerDirectory;
    private final EngineHealthReporter engineHealthReporter;
    private final EngineGrpcService engineGrpcService;
    private final String ip;
    private final int grpcPort;
    private final long createTimeUs = System.nanoTime() / 1000;
    private final String id = IdUtils.fastUuid();
    private final long syncRequestTimeoutMs;
    private static final int MAX_CONSECUTIVE_FAILURES = 3;
    private final CacheAwareService cacheAwareService;
    private final Executor callbackExecutor;

    public GrpcWorkerStatusRunner(String modelName, String ipPort, String site,
                                  RoleType roleType, String ignoredGroup,
                                  WorkerStatus workerStatus,
                                  WorkerStatus.PollLease pollLease,
                                  WorkerDirectory workerDirectory,
                                  EngineHealthReporter engineHealthReporter,
                                  EngineGrpcService engineGrpcService,
                                  long syncRequestTimeoutMs,
                                  CacheAwareService cacheAwareService,
                                  Executor callbackExecutor) {
        this.ipPort = ipPort;
        String[] split = ipPort.split(":");
        this.ip = split[0];
        this.grpcPort = CommonUtils.toGrpcPort(Integer.parseInt(split[1]));
        this.modelName = modelName;
        this.workerStatus = workerStatus;
        this.pollLease = Objects.requireNonNull(pollLease, "pollLease");
        workerStatus.requireStatusPollLease(pollLease);
        this.workerDirectory = Objects.requireNonNull(
                workerDirectory, "workerDirectory");
        this.site = site;
        this.roleType = roleType;
        this.engineHealthReporter = engineHealthReporter;
        this.engineGrpcService = engineGrpcService;
        this.syncRequestTimeoutMs = syncRequestTimeoutMs;
        this.cacheAwareService = Objects.requireNonNull(
                cacheAwareService, "cacheAwareService");
        this.callbackExecutor = callbackExecutor;
    }

    @Override
    public void run() {
        boolean asyncInitiated = false;
        try {
            logger.debug("GrpcWorkerStatusRunner run for {}", ipPort);
            long startTime = System.nanoTime() / 1000;

            long latestFinishedTaskVersion = workerStatus.appliedStatusCursor()
                    .latestFinishedTaskVersion();

            engineGrpcService.getWorkerStatusAsync(
                            ip, grpcPort, latestFinishedTaskVersion,
                            syncRequestTimeoutMs, roleType)
                    .thenApply(response -> EngineStatusConverter
                            .convertToStatusObservation(workerStatus, response))
                    .handleAsync((observation, failure) -> {
                        try {
                            Throwable cause = unwrapCompletionFailure(failure);
                            if (cause != null) {
                                handleException(cause);
                                recordStatusCheckFailure(cause);
                            } else {
                                handleStatusResponse(observation, startTime);
                            }
                        } catch (Throwable callbackFailure) {
                            logger.error("Worker status callback failed for {}",
                                    ipPort, callbackFailure);
                        }
                        return null;
                    }, callbackExecutor)
                    .whenComplete((ignored, callbackFailure) -> {
                        pollLease.close();
                        if (callbackFailure != null) {
                            logger.error(
                                    "Worker status callback was not scheduled for {}",
                                    ipPort,
                                    unwrapCompletionFailure(callbackFailure));
                        }
                    });
            asyncInitiated = true;
        } finally {
            if (!asyncInitiated) {
                pollLease.close();
            }
        }
    }

    private static Throwable unwrapCompletionFailure(Throwable failure) {
        return failure instanceof CompletionException
                && failure.getCause() != null
                ? failure.getCause() : failure;
    }

    private void handleStatusResponse(
            WorkerStatus.StatusObservation observation,
            long startTime) {
        try {
            if (observation == null) {
                logger.debug("query engine worker status via gRPC, response body is null");
                engineHealthReporter.reportStatusCheckerFail(
                        modelName, BalanceStatusEnum.RESPONSE_NULL, roleType);
                return;
            }
            if (!workerDirectory.isCurrentStatus(
                    roleType, ipPort, workerStatus)) {
                logger.debug("Ignore stale worker status callback for {}, role: {}", ipPort, roleType);
                return;
            }
            WorkerEndpoint ep;
            WorkerStatus.StatusObservation committedObservation;
            Runnable statusProjection = NO_STATUS_PROJECTION;
            Runnable activityProjection = NO_STATUS_PROJECTION;
            EndpointRegistry.DetachedGeneration endpointToRetire = null;
            boolean generationRetiring = false;
            workerStatus.lock.lock();
            try {
                if (!workerDirectory.isCurrentStatus(
                        roleType, ipPort, workerStatus)) {
                    logger.debug(
                            "Ignore stale worker status callback for {}, role: {}",
                            ipPort, roleType);
                    return;
                }
                if (!workerStatus.isActiveGeneration()) {
                    logger.debug(
                            "Ignore callback for retiring WorkerStatus generation {} at {}",
                            workerStatus.getGenerationId(), ipPort);
                    return;
                }
                Long responseVersion = observation.statusVersion();
                if (responseVersion == null || responseVersion <= 0L) {
                    endpointToRetire = workerDirectory.beginRetirement(
                            roleType, ipPort, workerStatus);
                    generationRetiring = true;
                    throw new IllegalArgumentException(
                            "Worker status version must be positive: "
                                    + responseVersion);
                }
                if (observation.role() != roleType) {
                    endpointToRetire = workerDirectory.beginRetirement(
                            roleType, ipPort, workerStatus);
                    generationRetiring = true;
                    throw new IllegalStateException(
                            "Worker status role does not match discovery role: expected="
                                    + roleType + ", actual="
                                    + observation.role());
                }
                committedObservation = observation;

                WorkerStatus.AppliedStatusCursor cursor =
                        workerStatus.appliedStatusCursor();
                if (responseVersion < cursor.statusVersion()) {
                    endpointToRetire = workerDirectory.beginRetirement(
                            roleType, ipPort, workerStatus);
                    generationRetiring = true;
                    throw new IllegalStateException(
                            "Worker status version regressed: committed="
                                    + cursor.statusVersion() + ", response="
                                    + responseVersion);
                }
                workerStatus.recordSuccessfulPoll(observation.alive());
                WorkerEndpoint exactEndpoint = workerDirectory.exactEndpoint(
                        roleType, ipPort, workerStatus);

                if (responseVersion > cursor.statusVersion()) {
                    if (observation.alive()
                            && exactEndpoint == null
                            && cursor.statusVersion() >= 0L) {
                        endpointToRetire = workerDirectory.beginRetirement(
                                roleType, ipPort, workerStatus);
                        generationRetiring = true;
                        throw new IllegalStateException(
                                "Endpoint generation cannot be recreated for committed WorkerStatus "
                                        + ipPort + "#"
                                        + workerStatus.getGenerationId());
                    }
                    WorkerStatus.PreparedStatus prepared =
                            workerStatus.prepareNewStatus(observation);
                    try {
                        EndpointRegistry.EndpointPublication application =
                                applyNewStatusVersion(prepared, exactEndpoint);
                        ep = application.endpoint();
                        statusProjection = application.statusProjection();
                    } catch (Throwable reductionOrPublicationFailure) {
                        endpointToRetire = workerDirectory.beginRetirement(
                                roleType, ipPort, workerStatus);
                        generationRetiring = true;
                        throw propagate(reductionOrPublicationFailure);
                    }
                    if (!observation.alive()) {
                        endpointToRetire = workerDirectory.beginRetirement(
                                roleType, ipPort, workerStatus);
                        generationRetiring = true;
                        ep = null;
                    }
                } else {
                    if (!observation.alive()) {
                        endpointToRetire = workerDirectory.beginRetirement(
                                roleType, ipPort, workerStatus);
                        generationRetiring = true;
                        ep = null;
                    } else if (exactEndpoint != null) {
                        ep = exactEndpoint;
                    } else {
                        endpointToRetire = workerDirectory.beginRetirement(
                                roleType, ipPort, workerStatus);
                        generationRetiring = true;
                        throw new IllegalStateException(
                                "Committed WorkerStatus has no exact endpoint generation: "
                                        + ipPort + "#"
                                        + workerStatus.getGenerationId());
                    }
                }
                if (ep != null && responseVersion == cursor.statusVersion()) {
                    // running_tasks is a full active snapshot even when its
                    // status_version is unchanged. Derive exact endpoint-owned
                    // liveness facts without replaying versioned mutation.
                    activityProjection = ep.observeStatusHeartbeat(
                            workerStatus, observation);
                }
            } finally {
                workerStatus.lock.unlock();
                try {
                    statusProjection.run();
                    activityProjection.run();
                } finally {
                    if (generationRetiring) {
                        workerDirectory.completeRetirement(
                                roleType, ipPort, workerStatus,
                                endpointToRetire, cacheAwareService, logger);
                    }
                }
            }

            reportSuccessfulStatus(
                    committedObservation,
                    startTime,
                    ep);

            logWorkerStatusUpdate(startTime, workerStatus);

        } catch (Throwable e) {
            logger.error("Worker status response handling failed after callback for {}",
                    ipPort, e);
            engineHealthReporter.reportStatusCheckerFail(
                    modelName, BalanceStatusEnum.UNKNOWN_ERROR, roleType);
        }
    }

    private EndpointRegistry.EndpointPublication applyNewStatusVersion(
            WorkerStatus.PreparedStatus prepared,
            WorkerEndpoint exactEndpoint) {
        WorkerStatus.StatusObservation observation = prepared.observation();
        if (exactEndpoint != null) {
            Runnable projection =
                    exactEndpoint.applyPreparedStatus(workerStatus, prepared);
            return new EndpointRegistry.EndpointPublication(
                    observation.alive() ? exactEndpoint : null,
                    projection);
        }

        if (!observation.alive()) {
            workerStatus.publishPreparedStatus(prepared);
            return new EndpointRegistry.EndpointPublication(
                    null, NO_STATUS_PROJECTION);
        }

        // Only the first committed status of this WorkerStatus generation may
        // create an endpoint. Every later endpoint loss retires the entire
        // WorkerStatus generation instead of constructing a second owner.
        return workerDirectory.publishPreparedEndpoint(
                ipPort, workerStatus, prepared);
    }

    private void recordStatusCheckFailure(Throwable failure) {
        EndpointRegistry.DetachedGeneration endpointToRetire = null;
        boolean generationRetiring = false;
        workerStatus.lock.lock();
        try {
            if (!workerDirectory.isCurrentStatus(
                    roleType, ipPort, workerStatus)) {
                return;
            }
            if (!workerStatus.isActiveGeneration()) {
                return;
            }
            WorkerStatus.PollHealth health =
                    workerStatus.recordTransportFailure();
            long failures = health.consecutiveTransportFailures();
            logger.debug("gRPC status check failed, consecutiveFailures={}/{}, msg={}",
                    failures, MAX_CONSECUTIVE_FAILURES, failure.getMessage());
            if (failures < MAX_CONSECUTIVE_FAILURES) {
                return;
            }
            endpointToRetire = workerDirectory.beginRetirement(
                    roleType, ipPort, workerStatus);
            generationRetiring = true;
            if (failures == MAX_CONSECUTIVE_FAILURES) {
                logger.error("worker {} marked dead after {} consecutive gRPC failures",
                        ipPort, failures);
            }
        } finally {
            workerStatus.lock.unlock();
        }
        if (generationRetiring) {
            workerDirectory.completeRetirement(
                    roleType, ipPort, workerStatus,
                    endpointToRetire, cacheAwareService, logger);
        }
    }

    private static RuntimeException propagate(Throwable failure) {
        if (failure instanceof RuntimeException runtimeFailure) {
            return runtimeFailure;
        }
        if (failure instanceof Error error) {
            throw error;
        }
        return new IllegalStateException("Worker status reconciliation failed", failure);
    }

    private void reportSuccessfulStatus(
            WorkerStatus.StatusObservation observation,
            long startTime,
            WorkerEndpoint endpoint) {
        try {
            engineHealthReporter.reportStatusCheckRemoteInfo(
                    modelName, observation.role().name(), startTime);
            engineHealthReporter.reportStatusCheckerSuccess(
                    modelName,
                    workerStatus,
                    endpoint,
                    observation.runningTasks().size(),
                    observation.finishedTasks().size());
        } catch (Throwable telemetryFailure) {
            logger.warn("Worker status telemetry failed after commit for {}: {}",
                    ipPort, telemetryFailure.getMessage());
        }
    }

    private void logWorkerStatusUpdate(long startTime, WorkerStatus workerStatus) {
        WorkerStatus.EngineObservation status =
                workerStatus.committedEngineObservation();
        WorkerStatus.PollHealth health = workerStatus.pollHealth();
        Map<String, WorkerStatus.TaskObservation> runningTasks =
                status.runningTaskList();
        logger.debug("gRPC Worker Status - {}, role:{}, alive:{}, concurrency:{}, "
                        + "step_latency_ms:{}, iterate_count:{}, "
                        + "dp_rank:{}, dp_size:{}, tp_size:{}, "
                        + "avail_kv_tokens:{}, used_kv_tokens:{}, "
                        + "waiting_tasks:{}, running_tasks:{}, "
                        + "version:{}, sync_cost_us:{}",
                ipPort,
                status.role(),
                health.reportedAlive(),
                status.availableConcurrency(),
                status.stepLatencyMs(),
                status.iterateCount(),
                status.dpRank(),
                status.dpSize(),
                status.tpSize(),
                status.availableKvCacheTokens(),
                status.totalKvCacheTokens() - status.availableKvCacheTokens(),
                runningTasks.values().stream()
                        .filter(t -> t.phase()
                                != org.flexlb.enums.TaskPhase.RUNNING).count(),
                runningTasks.size(),
                workerStatus.appliedStatusCursor().statusVersion(),
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
