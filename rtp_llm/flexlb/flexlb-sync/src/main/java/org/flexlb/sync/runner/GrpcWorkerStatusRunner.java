package org.flexlb.sync.runner;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.dao.master.EndpointTrace;
import org.flexlb.dao.master.EndpointTrace.Stage;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.GrpcFailure;
import org.flexlb.engine.grpc.RoleTypeProtoConverter;
import org.flexlb.enums.BalanceStatusEnum;
import org.flexlb.service.grpc.EngineGrpcService;
import org.flexlb.service.grpc.EngineStatusConverter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.WorkerDirectory;
import org.flexlb.util.CommonUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;
import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executor;

public class GrpcWorkerStatusRunner implements Runnable {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");
    private static final Runnable NO_STATUS_PROJECTION = () -> { };

    private final String ipPort;
    private final String modelName;
    private final RoleType roleType;
    private final WorkerStatus workerStatus;
    private final WorkerStatus.PollLease pollLease;
    private final WorkerDirectory workerDirectory;
    private final EngineHealthReporter engineHealthReporter;
    private final EngineGrpcService engineGrpcService;
    private final String ip;
    private final int grpcPort;
    private final long createTimeUs = System.nanoTime() / 1000;
    private final long syncRequestTimeoutMs;
    private final CacheAwareService cacheAwareService;
    private final Executor callbackExecutor;

    public GrpcWorkerStatusRunner(String modelName, String ipPort,
                                  RoleType roleType,
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
        var pollTrace = pollLease.pollTrace();
        try {
            pollTrace.record(Stage.RUNNER_STARTED);
            logger.debug("GrpcWorkerStatusRunner run for {}", ipPort);
            long startTime = System.nanoTime() / 1000;
            long latestFinishedTaskVersion = workerStatus.appliedStatusCursor().latestFinishedTaskVersion();

            pollTrace.recordRpcStarted(syncRequestTimeoutMs);
            CompletableFuture<EngineRpcService.WorkerStatusPB> rpc;
            try {
                rpc = engineGrpcService.getWorkerStatusAsync(
                        ip, grpcPort, latestFinishedTaskVersion, syncRequestTimeoutMs, roleType);
            } catch (RuntimeException | Error failure) {
                workerStatus.recordStatusRpcResult(pollLease, false, false, failure);
                pollTrace.record(Stage.RPC_INITIATION_FAILED, failure);
                reportFailure(failure);
                throw failure;
            }
            rpc.whenComplete((response, failure) -> {
                try {
                    Throwable cause = EndpointTrace.unwrap(failure);
                    if (failure != null) {
                        // Record the RPC outcome before exception diagnostics can fail.
                        workerStatus.recordStatusRpcResult(pollLease, false, false, cause);
                    }
                    pollTrace.record(Stage.RPC_COMPLETED, failure);
                    boolean valid = cause == null && validStatusResponse(response);
                    if (failure == null) {
                        boolean alive = valid && response.getAlive();
                        workerStatus.recordStatusRpcResult(pollLease, valid, alive, null);
                    }
                    if (cause != null) {
                        reportFailure(cause);
                        return;
                    }
                    if (valid && !response.getAlive()) {
                        // Count the worker's health report without replacing usable capacity with an error body.
                        pollTrace.record(Stage.RESPONSE_DISCARDED, "WORKER_REPORTED_NOT_ALIVE");
                        engineHealthReporter.reportStatusCheckerFail(
                                modelName, BalanceStatusEnum.WORKER_STATUS_SERVER_FAILED, roleType);
                        return;
                    }
                    pollTrace.record(Stage.CALLBACK_READY);
                    if (!workerStatus.tryBeginStatusApplication(pollLease)) {
                        pollTrace.record(Stage.RESPONSE_DISCARDED, "APPLICATION_BUSY_OR_GENERATION_UNAVAILABLE");
                        return;
                    }
                } catch (RuntimeException | Error localFailure) {
                    // whenComplete otherwise hides this exception in an unobserved dependent future.
                    pollTrace.record(Stage.CALLBACK_FAILED, localFailure);
                    logger.error("Worker status local completion failed for {}", ipPort, localFailure);
                    reportFailure(localFailure);
                    return;
                } finally {
                    // Heartbeats continue even while an earlier response waits for local application.
                    pollLease.close();
                }
                try {
                    callbackExecutor.execute(() -> {
                        try {
                            pollTrace.record(Stage.CALLBACK_STARTED);
                            if (!workerDirectory.isCurrentStatus(roleType, ipPort, workerStatus)
                                    || !workerStatus.isActiveGeneration() || workerStatus.retirementReason() != null) {
                                pollTrace.record(Stage.RESPONSE_DISCARDED, "GENERATION_UNAVAILABLE");
                                return;
                            }
                            handleStatusResponse(convertStatusResponse(response), startTime);
                        } catch (RuntimeException | Error localFailure) {
                            pollTrace.record(Stage.CALLBACK_FAILED, localFailure);
                            logger.error("Worker status local processing failed for {}", ipPort, localFailure);
                            reportFailure(localFailure);
                        } finally {
                            workerStatus.finishStatusApplication(pollLease);
                            pollTrace.record(Stage.CALLBACK_ENDED);
                        }
                    });
                } catch (RuntimeException | Error rejected) {
                    workerStatus.finishStatusApplication(pollLease);
                    pollTrace.record(Stage.CALLBACK_SUBMISSION_FAILED, rejected);
                    logger.error("Worker status application rejected for {}", ipPort, rejected);
                    reportFailure(rejected);
                }
            });
        } catch (RuntimeException | Error failure) {
            pollTrace.record(Stage.RUNNER_FAILED, failure);
            pollLease.close();
            throw failure;
        }
    }

    private WorkerStatus.StatusObservation convertStatusResponse(EngineRpcService.WorkerStatusPB response) {
        var trace = pollLease.pollTrace();
        trace.record(Stage.CONVERSION_STARTED);
        try {
            var observation = EngineStatusConverter.convertToStatusObservation(workerStatus, response);
            trace.record(Stage.CONVERSION_COMPLETED);
            return observation;
        } catch (RuntimeException | Error localFailure) {
            trace.record(Stage.CONVERSION_FAILED, localFailure);
            throw localFailure;
        }
    }

    private boolean validStatusResponse(EngineRpcService.WorkerStatusPB response) {
        if (response == null || response.getStatusVersion() <= 0L
                || response.getStatusVersion() < workerStatus.appliedStatusCursor().statusVersion()) {
            return false;
        }
        try {
            return RoleTypeProtoConverter.fromWorkerStatus(response) == roleType;
        } catch (IllegalArgumentException invalidRole) {
            // Conversion will classify malformed metadata as a local failure.
            // Never let this scalar probe turn RPC=OK into a transport failure.
            return false;
        }
    }

    private void handleStatusResponse(
            WorkerStatus.StatusObservation observation,
            long startTime) {
        try {
            WorkerEndpoint ep;
            Runnable statusProjection = NO_STATUS_PROJECTION;
            Runnable activityProjection = NO_STATUS_PROJECTION;
            EndpointRegistry.DetachedGeneration endpointToRetire = null;
            boolean generationRetiring = false;
            pollLease.pollTrace().record(Stage.STATUS_LOCK_ATTEMPT);
            workerStatus.lock.lock();
            try {
                pollLease.pollTrace().record(Stage.STATUS_LOCK_ACQUIRED);
                if (!workerDirectory.isCurrentStatus(
                        roleType, ipPort, workerStatus)) {
                    pollLease.pollTrace().record(Stage.RESPONSE_DISCARDED, "GENERATION_CHANGED");
                    logger.debug(
                            "Ignore stale worker status callback for {}, role: {}",
                            ipPort, roleType);
                    return;
                }
                if (!workerStatus.isActiveGeneration() || workerStatus.retirementReason() != null) {
                    pollLease.pollTrace().record(Stage.RESPONSE_DISCARDED, "UNAVAILABLE_OR_RETIRING");
                    logger.debug(
                            "Ignore callback for retiring WorkerStatus generation {} at {}",
                            workerStatus.getGenerationId(), ipPort);
                    return;
                }
                Long responseVersion = observation.statusVersion();
                if (responseVersion == null || responseVersion <= 0L) {
                    endpointToRetire = workerDirectory.beginRetirement(
                            roleType, ipPort, workerStatus, "INVALID_STATUS_VERSION");
                    generationRetiring = true;
                    throw new IllegalArgumentException(
                            "Worker status version must be positive: "
                                    + responseVersion);
                }
                if (observation.role() != roleType) {
                    endpointToRetire = workerDirectory.beginRetirement(
                            roleType, ipPort, workerStatus, "ROLE_MISMATCH");
                    generationRetiring = true;
                    throw new IllegalStateException(
                            "Worker status role does not match discovery role: expected="
                                    + roleType + ", actual="
                                    + observation.role());
                }
                WorkerStatus.AppliedStatusCursor cursor =
                        workerStatus.appliedStatusCursor();
                if (responseVersion < cursor.statusVersion()) {
                    endpointToRetire = workerDirectory.beginRetirement(
                            roleType, ipPort, workerStatus, "STATUS_VERSION_REGRESSED");
                    generationRetiring = true;
                    throw new IllegalStateException(
                            "Worker status version regressed: committed="
                                    + cursor.statusVersion() + ", response="
                                    + responseVersion);
                }
                WorkerEndpoint exactEndpoint = workerDirectory.exactEndpoint(
                        roleType, ipPort, workerStatus);

                if (responseVersion > cursor.statusVersion()) {
                    if (exactEndpoint == null
                            && cursor.statusVersion() >= 0L) {
                        endpointToRetire = workerDirectory.beginRetirement(
                                roleType, ipPort, workerStatus, "ENDPOINT_MISSING");
                        generationRetiring = true;
                        throw new IllegalStateException(
                                "Endpoint generation cannot be recreated for committed WorkerStatus "
                                        + ipPort + "#"
                                        + workerStatus.getGenerationId());
                    }
                    WorkerStatus.PreparedStatus prepared =
                            workerStatus.prepareNewStatus(observation, pollLease);
                    try {
                        EndpointRegistry.EndpointPublication application =
                                applyNewStatusVersion(prepared, exactEndpoint);
                        ep = application.endpoint();
                        statusProjection = application.statusProjection();
                    } catch (Throwable reductionOrPublicationFailure) {
                        pollLease.pollTrace().record(Stage.STATUS_RECONCILIATION_FAILED, reductionOrPublicationFailure);
                        endpointToRetire = workerDirectory.beginRetirement(
                                roleType, ipPort, workerStatus, "STATUS_RECONCILIATION_FAILED");
                        generationRetiring = true;
                        throw propagate(reductionOrPublicationFailure);
                    }
                } else if (exactEndpoint != null) {
                    ep = exactEndpoint;
                } else {
                    endpointToRetire = workerDirectory.beginRetirement(
                            roleType, ipPort, workerStatus, "ENDPOINT_MISSING");
                    generationRetiring = true;
                    throw new IllegalStateException(
                            "Committed WorkerStatus has no exact endpoint generation: "
                                    + ipPort + "#"
                                    + workerStatus.getGenerationId());
                }
                if (ep != null && responseVersion == cursor.statusVersion()) {
                    // running_tasks is a full active snapshot even when its
                    // status_version is unchanged. Derive exact endpoint-owned
                    // liveness facts without replaying versioned mutation.
                    activityProjection = ep.observeStatusHeartbeat(
                            workerStatus, observation);
                }
                pollLease.pollTrace().record(Stage.STATUS_APPLIED);
            } catch (Throwable failure) {
                // Capture before the finally block performs and logs retirement.
                pollLease.pollTrace().record(Stage.STATUS_RESPONSE_FAILED, failure);
                throw failure;
            } finally {
                workerStatus.lock.unlock();
                pollLease.pollTrace().record(Stage.STATUS_LOCK_RELEASED);
                try {
                    statusProjection.run();
                    activityProjection.run();
                } catch (Throwable projectionFailure) {
                    pollLease.pollTrace().record(Stage.STATUS_PROJECTION_FAILED, projectionFailure);
                    throw projectionFailure;
                } finally {
                    if (generationRetiring) {
                        workerDirectory.completeRetirementAsync(
                                roleType, ipPort, workerStatus,
                                endpointToRetire, cacheAwareService, logger);
                    }
                }
            }

            reportSuccessfulStatus(
                    observation,
                    startTime,
                    ep);

            logWorkerStatusUpdate(startTime, workerStatus);

        } catch (Throwable e) {
            pollLease.pollTrace().record(Stage.STATUS_RESPONSE_FAILED, e);
            logger.error("Worker status response handling failed after callback for {}",
                    ipPort, e);
            reportFailure(e);
        }
    }

    private EndpointRegistry.EndpointPublication applyNewStatusVersion(
            WorkerStatus.PreparedStatus prepared,
            WorkerEndpoint exactEndpoint) {
        if (exactEndpoint != null) {
            Runnable projection =
                    exactEndpoint.applyPreparedStatus(workerStatus, prepared);
            return new EndpointRegistry.EndpointPublication(
                    exactEndpoint, projection);
        }

        // Only the first committed status of this WorkerStatus generation may
        // create an endpoint. Every later endpoint loss retires the entire
        // WorkerStatus generation instead of constructing a second owner.
        return workerDirectory.publishPreparedEndpoint(
                ipPort, workerStatus, prepared);
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
        if (!logger.isDebugEnabled()) {
            return;
        }
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

    private void reportFailure(Throwable cause) {
        try {
            GrpcFailure.Analysis failure = GrpcFailure.analyze(cause);
            Stage stage = pollLease.pollTrace().failureStage();
            if (stage == null) {
                stage = Stage.CALLBACK_FAILED;
            }
            boolean rpcFailure = stage == Stage.RPC_COMPLETED;
            BalanceStatusEnum metric = switch (failure.kind()) {
                case LOCAL_MEMORY_EXHAUSTED -> BalanceStatusEnum.MASTER_STATUS_MEMORY_EXHAUSTED;
                case LOCAL_EXECUTOR_REJECTED -> BalanceStatusEnum.MASTER_STATUS_EXECUTOR_REJECTED;
                case LOCAL_BIND_FAILED -> BalanceStatusEnum.MASTER_STATUS_BIND_FAILED;
                case DEADLINE_EXCEEDED -> rpcFailure ? BalanceStatusEnum.WORKER_STATUS_GRPC_TIMEOUT
                        : BalanceStatusEnum.MASTER_STATUS_PROCESSING_FAILED;
                case CONNECT_TIMEOUT, CONNECT_REFUSED, CONNECT_FAILED, NO_ROUTE, CONNECTION_RESET, BROKEN_PIPE,
                        CHANNEL_CLOSED, UNEXPECTED_EOF, IO_INTERRUPTED, ASYNC_CHANNEL_CLOSED, IO_TIMEOUT, SOCKET_FAILURE ->
                        rpcFailure ? BalanceStatusEnum.WORKER_STATUS_CONNECTION_FAILED
                                : BalanceStatusEnum.MASTER_STATUS_PROCESSING_FAILED;
                default -> rpcFailure ? BalanceStatusEnum.WORKER_STATUS_GRPC_FAILED
                        : BalanceStatusEnum.MASTER_STATUS_PROCESSING_FAILED;
            };
            String reason = stage.name();
            if (rpcFailure || failure.confirmedLocalFailure()) {
                reason = failure.kind().name();
            } else if (stage == Stage.STATUS_RESPONSE_FAILED || stage == Stage.STATUS_RECONCILIATION_FAILED) {
                String retirementReason = workerStatus.trace().unavailableReason();
                if (!"UNKNOWN".equals(retirementReason)) {
                    reason = retirementReason;
                }
            }
            engineHealthReporter.reportStatusCheckerFail(modelName, metric, roleType, stage, reason, failure);
            if (rpcFailure) {
                logger.debug("event=worker_status_rpc_failed endpoint={} origin={} reason={} grpcCode={} rootType={} elapsedUs={}",
                        ipPort, failure.confirmedLocalFailure() ? "MASTER" : "UNCONFIRMED",
                        failure.kind(), failure.grpcCode(), failure.rootCause().getClass().getName(),
                        System.nanoTime() / 1000 - createTimeUs, cause);
            }
        } catch (RuntimeException | Error reportingFailure) {
            // The processing failure is already recorded; metrics cannot prevent application cleanup.
            logger.error("Worker status local failure metrics failed for {}", ipPort, reportingFailure);
        }
    }
}
