package org.flexlb.sync.runner;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.cache.service.CacheAwareService;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.EndpointTrace.Stage;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.BalanceStatusEnum;
import org.flexlb.service.grpc.WorkerStatusRpcClient;
import org.flexlb.service.grpc.EngineStatusConverter;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.RoleTypeProtoConverter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.util.IdUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Map;
import java.util.Objects;
import java.util.concurrent.Executor;
import java.util.concurrent.TimeUnit;


public class GrpcWorkerStatusRunner implements Runnable {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");


    private final String ipPort;
    private final String modelName;
    private final RoleType roleType;
    private final WorkerStatus workerStatus;
    private final WorkerStatus.PollLease pollLease;
    private final EndpointRegistry endpointRegistry;
    private final EngineHealthReporter engineHealthReporter;
    private final WorkerStatusRpcClient workerStatusRpcClient;
    private final long createTimeUs = TimeUnit.NANOSECONDS.toMicros(System.nanoTime());
    private final String id = IdUtils.fastUuid();
    private final long syncRequestTimeoutMs;
    private final CacheAwareService cacheAwareService;
    private final Executor callbackExecutor;

    public GrpcWorkerStatusRunner(String modelName, WorkerStatus workerStatus,
                                  WorkerStatus.PollLease pollLease,
                                  EndpointRegistry endpointRegistry,
                                  EngineHealthReporter engineHealthReporter,
                                  WorkerStatusRpcClient workerStatusRpcClient,
                                  long syncRequestTimeoutMs,
                                  CacheAwareService cacheAwareService,
                                  Executor callbackExecutor) {
        this.ipPort = workerStatus.getIpPort();
        this.modelName = modelName;
        this.workerStatus = workerStatus;
        this.pollLease = Objects.requireNonNull(pollLease, "pollLease");
        workerStatus.requireStatusPollLease(pollLease);
        this.endpointRegistry = Objects.requireNonNull(
                endpointRegistry, "endpointRegistry");
        this.roleType = workerStatus.getRole();
        this.engineHealthReporter = engineHealthReporter;
        this.workerStatusRpcClient = workerStatusRpcClient;
        this.syncRequestTimeoutMs = syncRequestTimeoutMs;
        this.cacheAwareService = Objects.requireNonNull(
                cacheAwareService, "cacheAwareService");
        this.callbackExecutor = callbackExecutor;
    }

    @Override
    public void run() {
        pollLease.trace().record(Stage.RUNNER_STARTED);
        boolean asyncInitiated = false;
        try {
            logger.debug("GrpcWorkerStatusRunner run for {}", ipPort);
            long startTime = TimeUnit.NANOSECONDS.toMicros(System.nanoTime());

            long latestFinishedTaskVersion = workerStatus.appliedStatusCursor()
                    .latestFinishedTaskVersion();

            pollLease.trace().recordRpcStarted(syncRequestTimeoutMs);
            workerStatusRpcClient.getWorkerStatusAsync(
                    workerStatus.getIp(), workerStatus.getGrpcPort(), latestFinishedTaskVersion,
                    syncRequestTimeoutMs, roleType).whenComplete(
                            (response, failure) -> onRpcCompletion(response, failure, startTime));
            asyncInitiated = true;
        } catch (RuntimeException | Error failure) {
            pollLease.trace().record(Stage.RPC_INITIATION_FAILED, failure);
            throw failure;
        } finally {
            if (!asyncInitiated) {
                pollLease.close();
            }
        }
    }

    private void onRpcCompletion(EngineRpcService.WorkerStatusPB response, Throwable failure, long startTime) {
        pollLease.trace().record(Stage.RPC_COMPLETED, failure);
        WorkerStatus.PollLease application;
        boolean valid;
        try {
            if (!endpointRegistry.isCurrentStatus(roleType, ipPort, workerStatus)
                    || !workerStatus.isActiveGeneration()) { return; }
            valid = failure == null && response != null && response.getStatusVersion() > 0L
                    && response.getStatusVersion() >= workerStatus.appliedStatusCursor().statusVersion()
                    && RoleTypeProtoConverter.fromWorkerStatus(response) == roleType;
            workerStatus.recordStatusRpcResult(pollLease, valid, valid && response.getAlive());
            application = workerStatus.tryBeginStatusApplication(pollLease);
        } catch (Throwable completionFailure) {
            logger.error("Worker status completion failed for {}", ipPort, completionFailure);
            return;
        } finally {
            pollLease.close();
        }
        if (application == null) { pollLease.trace().record(Stage.RESPONSE_DISCARDED, "APPLICATION_BUSY"); return; }
        pollLease.trace().record(Stage.CALLBACK_READY);
        try {
            callbackExecutor.execute(() -> {
                try (application) {
                    pollLease.trace().record(Stage.CALLBACK_STARTED);
                    if (failure != null) {
                        handleException(failure);
                    } else if (workerStatus.retirementReason() != null) {
                        var retirement = endpointRegistry.beginRequestedRetirement(roleType, ipPort, workerStatus);
                        if (retirement != null) { retirement.completeAsync(cacheAwareService, logger); }
                    } else if (!valid || response.getAlive()) {
                        pollLease.trace().record(Stage.CONVERSION_STARTED);
                        var observation = response == null ? null : EngineStatusConverter.convertToStatusObservation(workerStatus, response);
                        pollLease.trace().record(Stage.CONVERSION_COMPLETED);
                        handleStatusResponse(observation, startTime);
                    }
                } catch (Throwable localFailure) {
                    pollLease.trace().record(Stage.CALLBACK_FAILED, localFailure);
                    logger.error("Worker status application failed for {}", ipPort, localFailure);
                }
            });
        } catch (Throwable rejected) {
            pollLease.trace().record(Stage.CALLBACK_SUBMISSION_FAILED, rejected);
            application.close();
            logger.warn("Worker status application was not scheduled for {}", ipPort, rejected);
        }
    }

    /** The complete status workflow: one generation transaction, followed by lock-free publication. */
    private void handleStatusResponse(WorkerStatus.StatusObservation observation, long startTime) {
        try {
            if (observation == null) {
                engineHealthReporter.reportStatusCheckerFail(modelName, BalanceStatusEnum.RESPONSE_NULL, roleType);
                return;
            }
            StatusApplication applied = null;
            EndpointRegistry.Retirement retirement = null;
            pollLease.trace().record(Stage.STATUS_LOCK_ATTEMPT);
            workerStatus.lock.lock();
            pollLease.trace().record(Stage.STATUS_LOCK_ACQUIRED);
            try {
                if (!endpointRegistry.isCurrentStatus(roleType, ipPort, workerStatus)
                        || !workerStatus.isActiveGeneration()) { return; }
                try {
                    if (workerStatus.retirementReason() != null) {
                        retirement = endpointRegistry.beginRetirement(roleType, ipPort, workerStatus);
                        return;
                    }
                    applied = applyStatusLocked(observation);
                    pollLease.trace().record(Stage.STATUS_COMMITTED);
                } catch (Throwable failure) {
                    if (workerStatus.isActiveGeneration()) {
                        retirement = endpointRegistry.beginRetirement(roleType, ipPort, workerStatus);
                    }
                    throw failure;
                }
            } finally {
                workerStatus.lock.unlock();
                pollLease.trace().record(Stage.STATUS_LOCK_RELEASED);
                try {
                    if (applied != null && applied.endpoint() != null) {
                        applied.endpoint().publishStatus(applied.change());
                    }
                } finally {
                    if (retirement != null) { retirement.completeAsync(cacheAwareService, logger); }
                }
            }
            reportSuccessfulStatus(observation, startTime, applied.endpoint());
            logWorkerStatusUpdate(startTime);
        } catch (Throwable failure) {
            pollLease.trace().record(Stage.STATUS_RESPONSE_FAILED, failure);
            logger.error("Worker status response handling failed after callback for {}", ipPort, failure);
            engineHealthReporter.reportStatusCheckerFail(modelName, BalanceStatusEnum.UNKNOWN_ERROR, roleType);
        }
    }

    private record StatusApplication(WorkerEndpoint endpoint, WorkerEndpoint.StatusChange change) { }

    /** Validate the frame once and commit one exact generation; external effects are returned as data. */
    private StatusApplication applyStatusLocked(WorkerStatus.StatusObservation observation) {
        Long version = observation.statusVersion();
        if (version == null || version <= 0L) {
            throw new IllegalArgumentException("Worker status version must be positive: " + version);
        }
        if (observation.role() != roleType) {
            throw new IllegalStateException("Worker status role does not match discovery role: expected="
                    + roleType + ", actual=" + observation.role());
        }
        WorkerStatus.AppliedStatusCursor cursor = workerStatus.appliedStatusCursor();
        if (version < cursor.statusVersion()) {
            throw new IllegalStateException("Worker status version regressed: committed="
                    + cursor.statusVersion() + ", response=" + version);
        }
        WorkerEndpoint endpoint = endpointRegistry.get(roleType, ipPort, workerStatus);
        if (endpoint == null && cursor.statusVersion() >= 0L) {
            throw new IllegalStateException("Committed WorkerStatus has no exact endpoint generation: "
                    + ipPort + "#" + workerStatus.getGenerationId());
        }
        WorkerEndpoint.StatusChange change = WorkerEndpoint.NO_STATUS_CHANGE;
        if (version > cursor.statusVersion()) {
            WorkerStatus.PreparedStatus prepared = workerStatus.prepareNewStatus(observation);
            if (endpoint != null) { change = endpoint.applyPreparedStatus(workerStatus, prepared); }
            else { endpoint = endpointRegistry.publishPreparedEndpoint(ipPort, workerStatus, prepared); }
        } else if (endpoint != null) {
            // Same-version running_tasks is still a full liveness snapshot; never replay versioned mutations.
            change = endpoint.applyStatusHeartbeat(workerStatus, observation);
        }
        return new StatusApplication(endpoint, change);
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

    private void logWorkerStatusUpdate(long startTime) {
        if (!logger.isDebugEnabled()) { return; }
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
                TimeUnit.NANOSECONDS.toMicros(System.nanoTime()) - startTime);
    }

    private void handleException(Throwable ex) {
        logger.debug("[gRPC][{}][{}][{}][{}][{}μs]: {}", id, workerStatus.getSite(), ipPort, modelName,
                TimeUnit.NANOSECONDS.toMicros(System.nanoTime()) - createTimeUs,
                "gRPC worker status check failed, msg=" + ex.getMessage());
        if (logger.isDebugEnabled()) {
            logger.debug("Worker poll failure: endpoint={} generation={} trace={}",
                    ipPort, workerStatus.getGenerationId(), workerStatus.trace().snapshot().compact());
        }
        boolean timeout = io.grpc.Status.fromThrowable(ex).getCode() == io.grpc.Status.Code.DEADLINE_EXCEEDED;
        if (timeout) {
            logger.debug("gRPC worker status check timeout, msg={}, ipPort: {}, rt: {}", ex.getMessage(), ipPort, TimeUnit.NANOSECONDS.toMicros(System.nanoTime()) - createTimeUs);
        }
        engineHealthReporter.reportStatusCheckerFail(modelName,
                timeout ? BalanceStatusEnum.WORKER_STATUS_GRPC_TIMEOUT : BalanceStatusEnum.WORKER_SERVICE_UNAVAILABLE, roleType);
    }
}
