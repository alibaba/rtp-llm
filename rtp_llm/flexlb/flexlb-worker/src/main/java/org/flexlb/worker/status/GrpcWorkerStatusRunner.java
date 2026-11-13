package org.flexlb.worker.status;

import org.flexlb.worker.cache.WorkerCaches;
import org.flexlb.worker.endpoint.EncoderEndpoint;
import org.flexlb.worker.endpoint.EndpointRegistry;
import org.flexlb.worker.endpoint.WorkerEndpoint;
import org.flexlb.worker.rpc.RoleTypeProtoConverter;
import org.flexlb.worker.rpc.WorkerStatusRpcClient;
import org.flexlb.worker.rpc.protocol.EngineRpcService;
import org.flexlb.worker.status.EndpointTrace.Stage;

import java.util.Map;
import java.util.Objects;
import java.util.concurrent.Executor;
import java.util.concurrent.TimeUnit;

/**
 * <p>One poll lease starts a status RPC for an exact discovered generation. The response is
 * validated and its heartbeat evidence is recorded before status application is handed to the
 * callback executor. Stale generations cannot publish new endpoint state.
 */
public class GrpcWorkerStatusRunner extends AbstractStatusRunner {
    public GrpcWorkerStatusRunner(
            String modelName,
            WorkerStatus workerStatus,
            WorkerStatus.PollLease pollLease,
            EndpointRegistry endpointRegistry,
            ServiceMetricsReporter serviceMetricsReporter,
            WorkerStatusRpcClient workerStatusRpcClient,
            long syncRequestTimeoutMs,
            WorkerCaches cacheAwareService,
            Executor callbackExecutor) {
        super(
                modelName,
                workerStatus,
                pollLease,
                endpointRegistry,
                serviceMetricsReporter,
                workerStatusRpcClient,
                cacheAwareService,
                syncRequestTimeoutMs,
                callbackExecutor);
        workerStatus.requireStatusPollLease(pollLease);
        Objects.requireNonNull(cacheAwareService, "cacheAwareService");
    }

    @Override
    protected void startRpc() {
        logger.debug("GrpcWorkerStatusRunner run for {}", ipPort);
        long startTime = TimeUnit.NANOSECONDS.toMicros(System.nanoTime());

        long latestFinishedTaskVersion =
                workerStatus.appliedStatusCursor().latestFinishedTaskVersion();

        pollLease.trace().recordRpcStarted(requestTimeoutMs);
        workerStatusRpcClient
                .getWorkerStatusAsync(
                        workerStatus.getIp(),
                        workerStatus.getWorkerStatusPort(),
                        latestFinishedTaskVersion,
                        requestTimeoutMs,
                        roleType)
                .whenComplete((response, failure) -> onRpcCompletion(response, failure, startTime));
    }

    @Override
    protected void reportRpcFailure(boolean timeout) {
        serviceMetricsReporter.reportStatusCheckerFail(
                timeout
                        ? BalanceStatusEnum.WORKER_STATUS_GRPC_TIMEOUT
                        : BalanceStatusEnum.WORKER_SERVICE_UNAVAILABLE,
                workerStatus.getMetricIpPort(),
                roleType);
    }

    private void onRpcCompletion(
            EngineRpcService.WorkerStatusPB response, Throwable failure, long startTime) {
        pollLease.trace().record(Stage.RPC_COMPLETED, failure);
        WorkerStatus.PollLease application;
        boolean valid;
        try {
            if (!endpointRegistry.isCurrentStatus(roleType, ipPort, workerStatus)
                    || !workerStatus.isActiveGeneration()) {
                return;
            }
            valid =
                    failure == null
                            && response != null
                            && response.getStatusVersion() > 0L
                            && response.getStatusVersion()
                                    >= workerStatus.appliedStatusCursor().statusVersion()
                            && RoleTypeProtoConverter.fromWorkerStatus(response) == roleType;
            workerStatus.recordStatusRpcResult(pollLease, valid, valid && response.getAlive());
            application = workerStatus.tryBeginStatusApplication(pollLease);
        } catch (Throwable completionFailure) {
            logger.error("Worker status completion failed for {}", ipPort, completionFailure);
            return;
        } finally {
            pollLease.close();
        }
        if (application == null) {
            pollLease.trace().record(Stage.RESPONSE_DISCARDED, "APPLICATION_BUSY");
            return;
        }
        pollLease.trace().record(Stage.CALLBACK_READY);
        try {
            callbackExecutor.execute(
                    () -> {
                        try (application) {
                            pollLease.trace().record(Stage.CALLBACK_STARTED);
                            if (failure != null) {
                                handleException(failure);
                            } else if (workerStatus.closeReason() != null) {
                                endpointRegistry.closeUnavailableWorker(
                                        roleType, ipPort, workerStatus);
                            } else if (!valid || response.getAlive()) {
                                pollLease.trace().record(Stage.CONVERSION_STARTED);
                                var observation =
                                        response == null
                                                ? null
                                                : WorkerStatusRpcClient.convertToStatusObservation(
                                                        workerStatus, response);
                                pollLease.trace().record(Stage.CONVERSION_COMPLETED);
                                handleStatusResponse(observation, startTime);
                            }
                        } catch (Throwable localFailure) {
                            pollLease.trace().record(Stage.CALLBACK_FAILED, localFailure);
                            logger.error(
                                    "Worker status application failed for {}",
                                    ipPort,
                                    localFailure);
                        }
                    });
        } catch (Throwable rejected) {
            pollLease.trace().record(Stage.CALLBACK_SUBMISSION_FAILED, rejected);
            application.close();
            logger.warn("Worker status application was not scheduled for {}", ipPort, rejected);
        }
    }

    /**
     * The complete status workflow: one generation transaction, followed by lock-free publication.
     */
    private void handleStatusResponse(WorkerStatus.StatusObservation observation, long startTime) {
        try {
            if (observation == null) {
                serviceMetricsReporter.reportStatusCheckerFail(
                        BalanceStatusEnum.RESPONSE_NULL, workerStatus.getMetricIpPort(), roleType);
                return;
            }
            StatusApplication applied = null;
            pollLease.trace().record(Stage.STATUS_LOCK_ATTEMPT);
            workerStatus.lock.lock();
            pollLease.trace().record(Stage.STATUS_LOCK_ACQUIRED);
            try {
                if (!endpointRegistry.isCurrentStatus(roleType, ipPort, workerStatus)
                        || !workerStatus.isActiveGeneration()) {
                    return;
                }
                try {
                    if (workerStatus.closeReason() != null) {
                        endpointRegistry.closeWorker(roleType, ipPort, workerStatus);
                        return;
                    }
                    applied = applyStatusLocked(observation);
                    pollLease.trace().record(Stage.STATUS_COMMITTED);
                } catch (Throwable failure) {
                    if (workerStatus.isActiveGeneration()) {
                        endpointRegistry.closeWorker(roleType, ipPort, workerStatus);
                    }
                    throw failure;
                }
            } finally {
                workerStatus.lock.unlock();
                pollLease.trace().record(Stage.STATUS_LOCK_RELEASED);
                if (applied != null && applied.endpoint() != null) {
                    applied.endpoint().publishStatus(applied.change());
                }
            }
            reportSuccessfulStatus(observation, startTime, applied);
            observeCacheFeedback(observation);
            logWorkerStatusUpdate(startTime);
        } catch (Throwable failure) {
            pollLease.trace().record(Stage.STATUS_RESPONSE_FAILED, failure);
            logger.error(
                    "Worker status response handling failed after callback for {}",
                    ipPort,
                    failure);
            serviceMetricsReporter.reportStatusCheckerFail(
                    BalanceStatusEnum.UNKNOWN_ERROR, workerStatus.getMetricIpPort(), roleType);
        }
    }

    /**
     * Validate the frame once and commit one exact generation; external effects are returned as data.
     */
    private StatusApplication applyStatusLocked(WorkerStatus.StatusObservation observation) {
        Long version = observation.statusVersion();
        if (version == null || version <= 0L) {
            throw new IllegalArgumentException(
                    "Worker status version must be positive: " + version);
        }
        if (observation.role() != roleType) {
            throw new IllegalStateException(
                    "Worker status role does not match discovery role: expected="
                            + roleType
                            + ", actual="
                            + observation.role());
        }
        WorkerStatus.AppliedStatusCursor cursor = workerStatus.appliedStatusCursor();
        if (version < cursor.statusVersion()) {
            throw new IllegalStateException(
                    "Worker status version regressed: committed="
                            + cursor.statusVersion()
                            + ", response="
                            + version);
        }
        var previousStep = workerStatus.committedEngineObservation().lastStepMetrics();
        var update =
                endpointRegistry.updateEndpoint(
                        ipPort, workerStatus, workerStatus.prepareStatus(observation));
        return new StatusApplication(
                update.endpoint(),
                update.change(),
                version > cursor.statusVersion(),
                observation.latestFinishedVersion() != null
                        && observation.latestFinishedVersion() > cursor.latestFinishedTaskVersion(),
                previousStep);
    }

    private void reportSuccessfulStatus(
            WorkerStatus.StatusObservation observation, long startTime, StatusApplication applied) {
        var endpoint = applied.endpoint();
        try {
            serviceMetricsReporter.reportStatusCheckRemoteInfo(
                    modelName, observation.role().name(), startTime);
            serviceMetricsReporter.reportStatusCheckerSuccess(
                    workerStatus,
                    endpoint == null ? java.util.OptionalLong.empty() : endpoint.tryGetLoadMetric(),
                    observation.activeTasks().size(),
                    observation.finishedTasks().size());
            if (endpoint instanceof EncoderEndpoint encoder) {
                var load = encoder.selectionSnapshot();
                long total = load.requestCount();
                var engine = observation.engine();
                long pending =
                        Math.max(
                                0,
                                total
                                        - Math.max(0, engine.runningQueryLen())
                                        - Math.max(0, engine.waitingQueryLen()));
                serviceMetricsReporter.reportEncoderStatus(
                        workerStatus.getMetricIpPort(), pending, total, load.uncachedTokens());
            }
            var step = observation.engine().lastStepMetrics();
            if (applied.newStatus()
                    && step != null
                    && (applied.previousStep() == null
                            || step.stepId() != applied.previousStep().stepId()))
                serviceMetricsReporter.reportWorkerStepMetrics(workerStatus, step);
            if (applied.newFinished() && observation.role().supportsPrefill()) {
                for (var task : observation.finishedTasks().values()) {
                    if (task.telemetry() != null && task.telemetry().firstTokenTimeMs() > 0)
                        serviceMetricsReporter.reportPrefillWorkerStatusTask(
                                workerStatus.getMetricIpPort(),
                                observation.role().name(),
                                workerStatus.getGroup(),
                                task.telemetry());
                }
            }
        } catch (Throwable telemetryFailure) {
            logger.warn(
                    "Worker status telemetry failed after commit for {}: {}",
                    ipPort,
                    telemetryFailure.getMessage());
        }
    }

    private void observeCacheFeedback(WorkerStatus.StatusObservation observation) {
        try {
            for (var result :
                    cacheAwareService.observeCacheHitFeedback(workerStatus, observation)) {
                result.whenComplete(
                        (comparison, error) -> {
                            if (error != null || comparison == null) {
                                logger.warn("Cache comparison failed at {}", ipPort, error);
                                return;
                            }
                            try {
                                org.slf4j.LoggerFactory.getLogger("pvLogger")
                                        .info(
                                                org.flexlb.util.JsonUtils.toStringOrEmpty(
                                                        comparison));
                                serviceMetricsReporter.reportCacheHitComparisonMetrics(
                                        workerStatus, comparison);
                            } catch (RuntimeException failure) {
                                logger.warn(
                                        "Cache comparison telemetry failed at {}", ipPort, failure);
                            }
                        });
            }
        } catch (RuntimeException failure) {
            logger.warn("Cache feedback observation failed at {}", ipPort, failure);
        }
    }

    private void logWorkerStatusUpdate(long startTime) {
        if (!logger.isDebugEnabled()) {
            return;
        }
        WorkerStatus.EngineObservation status = workerStatus.committedEngineObservation();
        WorkerStatus.PollHealth health = workerStatus.pollHealth();
        Map<String, WorkerStatus.TaskObservation> runningTasks = status.runningTaskList();
        logger.debug(
                "gRPC Worker Status - {}, role:{}, alive:{}, concurrency:{}, "
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
                runningTasks.values().stream().filter(t -> t.phase() != TaskPhase.RUNNING).count(),
                runningTasks.size(),
                workerStatus.appliedStatusCursor().statusVersion(),
                TimeUnit.NANOSECONDS.toMicros(System.nanoTime()) - startTime);
    }

    private record StatusApplication(
            WorkerEndpoint endpoint,
            WorkerEndpoint.StatusChange change,
            boolean newStatus,
            boolean newFinished,
            WorkerStatus.StepMetrics previousStep) {}
}
