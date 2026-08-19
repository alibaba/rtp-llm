package org.flexlb.sync.runner;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.balance.scheduler.FlexlbBatchScheduler;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.BalanceStatusEnum;
import org.flexlb.service.grpc.EngineGrpcService;
import org.flexlb.service.grpc.EngineStatusConverter;
import org.flexlb.service.monitor.EngineHealthReporter;
import org.flexlb.sync.status.WorkerGenerationFence;
import org.flexlb.sync.status.WorkerGenerationManager;
import org.flexlb.util.CommonUtils;

import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.Executor;

import static org.flexlb.constant.CommonConstants.DEADLINE_EXCEEDED_MESSAGE;

/** Applies one current WorkerStatus generation response. */
public class GrpcWorkerStatusRunner implements Runnable {

    private static final int MAX_CONSECUTIVE_FAILURES = 3;

    private enum StatusApplyResult {
        STALE_GENERATION,
        VERSION_ROLLBACK,
        APPLIED
    }

    private final String ipPort;
    private final String modelName;
    private final String site;
    private final RoleType roleType;
    private final String group;
    private final WorkerStatus workerStatus;
    private final ConcurrentMap<String, WorkerStatus> workerStatusMap;
    private final EngineHealthReporter engineHealthReporter;
    private final EngineGrpcService engineGrpcService;
    private final FlexlbBatchScheduler batchScheduler;
    private final EndpointRegistry endpointRegistry;
    private final WorkerGenerationManager generationManager;
    private final WorkerGenerationFence generationFence;
    private final String ip;
    private final int grpcPort;
    private final long syncRequestTimeoutMs;
    private final Executor callbackExecutor;

    public GrpcWorkerStatusRunner(String modelName, String ipPort, String site,
                                  RoleType roleType, String group,
                                  WorkerStatus workerStatus,
                                  ConcurrentMap<String, WorkerStatus> workerStatusMap,
                                  EngineHealthReporter engineHealthReporter,
                                  EngineGrpcService engineGrpcService,
                                  long syncRequestTimeoutMs,
                                  FlexlbBatchScheduler batchScheduler,
                                  EndpointRegistry endpointRegistry,
                                  WorkerGenerationManager generationManager,
                                  WorkerGenerationFence generationFence,
                                  Executor callbackExecutor) {
        this.ipPort = ipPort;
        int separator = ipPort.lastIndexOf(':');
        this.ip = ipPort.substring(0, separator);
        this.grpcPort = CommonUtils.toGrpcPort(
                Integer.parseInt(ipPort.substring(separator + 1)));
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
        this.generationManager = generationManager;
        this.generationFence = generationFence;
        this.callbackExecutor = callbackExecutor;
    }

    @Override
    public void run() {
        boolean callbackInstalled = false;
        try {
            long startTimeUs = System.nanoTime() / 1000;
            CompletableFuture<WorkerStatusResponse> rpc = engineGrpcService
                    .getWorkerStatusAsync(
                            ip, grpcPort,
                            workerStatus.getLatestFinishedTaskVersion().get(),
                            syncRequestTimeoutMs, roleType)
                    .thenApply(EngineStatusConverter::convertToWorkerStatusResponse);
            CompletableFuture<WorkerStatusResponse> callback = rpc.whenCompleteAsync(
                    (response, failure) -> {
                        if (failure != null) {
                            handleFailure(unwrap(failure));
                            return;
                        }
                        handleResponse(response, startTimeUs);
                    }, callbackExecutor);
            callback.whenComplete((ignored, callbackFailure) ->
                    workerStatus.getStatusCheckInProgress().set(false));
            callbackInstalled = true;
        } finally {
            if (!callbackInstalled) {
                workerStatus.getStatusCheckInProgress().set(false);
            }
        }
    }

    private void handleResponse(WorkerStatusResponse response, long startTimeUs) {
        try {
            if (response == null) {
                if (isCurrentGeneration()) {
                    engineHealthReporter.reportStatusCheckerFail(
                            modelName, BalanceStatusEnum.RESPONSE_NULL, roleType);
                }
                return;
            }
            if (response.getRole() != roleType) {
                if (isCurrentGeneration()) {
                    engineHealthReporter.reportStatusCheckerFail(
                            modelName, BalanceStatusEnum.UNKNOWN_ERROR, roleType);
                }
                return;
            }
            engineHealthReporter.reportStatusCheckRemoteInfo(
                    modelName, roleType.name(), startTimeUs);

            long responseVersion = response.getStatusVersion();
            if (responseVersion < 0) {
                return;
            }
            StatusApplyResult result = generationFence.read(
                    ipPort, () -> applyCurrentGeneration(response));
            if (result == StatusApplyResult.VERSION_ROLLBACK) {
                generationManager.rotateOnVersionRollback(
                        workerStatusMap, roleType, ipPort, workerStatus, responseVersion);
                return;
            }
            if (result != StatusApplyResult.APPLIED) {
                return;
            }

            engineHealthReporter.reportStatusCheckerSuccess(
                    modelName, workerStatus,
                    sizeOf(response.getRunningTaskInfo()),
                    sizeOf(response.getFinishedTaskInfo()));
        } catch (Throwable applyFailure) {
            engineHealthReporter.reportStatusCheckerFail(
                    modelName, BalanceStatusEnum.UNKNOWN_ERROR, roleType);
        }
    }

    private StatusApplyResult applyCurrentGeneration(WorkerStatusResponse response) {
        if (!generationManager.isCurrent(workerStatusMap, ipPort, workerStatus)) {
            return StatusApplyResult.STALE_GENERATION;
        }
        long responseVersion = response.getStatusVersion();
        long currentVersion = workerStatus.getStatusVersion().get();
        if (responseVersion < currentVersion) {
            return StatusApplyResult.VERSION_ROLLBACK;
        }

        boolean aliveChanged = workerStatus.isAlive() != response.isAlive();
        boolean versionAdvanced = responseVersion > currentVersion;
        workerStatus.getConsecutiveFailures().set(0);
        workerStatus.setSite(site);
        workerStatus.setGroup(group);
        if (versionAdvanced) {
            workerStatus.updateFromResponse(response);
        } else {
            workerStatus.refreshStatusHeartbeat(response.isAlive());
        }

        RuntimeException pipelineFailure = null;
        if (batchLifecycleEnabled()) {
            try {
                batchScheduler.recordRequestActivity(workerStatus, response);
            } catch (RuntimeException activityFailure) {
                pipelineFailure = activityFailure;
            }
        }

        long finishedVersion = response.getLatestFinishedVersion() == null
                ? -1L : response.getLatestFinishedVersion();
        boolean finishedCursorPending = finishedVersion
                > workerStatus.getLatestFinishedTaskVersion().get();
        boolean statusUpdatePending = responseVersion
                > workerStatus.getLastAppliedStatusVersion().get();
        try {
            if (statusUpdatePending || finishedCursorPending || aliveChanged) {
                endpointRegistry.updateEndpointFromWorkerStatus(workerStatus, response);
            } else {
                endpointRegistry.refreshEndpointActivity(workerStatus, response);
            }
        } catch (RuntimeException endpointFailure) {
            pipelineFailure = mergeFailure(pipelineFailure, endpointFailure);
        }
        if (statusUpdatePending || finishedCursorPending || aliveChanged) {
            if (batchLifecycleEnabled()) {
                try {
                    batchScheduler.updateRequestLifecycleFromWorkerStatus(
                            workerStatus, response);
                } catch (RuntimeException lifecycleFailure) {
                    pipelineFailure = mergeFailure(pipelineFailure, lifecycleFailure);
                }
            }
            if (pipelineFailure != null) {
                throw pipelineFailure;
            }
            if (!generationManager.isCurrent(workerStatusMap, ipPort, workerStatus)) {
                return StatusApplyResult.STALE_GENERATION;
            }
            if (finishedCursorPending) {
                workerStatus.getLatestFinishedTaskVersion().set(finishedVersion);
            }
            workerStatus.getLastAppliedStatusVersion().set(responseVersion);
        }
        if (pipelineFailure != null) {
            throw pipelineFailure;
        }
        return StatusApplyResult.APPLIED;
    }

    private void handleFailure(Throwable failure) {
        String message = failure == null ? null : failure.getMessage();
        BalanceStatusEnum status = message != null
                && message.toLowerCase().contains(DEADLINE_EXCEEDED_MESSAGE.toLowerCase())
                ? BalanceStatusEnum.WORKER_STATUS_GRPC_TIMEOUT
                : BalanceStatusEnum.WORKER_SERVICE_UNAVAILABLE;
        // A failed RPC is not a WorkerStatus heartbeat. Keep the generation
        // probeable, but leave its last-success timestamp unchanged so the
        // expiration owner can eventually retire endpoint and cache state.
        boolean currentGeneration = generationFence.read(ipPort, () -> {
            if (!generationManager.isCurrent(workerStatusMap, ipPort, workerStatus)) {
                return false;
            }
            long failures = workerStatus.getConsecutiveFailures().incrementAndGet();
            if (failures >= MAX_CONSECUTIVE_FAILURES) {
                endpointRegistry.beginEndpointRetirement(roleType, ipPort, workerStatus);
                workerStatus.setAlive(false);
                endpointRegistry.remove(roleType, ipPort, workerStatus);
            }
            return true;
        });
        if (currentGeneration) {
            engineHealthReporter.reportStatusCheckerFail(modelName, status, roleType);
        }
    }

    private boolean batchLifecycleEnabled() {
        return batchScheduler != null
                && (roleType == RoleType.PREFILL || roleType == RoleType.DECODE);
    }

    private boolean isCurrentGeneration() {
        return generationFence.read(
                ipPort, () -> generationManager.isCurrent(
                        workerStatusMap, ipPort, workerStatus));
    }

    private static int sizeOf(Map<?, ?> value) {
        return value == null ? 0 : value.size();
    }

    private static RuntimeException mergeFailure(
            RuntimeException previous, RuntimeException next) {
        if (previous == null) {
            return next;
        }
        previous.addSuppressed(next);
        return previous;
    }

    private static Throwable unwrap(Throwable failure) {
        return failure instanceof CompletionException && failure.getCause() != null
                ? failure.getCause() : failure;
    }
}
