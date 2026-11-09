package org.flexlb.testfixture;

import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatus.EngineObservation;
import org.flexlb.dao.master.WorkerStatus.StatusObservation;
import org.flexlb.dao.master.WorkerStatus.TaskObservation;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * Sets up worker observations and records test polls through the production APIs.
 * Live status pumps skip busy or unavailable generations; strict setup requires an admitted poll.
 */
public final class WorkerStatusTestFixtures {
    private WorkerStatusTestFixtures() {}

    public static WorkerStatus readyForScheduling(WorkerStatus worker) {
        receive(worker, true, null);
        return worker;
    }

    public static WorkerStatus.PollHealth receive(
            WorkerStatus worker, boolean alive, Throwable failure) {
        try (WorkerStatus.PollLease lease = worker.tryBeginStatusPoll(1L)) {
            if (lease != null) {
                worker.recordStatusRpcResult(lease, failure == null, alive, failure);
            }
        }
        return worker.pollHealth();
    }

    /**
     * Caller holds the generation lock, as required by prepareNewStatus.
     */
    public static WorkerStatus.PreparedStatus prepare(
            WorkerStatus worker, StatusObservation observation) {
        try (WorkerStatus.PollLease lease = beginStatusPoll(worker, observation.alive())) {
            if (!worker.tryBeginStatusApplication(lease)) {
                throw new IllegalStateException("test status application is busy or unavailable");
            }
            lease.close();
            try {
                return worker.prepareNewStatus(observation, lease);
            } finally {
                worker.finishStatusApplication(lease);
            }
        }
    }

    public static WorkerStatus.PollLease beginStatusPoll(WorkerStatus worker, boolean alive) {
        WorkerStatus.PollLease lease = worker.tryBeginStatusPoll(1L);
        if (lease == null) {
            throw new IllegalStateException(
                    "test setup requires an available poll: generation="
                            + worker.getGenerationId()
                            + ", retirement="
                            + worker.retirementReason());
        }
        worker.recordStatusRpcResult(lease, true, alive, null);
        return lease;
    }

    /**
     * Records one heartbeat and reserves a valid alive response for local application.
     * The RPC lease is closed before returning; the caller must finishStatusApplication in finally.
     */
    public static WorkerStatus.PollLease tryBeginStatusApplication(
            WorkerStatus worker, StatusObservation observation) {
        WorkerStatus.PollLease lease = worker.tryBeginStatusPoll(1L);
        if (lease == null) {
            return null;
        }
        try (lease) {
            boolean valid =
                    observation.owner() == worker
                            && observation.role() == worker.getRole()
                            && observation.statusVersion() != null
                            && observation.statusVersion() > 0L
                            && observation.statusVersion()
                                    >= worker.appliedStatusCursor().statusVersion();
            worker.recordStatusRpcResult(lease, valid, observation.alive(), null);
            return valid && observation.alive() && worker.tryBeginStatusApplication(lease)
                    ? lease
                    : null;
        }
    }

    /**
     * Deep-freeze one RPC response. Finished tasks remain response-local and
     * are never copied into the committed Engine observation.
     */
    public static StatusObservation observation(
            WorkerStatus worker, WorkerStatusResponse response) {
        Objects.requireNonNull(response, "response");
        Map<String, TaskObservation> runningTasks = freezeTaskMap(response.getRunningTaskInfo());
        Map<String, TaskObservation> finishedTasks = freezeTaskMap(response.getFinishedTaskInfo());
        EngineObservation engine =
                new EngineObservation(
                        response.getRole(),
                        response.getAvailableConcurrency(),
                        response.getAvailableKvCacheTokens(),
                        response.getTotalKvCacheTokens(),
                        runningTasks,
                        response.getStepLatencyMs(),
                        response.getIterateCount(),
                        response.getDpSize(),
                        response.getTpSize(),
                        response.getDpRank(),
                        response.getBlockSize(),
                        response.getBlockHashLookaheadTokens(),
                        response.getCacheMatchRollbackBlocks(),
                        response.getKvCacheGroupMode(),
                        response.getMaxSeqLen(),
                        response.getMaxBatchTokensSize(),
                        response.getRunningQueryLen(),
                        response.getWaitingQueryLen(),
                        response.getLastStepMetrics());
        return worker.bindStatusObservation(
                engine,
                response.isAlive(),
                response.getStatusVersion(),
                response.getLatestFinishedVersion(),
                finishedTasks);
    }

    private static Map<String, TaskObservation> freezeTaskMap(Map<String, TaskInfo> tasks) {
        if (tasks == null || tasks.isEmpty()) {
            return Map.of();
        }
        Map<String, TaskObservation> frozen = new HashMap<>(tasks.size());
        for (Map.Entry<String, TaskInfo> entry : tasks.entrySet()) {
            if (entry.getKey() != null && entry.getValue() != null) {
                frozen.put(entry.getKey(), copyTask(entry.getValue()));
            }
        }
        return Map.copyOf(frozen);
    }

    private static TaskObservation copyTask(TaskInfo task) {
        return new TaskObservation(
                task.getRequestId(),
                task.getPrefixLength(),
                task.getPrefillTime(),
                task.getInputLength(),
                task.getWaitingTime(),
                task.getIterateCount(),
                task.getEndTimeMs(),
                task.getDpRank(),
                task.getErrorCode(),
                task.getErrorMessage(),
                task.getBatchId(),
                task.getPhase(),
                task.getExecutionTimeMs(),
                task.getPriorityPreemptionProgress(),
                new WorkerStatus.TaskTelemetry(
                        task.isPrefixLengthValid(),
                        task.getRequestReceivedTimeMs(),
                        task.getInputQueueEnqueueTimeMs(),
                        task.getInputQueueDrainTimeMs(),
                        task.getWaitingEnteredTimeMs(),
                        task.getRunningEnteredTimeMs(),
                        task.getRemoteKvWaitMs(),
                        task.getFirstTokenTimeMs(),
                        task.getHbmLocalMatchTokens(),
                        task.getRemoteKvAddedMatchTokens(),
                        task.getFirstPrefillStepId(),
                        task.getLastPrefillStepId(),
                        task.getPrefillStepCount(),
                        task.getPrefillNonfinalChunkTokensMin(),
                        task.getPrefillNonfinalChunkTokensMax(),
                        task.getCompletedPrefillTokens(),
                        task.getRemainingPrefillTokens() < 0
                                ? null
                                : task.getRemainingPrefillTokens(),
                        task.getLastCompletedPrefillStepId()));
    }
}
