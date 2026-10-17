package org.flexlb.testfixture;

import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatus.EngineObservation;
import org.flexlb.dao.master.WorkerStatus.StatusObservation;
import org.flexlb.dao.master.WorkerStatus.TaskObservation;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/** Test data setup through the production poll and publication APIs. */
public final class WorkerStatusTestFixtures {
    private WorkerStatusTestFixtures() { }

    public static WorkerStatus readyForScheduling(WorkerStatus worker) {
        receive(worker, true, null);
        return worker;
    }

    public static WorkerStatus.PollLease beginStatusPoll(WorkerStatus worker, boolean alive) {
        WorkerStatus.PollLease lease = Objects.requireNonNull(worker.tryBeginStatusPoll(1L));
        worker.recordStatusRpcResult(lease, true, alive, null);
        return lease;
    }

    /** Caller holds the generation lock, as required by prepareNewStatus. */
    public static WorkerStatus.PreparedStatus prepare(
            WorkerStatus worker, StatusObservation observation) {
        try (WorkerStatus.PollLease lease = beginStatusPoll(worker, observation.alive())) {
            return worker.prepareNewStatus(observation, lease);
        }
    }

    public static WorkerStatus.PollHealth receive(WorkerStatus worker, boolean alive, Throwable failure) {
        try (WorkerStatus.PollLease lease = Objects.requireNonNull(worker.tryBeginStatusPoll(1L))) {
            worker.recordStatusRpcResult(lease, failure == null, alive, failure);
        }
        return worker.pollHealth();
    }

    /**
     * Deep-freeze one RPC response. Finished tasks remain response-local and
     * are never copied into the committed Engine observation.
     */
    public static StatusObservation observation(
            WorkerStatus worker, WorkerStatusResponse response) {
        Objects.requireNonNull(response, "response");
        Map<String, TaskObservation> runningTasks = freezeTaskMap(
                response.getRunningTaskInfo());
        Map<String, TaskObservation> finishedTasks = freezeTaskMap(
                response.getFinishedTaskInfo());
        EngineObservation engine = new EngineObservation(
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
                response.getMaxSeqLen(),
                response.getMaxBatchTokensSize(),
                response.getRunningQueryLen(),
                response.getWaitingQueryLen());
        return worker.bindStatusObservation(
                engine,
                response.isAlive(),
                response.getStatusVersion(),
                response.getLatestFinishedVersion(),
                finishedTasks);
    }

    private static Map<String, TaskObservation> freezeTaskMap(
            Map<String, TaskInfo> tasks) {
        if (tasks == null || tasks.isEmpty()) {
            return Map.of();
        }
        Map<String, TaskObservation> frozen = new HashMap<>(tasks.size());
        for (Map.Entry<String, TaskInfo> entry : tasks.entrySet()) {
            if (entry.getKey() != null && entry.getValue() != null) {
                frozen.put(entry.getKey(), copyTask(
                        entry.getValue()));
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
                task.getPriorityPreemptionProgress());
    }

}
