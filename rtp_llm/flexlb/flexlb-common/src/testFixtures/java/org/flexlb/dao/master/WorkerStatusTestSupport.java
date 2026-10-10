package org.flexlb.dao.master;

import java.util.Map;
import java.util.HashMap;
import java.util.Objects;

/** Converts mutable test observations into the production immutable status protocol. */
public final class WorkerStatusTestSupport {
    private WorkerStatusTestSupport() {}

    public static WorkerStatus.StatusObservation freezeStatusResponse(WorkerStatus status, WorkerStatusResponse response) {
        Objects.requireNonNull(response, "response");
        Map<String, WorkerStatus.TaskObservation> runningTasks = freezeTaskMap(
                response.getRunningTaskInfo());
        Map<String, WorkerStatus.TaskObservation> finishedTasks = freezeTaskMap(
                response.getFinishedTaskInfo());
        WorkerStatus.EngineObservation engine = new WorkerStatus.EngineObservation(
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
        return status.bindStatusObservation(
                engine,
                response.isAlive(),
                response.getStatusVersion(),
                response.getLatestFinishedVersion(),
                finishedTasks);
    }

    private static Map<String, WorkerStatus.TaskObservation> freezeTaskMap(Map<String, TaskInfo> tasks) {
        if (tasks == null || tasks.isEmpty()) {
            return Map.of();
        }
        Map<String, WorkerStatus.TaskObservation> frozen = new HashMap<>(tasks.size());
        for (Map.Entry<String, TaskInfo> entry : tasks.entrySet()) {
            if (entry.getKey() != null && entry.getValue() != null) {
                frozen.put(entry.getKey(), copyOf(
                        entry.getValue()));
            }
        }
        return Map.copyOf(frozen);
    }

    private static WorkerStatus.TaskObservation copyOf(TaskInfo task) {
        return new WorkerStatus.TaskObservation(
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
