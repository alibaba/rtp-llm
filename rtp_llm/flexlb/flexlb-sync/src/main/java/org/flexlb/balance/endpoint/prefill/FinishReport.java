package org.flexlb.balance.endpoint.prefill;

import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.enums.PriorityPreemptionProgress;

import java.util.Objects;

public record FinishReport(String requestId,
                           long batchId,
                           long executionTimeMs,
                           long errorCode,
                           PriorityPreemptionProgress preemptionProgress,
                           boolean reportedByWorker) {
    public static FinishReport from(WorkerStatus.TaskObservation task, boolean preserveBatchId) {
        return new FinishReport(task.requestId(), preserveBatchId ? task.batchId() : -1L, task.executionTimeMs(),
                task.errorCode(), task.priorityPreemptionProgress(), true);
    }

    public static FinishReport local(String requestId) {
        return new FinishReport(requestId, -1L, -1L, 0L, PriorityPreemptionProgress.NONE, false);
    }

    public FinishReport merge(FinishReport other) {
        if (!Objects.equals(requestId, other.requestId)) {
            throw new IllegalArgumentException("cannot merge finish reports of different requests");
        }
        if (batchId >= 0L && other.batchId >= 0L && batchId != other.batchId) {
            throw new IllegalArgumentException("cannot merge finish reports of different batches");
        }
        long mergedError = errorCode != 0L ? errorCode : other.errorCode;
        return new FinishReport(requestId, batchId >= 0L ? batchId : other.batchId,
                Math.max(executionTimeMs, other.executionTimeMs), mergedError,
                PriorityPreemptionProgress.merge(preemptionProgress, other.preemptionProgress),
                reportedByWorker || other.reportedByWorker);
    }
}
