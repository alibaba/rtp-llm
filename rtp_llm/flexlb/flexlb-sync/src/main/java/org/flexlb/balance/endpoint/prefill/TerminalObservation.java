package org.flexlb.balance.endpoint.prefill;

import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.enums.PriorityPreemptionProgress;

import java.util.Objects;

public record TerminalObservation(String requestId,
                           long batchId,
                           long executionTimeMs,
                           long errorCode,
                           PriorityPreemptionProgress preemptionProgress,
                           boolean workerObserved) {
    public static TerminalObservation from(WorkerStatus.TaskObservation task, boolean preserveBatchId) {
        return new TerminalObservation(task.requestId(), preserveBatchId ? task.batchId() : -1L, task.executionTimeMs(),
                task.errorCode(), task.priorityPreemptionProgress(), true);
    }

    public static TerminalObservation external(String requestId) {
        return new TerminalObservation(requestId, -1L, -1L, 0L, PriorityPreemptionProgress.NONE, false);
    }

    public TerminalObservation merge(TerminalObservation other) {
        if (!Objects.equals(requestId, other.requestId)) {
            throw new IllegalArgumentException("cannot merge different terminal requests");
        }
        if (batchId >= 0L && other.batchId >= 0L && batchId != other.batchId) {
            throw new IllegalArgumentException("cannot merge different terminal batches");
        }
        long mergedError = errorCode != 0L ? errorCode : other.errorCode;
        return new TerminalObservation(requestId, batchId >= 0L ? batchId : other.batchId,
                Math.max(executionTimeMs, other.executionTimeMs), mergedError,
                PriorityPreemptionProgress.merge(preemptionProgress, other.preemptionProgress),
                workerObserved || other.workerObserved);
    }
}
