package org.flexlb.dao.master;

import java.util.List;

/**
 * Outcomes produced while reconciling local tasks with the latest worker task states.
 *
 * <p>Additional task-state update outcomes can be added here without changing the
 * {@link WorkerStatus#updateTaskStates} contract or its callers.
 */
public record TaskStateUpdateResult(
        List<CacheHitFeedback> cacheHitFeedbacks,
        List<Long> decisionToWaitingObservedLatenciesMs,
        List<Long> waitingToRunningObservedLatenciesMs,
        List<Long> engineWaitingToRunningLatenciesMs,
        List<Long> engineReceivedToWaitingLatenciesMs) {

    private static final TaskStateUpdateResult EMPTY =
            new TaskStateUpdateResult(List.of(), List.of(), List.of(), List.of(), List.of());

    public static TaskStateUpdateResult from(
            List<CacheHitFeedback> cacheHitFeedbacks,
            List<Long> decisionToWaitingObservedLatenciesMs,
            List<Long> waitingToRunningObservedLatenciesMs,
            List<Long> engineWaitingToRunningLatenciesMs,
            List<Long> engineReceivedToWaitingLatenciesMs) {
        if (cacheHitFeedbacks.isEmpty()
                && decisionToWaitingObservedLatenciesMs.isEmpty()
                && waitingToRunningObservedLatenciesMs.isEmpty()
                && engineWaitingToRunningLatenciesMs.isEmpty()
                && engineReceivedToWaitingLatenciesMs.isEmpty()) {
            return EMPTY;
        }
        return new TaskStateUpdateResult(
                cacheHitFeedbacks, decisionToWaitingObservedLatenciesMs,
                waitingToRunningObservedLatenciesMs,
                engineWaitingToRunningLatenciesMs, engineReceivedToWaitingLatenciesMs);
    }
}
