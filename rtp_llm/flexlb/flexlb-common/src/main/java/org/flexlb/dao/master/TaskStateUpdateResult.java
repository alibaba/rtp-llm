package org.flexlb.dao.master;

import java.util.List;

/**
 * Outcomes produced while reconciling local tasks with the latest worker task states.
 *
 * <p>Additional task-state update outcomes can be added here without changing the
 * {@link WorkerStatus#updateTaskStates} contract or its callers.
 */
public record TaskStateUpdateResult(List<CacheHitFeedback> cacheHitFeedbacks) {

    private static final TaskStateUpdateResult EMPTY = new TaskStateUpdateResult(List.of());

    public static TaskStateUpdateResult fromCacheHitFeedbacks(List<CacheHitFeedback> cacheHitFeedbacks) {
        if (cacheHitFeedbacks.isEmpty()) {
            return EMPTY;
        }
        return new TaskStateUpdateResult(cacheHitFeedbacks);
    }
}
