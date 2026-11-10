package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillRequestQueue.QueueItem;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.config.DecisionPolicyConfig;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.WorkerStatus;

import java.util.Comparator;

import static com.google.common.base.Preconditions.checkArgument;

/**
 * Immutable batch limits shared by routing projections and Prefill execution.
 */
public record QueueExecutionSettings(
        DispatcherConfig.Type dispatcherType,
        GroupPlanner grouping,
        int maxRequests,
        long collectionWaitMs,
        long executionBudgetMs) {
    /**
     * Stable equality preserves accepted order and puts projected arrivals at the tail.
     */
    public static Comparator<QueueItem> queueOrder() {
        return (left, right) -> 0;
    }

    public static QueueExecutionSettings capture(FlexlbConfig config, GroupPlanner grouping) {
        checkArgument(config.isQueue(), "QUEUE settings required");
        var decision = config.decisionPolicy();
        boolean single = decision.getType() == DecisionPolicyConfig.Type.SINGLE;
        return new QueueExecutionSettings(
                config.deliveryType(org.flexlb.dao.route.RequestPhase.GENERATION),
                grouping,
                single ? 1 : decision.resolveMaxRequests(),
                single ? 0L : Math.max(0L, decision.getMaxCollectionWaitMs()),
                single || decision.getMaxPredictedExecutionMs() == null
                        ? 0L
                        : decision.getMaxPredictedExecutionMs());
    }

    public GroupPlanner.Constraints constraints(WorkerStatus.EngineObservation engine) {
        return new GroupPlanner.Constraints(
                maxRequests,
                tokenCapacity(engine),
                kvCapacity(engine),
                executionBudgetMs,
                collectionWaitMs);
    }

    /** Compare the effective limits used by constraints, including unknown-capacity normalization. */
    public static boolean capacityChanged(
            WorkerStatus.EngineObservation before, WorkerStatus.EngineObservation after) {
        return tokenCapacity(before) != tokenCapacity(after)
                || kvCapacity(before) != kvCapacity(after);
    }

    private static long tokenCapacity(WorkerStatus.EngineObservation engine) {
        if (engine == null) return Long.MAX_VALUE;
        long maximum =
                engine.maxBatchTokensSize() > 0L ? engine.maxBatchTokensSize() : engine.maxSeqLen();
        return maximum <= 0L ? Long.MAX_VALUE : maximum;
    }

    private static long kvCapacity(WorkerStatus.EngineObservation engine) {
        if (engine == null || engine.totalKvCacheTokens() <= 0L) return Long.MAX_VALUE;
        return Math.clamp(engine.availableKvCacheTokens(), 0L, engine.totalKvCacheTokens());
    }
}
