package org.flexlb.balance.scheduler;

import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.config.DecisionPolicyConfig;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.WorkerStatus;

import java.util.Objects;

import static com.google.common.base.Preconditions.checkArgument;

/** Immutable queue rules shared by routing projections and Prefill execution. */
public record QueueExecutionSettings(
        boolean priorityOrdering,
        DispatcherConfig.Type dispatcherType,
        DecisionPolicyConfig.Type grouping,
        int maxRequests,
        long collectionWaitMs,
        long executionBudgetMs) {
    public static QueueExecutionSettings capture(FlexlbConfig config) {
        checkArgument(config.isQueue(), "QUEUE settings required");
        var decision = config.decisionPolicy();
        boolean single = decision.getType() == DecisionPolicyConfig.Type.SINGLE;
        return new QueueExecutionSettings(
                config.isPriorityOrdering(),
                config.getDispatcher().getType(),
                decision.getType(),
                single ? 1 : decision.resolveMaxRequests(),
                single ? 0L : Math.max(0L, decision.getMaxCollectionWaitMs()),
                single || decision.getMaxPredictedExecutionMs() == null
                        ? 0L
                        : decision.getMaxPredictedExecutionMs());
    }

    public GroupPlanner.Constraints constraints(WorkerStatus.EngineObservation engine) {
        return new GroupPlanner.Constraints(
                maxRequests,
                schedulingCapacity(engine, PlanningResource.TOKENS),
                schedulingCapacity(engine, PlanningResource.KV),
                executionBudgetMs,
                collectionWaitMs);
    }

    public GroupPlanner groupingPolicy() {
        return grouping == DecisionPolicyConfig.Type.SINGLE
                ? GroupPlanner.SINGLE
                : GroupPlanner.FIXED_WINDOW;
    }

    public enum PlanningResource {
        TOKENS,
        KV
    }

    /** The same effective limits drive worker selection and status input invalidation. */
    public static long schedulingCapacity(
            WorkerStatus.EngineObservation engine, PlanningResource resource) {
        Objects.requireNonNull(resource, "resource");
        if (engine == null) {
            return Long.MAX_VALUE;
        }
        long maximum =
                resource == PlanningResource.KV
                        ? engine.totalKvCacheTokens()
                        : engine.maxBatchTokensSize() > 0L
                                ? engine.maxBatchTokensSize()
                                : engine.maxSeqLen();
        return maximum <= 0L
                ? Long.MAX_VALUE
                : resource == PlanningResource.KV
                        ? Math.clamp(engine.availableKvCacheTokens(), 0L, maximum)
                        : maximum;
    }
}
