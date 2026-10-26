package org.flexlb.balance.scheduler;

import org.flexlb.config.DecisionPolicyConfig;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.config.FlexlbConfig;

import static com.google.common.base.Preconditions.checkArgument;

/**
 * Only the immutable configuration shared by all PrefillQueueRunners.
 */
public record QueueExecutionSettings(
        boolean priorityOrdering,
        DispatcherConfig.Type dispatcherType,
        DecisionPolicyConfig.Type grouping,
        int maxRequests,
        long collectionWaitMs,
        long executionBudgetMs,
        long fallbackBatchTokenCapacity) {
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
                        : decision.getMaxPredictedExecutionMs(),
                config.getFallbackBatchTokenCapacity());
    }
}
