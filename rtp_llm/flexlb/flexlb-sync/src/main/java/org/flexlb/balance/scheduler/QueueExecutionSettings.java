package org.flexlb.balance.scheduler;

import org.flexlb.config.DecisionPolicyConfig;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.VictimStage;

import static com.google.common.base.Preconditions.checkArgument;

/** Only the immutable configuration shared by all WorkerBatchers. */
public record QueueExecutionSettings(boolean priorityOrdering, boolean preemptQueued,
        DispatcherConfig.Type dispatcherType, long maxOutstandingRequests,
        DecisionPolicyConfig.Type grouping, int maxRequests, long collectionWaitMs, long executionBudgetMs) {
    public static QueueExecutionSettings capture(FlexlbConfig config) {
        checkArgument(config.isQueue(), "QUEUE settings required");
        var decision = config.decisionPolicy();
        boolean single = decision.getType() == DecisionPolicyConfig.Type.SINGLE;
        return new QueueExecutionSettings(config.isPriorityOrdering(), config.allowsPreemption(VictimStage.PREFILL_QUEUED),
                config.getDispatcher().getType(), config.getDispatcher().getType() == DispatcherConfig.Type.BATCH ? 0L : config.getDispatcher().getMaxInflightPerPrefillWorker(),
                decision.getType(), single ? 1 : decision.resolveMaxRequests(), single ? 0L : Math.max(0L, decision.getMaxCollectionWaitMs()),
                single || decision.getMaxPredictedExecutionMs() == null ? 0L : decision.getMaxPredictedExecutionMs());
    }
}
