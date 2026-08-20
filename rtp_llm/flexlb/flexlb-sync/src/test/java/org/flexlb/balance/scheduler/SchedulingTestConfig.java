package org.flexlb.balance.scheduler;

import org.flexlb.config.BatchDispatcherConfig;
import org.flexlb.config.EngineCancellationConfig;
import org.flexlb.config.FifoOrderingConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.NonBatchDispatcherConfig;
import org.flexlb.config.PreemptionConfig;
import org.flexlb.config.PriorityOrderingConfig;
import org.flexlb.config.QueueSchedulerConfig;
import org.flexlb.config.VictimStage;

import java.util.EnumSet;

/** Test-only builders for the public scheduler and dispatcher variants. */
public final class SchedulingTestConfig {

    private SchedulingTestConfig() {
    }

    public static FlexlbConfig batchConfig() {
        FlexlbConfig config = new FlexlbConfig();
        useBatchDispatcher(config);
        return config;
    }

    public static QueueSchedulerConfig usePriorityQueue(FlexlbConfig config) {
        QueueSchedulerConfig queue = activeQueueOrNew(config);
        PriorityOrderingConfig priority = queue.getOrdering() instanceof PriorityOrderingConfig active
                ? active : new PriorityOrderingConfig();
        queue.setOrdering(priority);
        config.setScheduler(queue);
        return queue;
    }

    public static QueueSchedulerConfig useFifoQueue(FlexlbConfig config) {
        QueueSchedulerConfig queue = activeQueueOrNew(config);
        queue.setOrdering(new FifoOrderingConfig());
        config.setScheduler(queue);
        return queue;
    }

    public static BatchDispatcherConfig useBatchDispatcher(FlexlbConfig config) {
        if (config.getDispatcher() instanceof BatchDispatcherConfig batch) {
            return batch;
        }
        BatchDispatcherConfig batch = new BatchDispatcherConfig();
        config.setDispatcher(batch);
        return batch;
    }

    public static NonBatchDispatcherConfig useNonBatchDispatcher(FlexlbConfig config) {
        if (config.getDispatcher() instanceof NonBatchDispatcherConfig nonBatch) {
            return nonBatch;
        }
        NonBatchDispatcherConfig nonBatch = new NonBatchDispatcherConfig();
        config.setDispatcher(nonBatch);
        return nonBatch;
    }

    public static PreemptionConfig preemption(FlexlbConfig config) {
        usePriorityQueue(config);
        PriorityOrderingConfig priority = config.priorityOrdering();
        if (priority.getPreemption() == null) {
            priority.setPreemption(new PreemptionConfig());
        }
        return priority.getPreemption();
    }

    public static void allowVictim(FlexlbConfig config, VictimStage stage) {
        PreemptionConfig preemption = preemption(config);
        EnumSet<VictimStage> stages = preemption.getAllowedVictimStages().isEmpty()
                ? EnumSet.noneOf(VictimStage.class)
                : EnumSet.copyOf(preemption.getAllowedVictimStages());
        stages.add(stage);
        preemption.setAllowedVictimStages(stages);
        if (stage == VictimStage.DECODE_ENGINE_OWNED
                && preemption.getEngineCancellation() == null) {
            preemption.setEngineCancellation(new EngineCancellationConfig());
        }
    }

    public static void disallowVictim(FlexlbConfig config, VictimStage stage) {
        PreemptionConfig preemption = preemption(config);
        EnumSet<VictimStage> stages = preemption.getAllowedVictimStages().isEmpty()
                ? EnumSet.noneOf(VictimStage.class)
                : EnumSet.copyOf(preemption.getAllowedVictimStages());
        stages.remove(stage);
        preemption.setAllowedVictimStages(stages);
        if (stage == VictimStage.DECODE_ENGINE_OWNED) {
            preemption.setEngineCancellation(null);
        }
    }

    public static EngineCancellationConfig engineCancellation(FlexlbConfig config) {
        PreemptionConfig preemption = preemption(config);
        if (preemption.getEngineCancellation() == null) {
            preemption.setEngineCancellation(new EngineCancellationConfig());
        }
        return preemption.getEngineCancellation();
    }

    private static QueueSchedulerConfig activeQueueOrNew(FlexlbConfig config) {
        return config.getScheduler() instanceof QueueSchedulerConfig queue
                ? queue : new QueueSchedulerConfig();
    }
}
