package org.flexlb.balance.scheduler;

import org.flexlb.config.DecisionPolicyConfig;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.PreemptionConfig;
import org.flexlb.config.QueueOrderingConfig;
import org.flexlb.config.SchedulerConfig;
import org.flexlb.config.VictimStage;

import java.util.EnumSet;

/** Test-only builders for the public scheduler and dispatcher variants. */
public final class SchedulingTestConfig {

    private SchedulingTestConfig() {
    }

    public static FlexlbConfig batchConfig() {
        FlexlbConfig config = newConfig();
        useBatchDispatcher(config);
        return config;
    }

    public static SchedulerConfig usePriorityQueue(FlexlbConfig config) {
        SchedulerConfig queue = activeQueueOrNew(config);
        QueueOrderingConfig priority = queue.getOrdering().getType()
                == QueueOrderingConfig.Type.PRIORITY
                ? queue.getOrdering() : QueueOrderingConfig.priority();
        queue.setOrdering(priority);
        config.setScheduler(queue);
        return queue;
    }

    public static SchedulerConfig useFifoQueue(FlexlbConfig config) {
        SchedulerConfig queue = activeQueueOrNew(config);
        queue.setOrdering(new QueueOrderingConfig());
        config.setScheduler(queue);
        return queue;
    }

    public static void useSingleDecision(FlexlbConfig config) {
        SchedulerConfig queue = activeQueueOrNew(config);
        queue.setDecision(DecisionPolicyConfig.single());
        config.setScheduler(queue);
    }

    public static DecisionPolicyConfig useFixedWindowDecision(FlexlbConfig config) {
        SchedulerConfig queue = activeQueueOrNew(config);
        if (queue.getDecision().getType()
                == DecisionPolicyConfig.Type.FIXED_WINDOW) {
            return queue.getDecision();
        }
        DecisionPolicyConfig fixedWindow = new DecisionPolicyConfig();
        queue.setDecision(fixedWindow);
        config.setScheduler(queue);
        return fixedWindow;
    }

    public static DispatcherConfig useBatchDispatcher(FlexlbConfig config) {
        configureRequiredValues(config);
        if (config.getDispatcher().getType() == DispatcherConfig.Type.BATCH) {
            return config.getDispatcher();
        }
        DispatcherConfig batch = new DispatcherConfig();
        batch.setMaxInflightPerPrefillWorker(2);
        config.setDispatcher(batch);
        return batch;
    }

    public static DispatcherConfig useNonBatchDispatcher(FlexlbConfig config) {
        configureRequiredValues(config);
        if (config.getDispatcher().getType()
                == DispatcherConfig.Type.NON_BATCH) {
            return config.getDispatcher();
        }
        DispatcherConfig nonBatch = DispatcherConfig.nonBatch();
        nonBatch.setMaxInflightPerPrefillWorker(64);
        config.setDispatcher(nonBatch);
        return nonBatch;
    }

    public static PreemptionConfig preemption(FlexlbConfig config) {
        usePriorityQueue(config);
        QueueOrderingConfig priority = config.priorityOrdering();
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
    }

    public static void disallowVictim(FlexlbConfig config, VictimStage stage) {
        PreemptionConfig preemption = preemption(config);
        EnumSet<VictimStage> stages = preemption.getAllowedVictimStages().isEmpty()
                ? EnumSet.noneOf(VictimStage.class)
                : EnumSet.copyOf(preemption.getAllowedVictimStages());
        stages.remove(stage);
        preemption.setAllowedVictimStages(stages);
    }

    public static FlexlbConfig newConfig() {
        FlexlbConfig config = new FlexlbConfig();
        config.getDispatcher().setMaxInflightPerPrefillWorker(2);
        configureRequiredValues(config);
        return config;
    }

    public static void configureRequiredValues(FlexlbConfig config) {
        if (config.getRequestLifecycle().getRequest().getTimeoutMs() == null) {
            config.getRequestLifecycle().getRequest().setTimeoutMs(60_000L);
        }
        if (config.getRequestLifecycle().getDecision().getLifetime() == null) {
            config.getRequestLifecycle().getDecision().setLifetime(2.0);
        }
    }

    private static SchedulerConfig activeQueueOrNew(FlexlbConfig config) {
        configureRequiredValues(config);
        return config.isQueue() ? config.getScheduler() : new SchedulerConfig();
    }
}
