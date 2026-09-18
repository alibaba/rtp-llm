package org.flexlb.config;

import com.fasterxml.jackson.annotation.JsonIgnore;
import lombok.Getter;
import lombok.Setter;

/**
 * Public FLEXLB_CONFIG contract, organized by stable responsibility owner.
 * Mutually exclusive behavior is represented by tagged unions so inactive
 * variant fields fail during deserialization.
 */
@Getter
@Setter
public final class FlexlbConfig {

    public static final int CURRENT_SCHEMA_VERSION = 3;

    private int schemaVersion = CURRENT_SCHEMA_VERSION;
    private SchedulerConfig scheduler = new SchedulerConfig();
    private DispatcherConfig dispatcher = new DispatcherConfig();
    private RequestLifecycleConfig requestLifecycle = new RequestLifecycleConfig();
    private RoutingConfig router = new RoutingConfig();
    private WorkerRegistryConfig workerRegistry = new WorkerRegistryConfig();
    private ObservabilityConfig observability = new ObservabilityConfig();
    private GrpcServerConfig grpcServer = new GrpcServerConfig();

    @JsonIgnore
    private final InternalRuntimeSettings internalRuntime = new InternalRuntimeSettings();

    @JsonIgnore
    public boolean isDirect() {
        return scheduler.getType() == SchedulerConfig.Type.DIRECT;
    }

    @JsonIgnore
    public boolean isQueue() {
        return scheduler.getType() == SchedulerConfig.Type.QUEUE;
    }

    @JsonIgnore
    public boolean isPriorityOrdering() {
        return isQueue()
                && scheduler.getOrdering().getType()
                == QueueOrderingConfig.Type.PRIORITY;
    }

    @JsonIgnore
    public boolean allowsPreemption(VictimStage stage) {
        return isPriorityOrdering() && scheduler.getOrdering().getPreemption() != null
                && scheduler.getOrdering().getPreemption().allows(stage);
    }

    /** Resolve the QUEUE decision policy from its single configuration owner. */
    @JsonIgnore
    public DecisionPolicyConfig decisionPolicy() {
        return queueScheduler().getDecision();
    }

    @JsonIgnore
    public boolean isSingleDecision() {
        return isQueue() && queueScheduler().getDecision().getType()
                == DecisionPolicyConfig.Type.SINGLE;
    }

    @JsonIgnore
    public boolean isFixedWindowDecision() {
        return isQueue() && queueScheduler().getDecision().getType()
                == DecisionPolicyConfig.Type.FIXED_WINDOW;
    }

    @JsonIgnore
    public DecisionPolicyConfig fixedWindowDecision() {
        DecisionPolicyConfig policy = decisionPolicy();
        if (policy.getType() == DecisionPolicyConfig.Type.FIXED_WINDOW) {
            return policy;
        }
        throw new IllegalStateException(
                "fixed-window decision configuration is not active");
    }

    @JsonIgnore
    public SchedulerConfig queueScheduler() {
        if (isQueue()) {
            return scheduler;
        }
        throw new IllegalStateException("queue scheduler configuration is not active");
    }

    @JsonIgnore
    public QueueOrderingConfig priorityOrdering() {
        QueueOrderingConfig ordering = queueScheduler().getOrdering();
        if (ordering.getType() == QueueOrderingConfig.Type.PRIORITY) {
            return ordering;
        }
        throw new IllegalStateException("priority ordering configuration is not active");
    }

    @Getter
    @Setter
    public static final class GrpcServerConfig {
        private int executorCoreSize = 1000;
        private int executorMaxSize = 1000;
        private int executorQueueSize = 1000;
    }

}
