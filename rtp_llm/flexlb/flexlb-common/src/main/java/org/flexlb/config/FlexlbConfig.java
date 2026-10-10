package org.flexlb.config;

import com.fasterxml.jackson.annotation.JsonIgnore;
import lombok.Getter;
import lombok.Setter;

import static com.google.common.base.Preconditions.checkState;

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
    private final InternalRuntimeSettings internalRuntime;

    public FlexlbConfig() { this(new InternalRuntimeSettings()); }

    public FlexlbConfig(InternalRuntimeSettings internalRuntime) {
        this.internalRuntime = java.util.Objects.requireNonNull(internalRuntime, "internalRuntime");
    }

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

    @JsonIgnore
    public long resolveExpiresAtMs(long startTime) {
        return isQueue() ? scheduler.resolveExpiresAtMs(startTime) : Long.MAX_VALUE;
    }

    @JsonIgnore
    public int defaultPriority() {
        return isPriorityOrdering() ? scheduler.getOrdering().getDefaultPriority() : 50;
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
    public SchedulerConfig queueScheduler() {
        checkState(isQueue(), "queue scheduler configuration is not active");
        return scheduler;
    }

    @JsonIgnore
    public QueueOrderingConfig priorityOrdering() {
        QueueOrderingConfig ordering = queueScheduler().getOrdering();
        checkState(ordering.getType() == QueueOrderingConfig.Type.PRIORITY,
                "priority ordering configuration is not active");
        return ordering;
    }

    @Getter
    @Setter
    public static final class GrpcServerConfig {
        private int executorCoreSize = 1000;
        private int executorMaxSize = 1000;
        private int executorQueueSize = 1000;
        /** 静默时间：下线后每次 Schedule 到达重新计时，期满后等待已接收 RPC 完成。 */
        private long shutdownQuietPeriodMs = 5_000L;
    }

}
