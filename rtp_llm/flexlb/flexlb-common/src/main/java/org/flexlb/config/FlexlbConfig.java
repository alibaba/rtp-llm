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
    private EngineGrpcConfig engineGrpc = new EngineGrpcConfig();
    private ServiceDiscoveryRuntimeConfig serviceDiscovery = new ServiceDiscoveryRuntimeConfig();
    private CacheMatchingConfig cacheMatching = new LocalSyncCacheMatchingConfig();
    private OptimizerRuntimeConfig optimizer = new OptimizerRuntimeConfig();
    private ConsistencyConfig consistency = new NoConsistencyConfig();

    private boolean enableFallback;
    private long fallbackBatchTokenCapacity = 1_048_576L;

    @JsonIgnore
    private final InternalRuntimeSettings internalRuntime = new InternalRuntimeSettings();

    @JsonIgnore
    public boolean isDirect() {
        return scheduler.getType() == SchedulerConfig.Type.DIRECT;
    }

    @JsonIgnore
    public boolean isKvcmCacheMatching() {
        return cacheMatching instanceof KvcmCacheMatchingConfig;
    }

    @JsonIgnore
    public KvcmCacheMatchingConfig kvcmCacheMatching() {
        if (cacheMatching instanceof KvcmCacheMatchingConfig kvcm) {
            return kvcm;
        }
        throw new IllegalStateException("KVCM cache matching configuration is not active");
    }

    @JsonIgnore
    public boolean isConsistencyEnabled() {
        return consistency instanceof ZookeeperConsistencyConfig;
    }

    @JsonIgnore
    public boolean isBatchDispatch() {
        return dispatcher.getType() == DispatcherConfig.Type.BATCH;
    }

    @JsonIgnore
    public boolean isPriorityOrdering() {
        return isQueue() && scheduler.getOrdering().getType() == QueueOrderingConfig.Type.PRIORITY;
    }

    @JsonIgnore
    public boolean isQueue() {
        return scheduler.getType() == SchedulerConfig.Type.QUEUE;
    }

    @JsonIgnore
    public boolean isSingleDecision() {
        return isQueue()
                && queueScheduler().getDecision().getType() == DecisionPolicyConfig.Type.SINGLE;
    }

    @JsonIgnore
    public SchedulerConfig queueScheduler() {
        if (isQueue()) {
            return scheduler;
        }
        throw new IllegalStateException("queue scheduler configuration is not active");
    }

    @JsonIgnore
    public boolean isFixedWindowDecision() {
        return isQueue()
                && queueScheduler().getDecision().getType()
                        == DecisionPolicyConfig.Type.FIXED_WINDOW;
    }

    @JsonIgnore
    public DecisionPolicyConfig fixedWindowDecision() {
        DecisionPolicyConfig policy = decisionPolicy();
        if (policy.getType() == DecisionPolicyConfig.Type.FIXED_WINDOW) {
            return policy;
        }
        throw new IllegalStateException("fixed-window decision configuration is not active");
    }

    /** Resolve the QUEUE decision policy from its single configuration owner. */
    @JsonIgnore
    public DecisionPolicyConfig decisionPolicy() {
        return queueScheduler().getDecision();
    }

    @JsonIgnore
    public QueueOrderingConfig priorityOrdering() {
        QueueOrderingConfig ordering = queueScheduler().getOrdering();
        if (ordering.getType() == QueueOrderingConfig.Type.PRIORITY) {
            return ordering;
        }
        throw new IllegalStateException("priority ordering configuration is not active");
    }

    /**
     * Sets serving executor capacities, shutdown timing, and HTTP/2 keepalive policy.
     */
    @Getter
    @Setter
    public static final class GrpcServerConfig {
        public static final int DEFAULT_MAX_RECEIVING_REQUESTS = 1024;
        public static final int DEFAULT_MAX_RECEIVING_CONTROL_REQUESTS = 128;

        private int executorCoreSize = 1000;
        private int executorMaxSize = 1000;
        private int executorQueueSize = 1000;
        private int maxReceivingRequests = DEFAULT_MAX_RECEIVING_REQUESTS;
        private int maxReceivingControlRequests = DEFAULT_MAX_RECEIVING_CONTROL_REQUESTS;

        /** 静默时间：下线后每次 Schedule 到达重新计时，期满后等待已接收 RPC 完成。 */
        private long shutdownQuietPeriodMs = 5_000L;

        private long permitKeepAliveTimeMs = 10_000L;
        private long receiveTimeoutMs = 30_000L;
        private long receiveBudgetBytes;
        private boolean permitKeepAliveWithoutCalls = true;
    }

    /**
     * Configures Engine gRPC callbacks, shared client I/O and the KV cache snapshot receive limit.
     */
    @Getter
    @Setter
    public static final class EngineGrpcConfig {
        private int executorThreads = 32;
        private int executorQueueCapacity = 10_000;
        private int eventLoopThreads = 8;
        private int kvCacheMaxInboundMessageBytes = 128 * 1024 * 1024;
    }
}
