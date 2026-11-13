package org.flexlb.config;

import com.fasterxml.jackson.annotation.JsonIgnore;
import lombok.Getter;
import lombok.Setter;
import org.flexlb.config.cache.CacheMatchingConfig;
import org.flexlb.config.cache.KvcmCacheMatchingConfig;
import org.flexlb.config.cache.LocalSyncCacheMatchingConfig;
import org.flexlb.config.loading.ConfigSchemaVersion;
import org.flexlb.config.master.ConsistencyConfig;
import org.flexlb.config.master.NoConsistencyConfig;
import org.flexlb.config.master.ZookeeperConsistencyConfig;
import org.flexlb.config.monitoring.ObservabilityConfig;
import org.flexlb.config.routing.OptimizerRuntimeConfig;
import org.flexlb.config.routing.RoutingConfig;
import org.flexlb.config.scheduler.PreemptionStage;
import org.flexlb.config.scheduler.QueueOrderingConfig;
import org.flexlb.config.scheduler.RequestLifecycleConfig;
import org.flexlb.config.scheduler.SchedulerConfig;
import org.flexlb.config.worker.WorkerRegistryConfig;
import org.flexlb.util.discovery.DiscoveryClientSettings;

import static com.google.common.base.Preconditions.checkState;

/**
 * Public FLEXLB_CONFIG contract, organized by stable responsibility owner.
 * Mutually exclusive behavior is represented by tagged unions so inactive
 * variant fields fail during deserialization.
 */
@Getter
@Setter
public final class FlexlbConfig {
    public static final int CURRENT_SCHEMA_VERSION = ConfigSchemaVersion.V4;

    private int schemaVersion = CURRENT_SCHEMA_VERSION;

    private SchedulerConfig scheduler = new SchedulerConfig();
    private RequestLifecycleConfig requestLifecycle = new RequestLifecycleConfig();
    private RoutingConfig router = new RoutingConfig();
    private WorkerRegistryConfig workerRegistry = new WorkerRegistryConfig();
    private ObservabilityConfig observability = new ObservabilityConfig();
    private GrpcServerConfig grpcServer = new GrpcServerConfig();
    private DiscoveryClientSettings serviceDiscovery = new DiscoveryClientSettings();
    private CacheMatchingConfig cacheMatching = new LocalSyncCacheMatchingConfig();
    private OptimizerRuntimeConfig optimizer = new OptimizerRuntimeConfig();
    private ConsistencyConfig consistency = new NoConsistencyConfig();

    private boolean enableFallback;

    @JsonIgnore private final InternalRuntimeSettings internalRuntime;

    public FlexlbConfig() {
        this(new InternalRuntimeSettings());
    }

    public FlexlbConfig(InternalRuntimeSettings internalRuntime) {
        this.internalRuntime = java.util.Objects.requireNonNull(internalRuntime, "internalRuntime");
    }

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
    public boolean allowsPreemption(PreemptionStage stage) {
        return isPriorityOrdering()
                && scheduler.getOrdering().getPreemption() != null
                && scheduler.getOrdering().getPreemption().allows(stage);
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
    public long resolveExpiresAtMs(long startTime) {
        return isQueue() ? scheduler.resolveExpiresAtMs(startTime) : Long.MAX_VALUE;
    }

    @JsonIgnore
    public int defaultPriority() {
        return isPriorityOrdering() ? scheduler.getOrdering().getDefaultPriority() : 50;
    }

    @JsonIgnore
    public QueueOrderingConfig priorityOrdering() {
        QueueOrderingConfig ordering = queueScheduler().getOrdering();
        checkState(
                ordering.getType() == QueueOrderingConfig.Type.PRIORITY,
                "priority ordering configuration is not active");
        return ordering;
    }

    @JsonIgnore
    public SchedulerConfig queueScheduler() {
        checkState(isQueue(), "queue scheduler configuration is not active");
        return scheduler;
    }

    /**
     * Configures gRPC executors, receiving request limits and receive-buffer budgets.
     */
    @Getter
    @Setter
    public static final class GrpcServerConfig {
        public static final int DEFAULT_MAX_RECEIVING_REQUESTS = 1024;
        public static final int DEFAULT_MAX_RECEIVING_CONTROL_REQUESTS = 128;
        private int executorCoreSize = 1000;
        private int executorMaxSize = 1000;
        private int executorQueueSize = 1000;

        /**
         * Zero derives Direct / 4 as a logical receiving wire-byte budget at startup.
         */
        private long receiveBudgetBytes = 0L;

        /**
         * Concurrent Schedule receiving calls; zero uses the fixed default.
         */
        private int maxReceivingRequests = DEFAULT_MAX_RECEIVING_REQUESTS;

        /**
         * Independent concurrent Cancel receiving calls; zero uses the fixed default.
         */
        private int maxReceivingControlRequests = DEFAULT_MAX_RECEIVING_CONTROL_REQUESTS;

        /**
         * Headers to decoded half-close; does not limit accepted scheduler queue time.
         */
        private long receiveTimeoutMs = 30_000L;

        /**
         * 静默时间：下线后每次 Schedule 到达重新计时，期满后等待已接收 RPC 完成。
         */
        private long shutdownQuietPeriodMs = 5_000L;
    }
}
