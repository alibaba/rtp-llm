package org.flexlb.config;

import lombok.Getter;
import lombok.Setter;
import org.flexlb.dao.route.RoleType;
import org.flexlb.dao.route.ServiceRoute;
import org.flexlb.enums.BlockHashStrategyType;
import org.flexlb.enums.LoadBalanceStrategyEnum;
import org.flexlb.enums.LogLevel;
import org.flexlb.enums.ResourceMeasureIndicatorEnum;

import static org.flexlb.enums.LoadBalanceStrategyEnum.RANDOM;
import static org.flexlb.enums.LoadBalanceStrategyEnum.SHORTEST_TTFT;
import static org.flexlb.enums.LoadBalanceStrategyEnum.WEIGHTED_CACHE;
import static org.flexlb.enums.ResourceMeasureIndicatorEnum.REMAINING_KV_CACHE;
import static org.flexlb.enums.ResourceMeasureIndicatorEnum.WAIT_TIME;

/**
 * Supports environment variable and Nacos override configuration
 * Environment variable naming rule: {FIELD_NAME_UPPER_SNAKE_CASE}
 * Example: enableQueueing -> ENABLE_QUEUEING
 */
@Getter
@Setter
public class FlexlbConfig {

    /**
     * Model routing, service discovery, KVCM, and optimizer configuration.
     */
    private ServiceRoute modelServiceConfig;

    /**
     * Log level for the FlexLB logging group.
     */
    private LogLevel flexlbLogLevel = LogLevel.INFO;

    /**
     * Whether root and PV logs are also written to container stdout.
     */
    private boolean enableStdoutLog = false;

    /**
     * Load balancer status consistency and master election configuration.
     */
    private LBConsistencyConfig flexlbSyncConsistencyConfig = new LBConsistencyConfig();

    /**
     * Block hash strategy used for cache matching.
     */
    private BlockHashStrategyType blockHashStrategy = BlockHashStrategyType.VLLM;

    /**
     * Load balancing strategy
     */
    private LoadBalanceStrategyEnum loadBalanceStrategy = LoadBalanceStrategyEnum.SHORTEST_TTFT;

    /**
     * Load balancing strategy for DECODE role
     */
    private LoadBalanceStrategyEnum decodeLoadBalanceStrategy = LoadBalanceStrategyEnum.WEIGHTED_CACHE;

    /**
     * Load balancing strategy for VIT role
     */
    private LoadBalanceStrategyEnum vitLoadBalanceStrategy = LoadBalanceStrategyEnum.RANDOM;
    /**
     * Weight decay factor, controls weight difference degree
     * Smaller value means smaller weight difference, larger value means more obvious weight difference
     * Recommended range: 0.001-0.01 (optimized for cache usage value range)
     */
    private double weightedCacheDecayFactor = 0.001;

    // ========== Queue Configuration ==========

    /**
     * Whether to enable queuing
     */
    private boolean enableQueueing = false;

    /**
     * Whether FlexLB should skip routing and instruct the caller to fall back.
     */
    private boolean enableFallback = false;

    /**
     * Maximum queue length per model
     */
    private int maxQueueSize = 1000000;

    /**
     * Maximum retry count for failed routing attempts.
     * When exceeded, the request is completed with an error instead of being re-queued.
     * Default 0 means unlimited retries (bounded by generateTimeout).
     */
    private int maxRetryCount = 0;

    /**
     * Delay between retryable routing attempts.
     * Keeps a request in the scheduling worker while engines are full instead of repeatedly
     * moving it in and out of the main queue.
     */
    private long routingRetryIntervalMs = 10;

    /**
     * Maximum time to retain a locally tracked task while waiting for WorkerStatus confirmation.
     */
    private long taskConfirmTimeoutMs = 300_000;

    /**
     * Prefill queue threshold used by pre-strategy resource filtering.
     *
     * <p>Once a worker reaches this value, it is hidden from the balancing strategy, so the
     * strategy cannot consider it or record its decision state. Prefer
     * {@link #outstandingUncachedTokensThreshold} for strategy-level routing control.
     */
    private long prefillQueueSizeThreshold = 1024;

    /**
     * Credit applied to cache blocks made available through one P2P fetch.
     *
     * <p>Local cache hits retain full credit. A value of 0.2 treats a P2P-added block as one
     * fifth of a local hit when estimating routing work.
     */
    private double p2pHitDiscount = 0.2;

    /**
     * Relative TTFT difference allowed when treating shortest-TTFT candidates as similar.
     * A value of 0.2 allows candidates within 20% of the minimum estimated TTFT.
     */
    private double shortestTtftSimilarityThresholdRatio = 0.2;

    /**
     * Maximum additional uncached prefill work allowed by CACHE_AFFINITY_FIRST when preferring
     * a worker for cache affinity.
     */
    private long cacheAffinityFirstMaxExtraWorkTokens = 0;

    /**
     * Outstanding uncached-token threshold shared by SHORTEST_TTFT and
     * CACHE_AFFINITY_FIRST. When absent, CACHE_AFFINITY_FIRST can use its deprecated
     * compatibility setting, while SHORTEST_TTFT leaves this protection disabled.
     */
    private Long outstandingUncachedTokensThreshold;

    /**
     * @deprecated Use {@link #outstandingUncachedTokensThreshold}.
     */
    @Deprecated
    private long cacheAffinityFirstOutstandingUncachedTokensThreshold = 0;

    /**
     * Minimum effective cache-hit percentage needed before CACHE_AFFINITY_FIRST may prefer cache.
     * A value of 0 disables this gate; 5 means 5%.
     */
    private double cacheAffinityFirstMinHitRate = 5;

    /**
     * KV cache available threshold for DECODE role (percentage)
     * When Worker's KV cache usage is below this threshold, the Worker is considered available
     * Range: 1-100, default 90 means Worker is unavailable when usage exceeds 90%
     */
    private long decodeAvailableMemoryThreshold = 90;

    /**
     * Resource availability hysteresis bias (percentage)
     * Used to prevent frequent switching of resource availability near threshold
     * Range: 0-100, default 15 means hysteresis range is 15%
     */
    private long hysteresisBiasPercent = 15;

    // ========== Worker Thread Pool Configuration ==========

    /**
     * Number of scheduling workers (default CPU core count)
     */
    private int scheduleWorkerSize = Runtime.getRuntime().availableProcessors();

    /**
     * Whether scheduling permits stay fixed at {@code scheduleWorkerSize}.
     * When enabled, resource water levels are observed but do not reduce scheduling concurrency.
     */
    private boolean fixedScheduleWorkerPermits = false;

    /**
     * Resource availability check interval (milliseconds, default 10ms)
     */
    private long resourceCheckIntervalMs = 10;

    // ========== Worker Status Synchronization Configuration ==========

    /**
     * Worker status synchronization interval in milliseconds.
     */
    private long syncStatusInterval = 20;

    /**
     * Worker status synchronization request timeout in milliseconds.
     */
    private long syncRequestTimeoutMs = 200;

    // ========== Resource Water Level Configuration ==========

    /**
     * Decode full speed threshold (used memory percentage)
     * When used memory is below this threshold, water level is 0 (full speed)
     * Default 40 means full speed when used memory < 40%
     */
    private long decodeFullSpeedThreshold = 40;

    /**
     * Decode stop threshold (used memory percentage)
     * When used memory is above this threshold, water level is 100 (stop)
     * Default 80 means stop when used memory > 80%
     */
    private long decodeStopThreshold = 80;

    // ========== Netty Thread Pool Configuration ==========

    /**
     * Netty select thread multiplier (default 1)
     * Actual select threads = availableProcessors * nettySelectThreadMultiplier
     */
    private int nettySelectThreadMultiplier = 1;

    /**
     * Netty worker thread multiplier (default 2)
     * Actual worker threads = availableProcessors * nettyWorkerThreadMultiplier
     */
    private int nettyWorkerThreadMultiplier = 2;

    /**
     * Get load balancing strategy for a role type
     * This method handles the logic of selecting the appropriate strategy based on role type and configuration
     *
     * @param roleType Role type
     * @return Load balancing strategy to use for this role
     */
    public LoadBalanceStrategyEnum getStrategyForRoleType(RoleType roleType) {
        switch (roleType) {
            case PDFUSION, PREFILL -> {
                return this.loadBalanceStrategy != null ? loadBalanceStrategy : SHORTEST_TTFT;
            }
            case DECODE -> {
                return this.decodeLoadBalanceStrategy != null ? decodeLoadBalanceStrategy : WEIGHTED_CACHE;
            }
            case VIT -> {
                return this.vitLoadBalanceStrategy != null ? vitLoadBalanceStrategy : RANDOM;
            }
            default -> {
                return null;
            }
        }
    }

    /**
     * Gets the outstanding uncached-token threshold for a balancing strategy.
     *
     * <p>The neutral configuration applies to both strategies. When it is absent, only
     * CACHE_AFFINITY_FIRST reads the deprecated cache-affinity compatibility value.
     */
    public long getEffectiveOutstandingUncachedTokensThreshold(LoadBalanceStrategyEnum strategy) {
        if (outstandingUncachedTokensThreshold != null) {
            return Math.max(0L, outstandingUncachedTokensThreshold);
        }
        if (strategy == LoadBalanceStrategyEnum.CACHE_AFFINITY_FIRST) {
            return Math.max(0L, cacheAffinityFirstOutstandingUncachedTokensThreshold);
        }
        return 0L;
    }

    /**
     * Get resource measure indicator for a role type
     * Returns configured value if exists, otherwise returns default from map
     *
     * @param roleType Role type
     * @return Resource measure indicator
     */
    public ResourceMeasureIndicatorEnum getResourceMeasureIndicator(RoleType roleType) {
        switch (roleType) {
            case PDFUSION, PREFILL, VIT -> {
                return WAIT_TIME;
            }
            case DECODE -> {
                return REMAINING_KV_CACHE;
            }
            default -> {
                return null;
            }
        }
    }
}
