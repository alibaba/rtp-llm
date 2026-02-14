package org.flexlb.config;

import lombok.Getter;
import lombok.Setter;
import org.flexlb.enums.LoadBalanceStrategyEnum;

/**
 * Supports environment variable override configuration
 * Environment variable naming rule: {FIELD_NAME_UPPER_SNAKE_CASE}
 * Example: enableQueueing -> ENABLE_QUEUEING
 */
@Getter
@Setter
public class WhaleMasterConfig {

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
     * Maximum queue length per model
     */
    private int maxQueueSize = 1000000;

    /**
     * Prefill role queuing threshold
     * When below this threshold, the Worker is considered available
     */
    private long prefillQueueSizeThreshold = 3;

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
     * Resource availability check interval (milliseconds, default 10ms)
     */
    private long resourceCheckIntervalMs = 10;

    /**
     * Prefill maximum queue size
     */
    private int maxPrefillQueueSize = 20;

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
}
