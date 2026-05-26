package org.flexlb.config;

import lombok.Getter;
import lombok.Setter;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.LoadBalanceStrategyEnum;
import org.flexlb.enums.ResourceMeasureIndicatorEnum;

import static org.flexlb.enums.LoadBalanceStrategyEnum.RANDOM;
import static org.flexlb.enums.LoadBalanceStrategyEnum.ROUND_ROBIN;
import static org.flexlb.enums.LoadBalanceStrategyEnum.SHORTEST_TTFT;
import static org.flexlb.enums.LoadBalanceStrategyEnum.WEIGHTED_CACHE;
import static org.flexlb.enums.ResourceMeasureIndicatorEnum.REMAINING_KV_CACHE;
import static org.flexlb.enums.ResourceMeasureIndicatorEnum.WAIT_TIME;

/**
 * Supports environment variable override configuration
 * Environment variable naming rule: {FIELD_NAME_UPPER_SNAKE_CASE}
 * Example: enableQueueing -> ENABLE_QUEUEING
 */
@Getter
@Setter
public class FlexlbConfig {

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
     * Load balancing strategy used by the {@code /batch_schedule} endpoint, decoupled from
     * {@link #loadBalanceStrategy} which governs {@code /schedule}. Default {@code ROUND_ROBIN}
     * because batch dispatch's value lies in atomic per-cursor distribution — the only
     * {@link org.flexlb.balance.strategy.BatchLoadBalancer} implementation today. Operators
     * who keep e.g. {@code SHORTEST_TTFT} for {@code /schedule} no longer have to give it up
     * to enable {@code /batch_schedule}; the two endpoints honor their own strategies.
     *
     * <p>Per-role overrides are not exposed yet because all four roles share the same default
     * (RR is the only batch-capable strategy). Add per-role fields if a future batch-aware
     * strategy ever justifies them.
     */
    private LoadBalanceStrategyEnum batchLoadBalanceStrategy = LoadBalanceStrategyEnum.ROUND_ROBIN;
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
     * Maximum retry count for failed routing attempts.
     * When exceeded, the request is completed with an error instead of being re-queued.
     * Default 0 means unlimited retries (bounded by generateTimeout).
     */
    private int maxRetryCount = 0;

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

    /**
     * Get the {@code /batch_schedule} load balancing strategy for a role. Defaults to
     * {@link LoadBalanceStrategyEnum#ROUND_ROBIN} so the batch path works out of the box
     * regardless of how {@link #getStrategyForRoleType} is configured.
     *
     * @param roleType Role type
     * @return Load balancing strategy to use for this role on the batch path
     */
    public LoadBalanceStrategyEnum getBatchStrategyForRoleType(RoleType roleType) {
        // Per-role override hook — currently a single global field; extend with per-role fields
        // (e.g. decodeBatchLoadBalanceStrategy) if/when a future batch-capable strategy makes
        // role-specific tuning meaningful.
        return batchLoadBalanceStrategy != null ? batchLoadBalanceStrategy : ROUND_ROBIN;
    }

    /**
     * Get load balancing strategy for a role type
     * This method handles the logic of selecting the appropriate strategy based on role type and configuration
     *
     * @param roleType Role type
     * @return Load balancing strategy to use for this role
     */
    public LoadBalanceStrategyEnum getStrategyForRoleType(RoleType roleType) {
        switch (roleType) {
            case PDFUSION -> {
                return this.loadBalanceStrategy != null ? loadBalanceStrategy : SHORTEST_TTFT;
            }
            case PREFILL -> {
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
     * Get resource measure indicator for a role type
     * Returns configured value if exists, otherwise returns default from map
     *
     * @param roleType Role type
     * @return Resource measure indicator
     */
    public ResourceMeasureIndicatorEnum getResourceMeasureIndicator(RoleType roleType) {
        switch (roleType) {
            case PDFUSION -> {
                return WAIT_TIME;
            }
            case PREFILL -> {
                return WAIT_TIME;
            }
            case DECODE -> {
                return REMAINING_KV_CACHE;
            }
            case VIT -> {
                return WAIT_TIME;
            }
            default -> {
                return null;
            }
        }
    }
}
