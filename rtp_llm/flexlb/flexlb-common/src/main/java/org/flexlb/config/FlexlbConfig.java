package org.flexlb.config;

import lombok.Getter;
import lombok.Setter;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.LoadBalanceStrategyEnum;
import org.flexlb.enums.ResourceMeasureIndicatorEnum;
import org.springframework.boot.context.properties.ConfigurationProperties;

import static org.flexlb.enums.LoadBalanceStrategyEnum.RANDOM;
import static org.flexlb.enums.LoadBalanceStrategyEnum.SHORTEST_TTFT;
import static org.flexlb.enums.LoadBalanceStrategyEnum.WEIGHTED_CACHE;
import static org.flexlb.enums.ResourceMeasureIndicatorEnum.REMAINING_KV_CACHE;
import static org.flexlb.enums.ResourceMeasureIndicatorEnum.WAIT_TIME;

/**
 * Bound from {@code flexlb.*} properties (application.yml, command-line args,
 * env vars via Spring relaxed binding) and from the legacy {@code FLEXLB_CONFIG}
 * JSON env var (merged via {@link FlexlbConfigJsonEnvironmentPostProcessor}).
 * Unprefixed per-field env vars (e.g. {@code ENABLE_QUEUEING}) layer on top via
 * {@link ConfigService#applyEnvironmentOverrides}.
 */
@Getter
@Setter
@ConfigurationProperties(prefix = "flexlb")
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
     * Weight decay factor, controls weight difference degree
     * Smaller value means smaller weight difference, larger value means more obvious weight difference
     * Recommended range: 0.001-0.01 (optimized for cache usage value range)
     */
    private double weightedCacheDecayFactor = 0.001;

    /**
     * Time window for recent cache-key hit ratio metrics in milliseconds.
     * Default is 30 minutes. Environment override: CACHE_HIT_TIME_WINDOW_MS.
     */
    private long cacheHitTimeWindowMs = 30L * 60L * 1000L;

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
     * Maximum in-flight requests per DECODE worker.
     * FlexLB counts reported waiting/running tasks plus local in-transit selections.
     * Values <= 0 disable the FlexLB-side decode concurrency limit.
     */
    private long decodeConcurrencyLimit = 0;

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

    // ========== V1-α DP Batching Configuration ==========

    /**
     * Master-side DP batching switch. When enabled, RouteService bypasses the global
     * QueueManager for DP-capable models and routes through DpBatchScheduler instead.
     * Default on — V1 DP-batching is the supported path. Set to false to fall back
     * to the legacy V0 router.
     */
    private boolean dpBalanceEnabled = true;

    /**
     * Cache-aware scheduling kill switch. When false, FlexLB stops polling per-worker
     * KV cache state ({@code GrpcCacheStatusCheckRunner} submission is suppressed) and
     * both the batcher's cache-prefix enrichment and the dispatcher's group selection
     * fall back to no-cache paths (bucket 0, first-alive group). Set this when the
     * cache-status pipeline is unavailable or when tests want deterministic
     * round-robin behaviour without exercising cache-affinity code.
     */
    private boolean cacheAwareSchedulingEnabled = true;

    /**
     * Max requests per Prefill.Enqueue batch. 0 = auto-pick worker.dpSize so one batch
     * fills exactly one DP cycle. Only effective when dpBalanceEnabled=true.
     */
    private int dpBatchSizeMax = 0;

    /**
     * Batch window (ms). First request entering an empty global batcher starts the
     * timer; timeout flushes whatever has accumulated. Pick around one prefill step
     * duration.
     */
    private long dpBatchWindowMs = 30;

    /**
     * Per-request max wait (ms). Any single request waiting longer than this triggers
     * an immediate flush regardless of batch fill — protects low-QPS scenarios from
     * being permanently stalled by the DP barrier.
     */
    private long dpBatchTimeoutMs = 100;

    /**
     * dp_rank assignment strategy within a single batch. V1 only supports "RR"
     * (positional, the i-th request goes to slot {@code i % dpSize}). Cross-batch
     * fairness lives one level up in {@link #dpGroupSelector}, since each batch
     * is dispatched as a unit to one DP group.
     * V2 plans to add "LPT" / "CACHE_AWARE_LPT" for length-aware bin packing.
     */
    private String dpAssignStrategy = "RR";

    /**
     * Ordered traffic policy rules. A matched rule forces the whole request to a worker group.
     */
    private volatile TrafficPolicyConfig trafficPolicy = new TrafficPolicyConfig();

    /**
     * TTFT SLO target (ms). Used by the SLO-aware batch flush trigger to compute
     * how long a request can wait before its TTFT budget is consumed.
     */
    private long dpTtftSloMs = 500;

    /**
     * Safety margin subtracted from the SLO slack when computing the batch
     * deadline (ms). Guards against estimation inaccuracy.
     */
    private long dpSafeMarginMs = 50;

    /**
     * Minimum batch interval (ms). Even when estimated TTFT leaves plenty of
     * slack, never batch faster than this to avoid thrashing.
     */
    private long dpMinBatchIntervalMs = 10;

    /**
     * Maximum batch interval (ms). Even when estimated TTFT is tight, never
     * wait longer than this for a batch to fill.
     */
    private long dpMaxBatchIntervalMs = 100;

    /**
     * Bucket interval in tokens for grouping requests by compute_token_length.
     * Requests with similar effective compute land in the same bucket, producing
     * homogeneous batches that waste less time at the DP barrier.
     * 0 = disable bucketing (all requests share one bucket).
     */
    private int dpBucketIntervalTokens = 0;

    /**
     * Plug-point name for the {@code GroupSelector} consulted by
     * {@code DefaultDispatchPlanner} once per drained batch.
     * Selects the best DP group via cache-aware scoring; falls back to
     * round-robin when no cache keys are present.
     */
    private String dpGroupSelector = "CACHE_AWARE";

    // ========== SloBudgetBatcher (dpSize=1 single-DP path) ==========

    /**
     * Per-batch total compute-token cap used by {@code SloBudgetBatcher} when
     * dpSize == 1. The batcher greedily packs requests up to this many tokens
     * (subject to the SLO budget constraint) so a single prefill step keeps GPU
     * occupancy high without overflowing the kernel's max-batch token budget.
     */
    private int batchMaxTokens = 8192;

    /**
     * Safety margin (ms) shaved off the head request's TTFT deadline before
     * computing the batch budget. Guards against estimator inaccuracy + network
     * dispatch latency so batches still meet SLO under noisy predictions.
     */
    private long sloSafetyMargin = 50;

    /**
     * Bounded look-ahead distance for {@code SloBudgetBatcher}'s backward scan.
     * After the FIFO head is locked in, the batcher scans at most this many
     * additional requests to find smaller ones that fit the remaining budget,
     * bypassing head-of-line blocking from oversized requests without an
     * unbounded O(N) walk.
     */
    private int dpMaxScanAhead = 64;

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
