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
     */
    private int maxRetryCount = 100;

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
     * Safety margin (ms) shaved off the head request's TTFT deadline before
     * computing the batch budget. Guards against estimator inaccuracy + network
     * dispatch latency so batches still meet SLO under noisy predictions.
     */
    private long sloSafetyMargin = 50;

    /**
     * gRPC deadline (ms) for the {@code Master.BatchEnqueue} RPC sent to prefill
     * workers. Under heavy load a batch may take longer than the default to be
     * acknowledged; increase this when DEADLINE_EXCEEDED errors appear in logs.
     */
    private long dpBatchEnqueueDeadlineMs = 5000;

    /**
     * Bounded look-ahead distance for {@code SloBudgetBatcher}'s greedy fill scan.
     * After the EDF head is locked in, the batcher scans at most this many
     * additional requests to find ones that fit the remaining batch capacity.
     */
    private int dpMaxScanAhead = 64;

    /**
     * Fill-ratio threshold for dispatching a batch. The batcher dispatches when
     * {@code sumTokens / batchMaxTokens >= threshold}. Lower values send smaller
     * batches sooner; higher values wait longer for better GPU utilization.
     */
    private double batchFillThreshold = 0.7;

    /**
     * Maximum iterations for the binary search that computes the optimal batch
     * token capacity per iteration. 12 iterations yield ~1/4096 precision over
     * a [headTokens, batchMaxCapacity] range.
     */
    private int binarySearchMaxIter = 12;

    /**
     * Upper bound (in tokens) for the binary search range when computing the
     * dynamic batch capacity. Represents the theoretical maximum tokens a
     * single prefill step could handle.
     */
    private int batchMaxCapacity = 1_000_000;

    // ========== Prefill Profiling Configuration ==========

    /**
     * Enable startup profiling of prefill latency. FlexLB sends fake requests
     * with varying input lengths to a prefill worker and fits a polynomial
     * to replace the hardcoded prefill time formula.
     */
    private boolean prefillProfilingEnabled = true;

    /**
     * Comma-separated token lengths to probe during profiling.
     */
    private String prefillProfilingTokenLengths = "32,64,128,256,512,1024,2048";

    /**
     * Number of repeat measurements per token length.
     */
    private int prefillProfilingRepeats = 3;

    /**
     * Maximum time (ms) to wait for at least one PREFILL worker before
     * giving up on profiling and falling back to default coefficients.
     */
    private long prefillProfilingTimeoutMs = 30000;

    /**
     * Per-request gRPC deadline (seconds) for each profiling probe.
     * Cold-start requests (first inference after model load) may trigger
     * CUDA compilation and framework init, so this must accommodate that.
     */
    private int prefillProfilingRequestTimeoutSeconds = 120;

    /**
     * Manual override: comma-separated "c0,c1,c2" polynomial coefficients.
     * If non-empty, profiling is skipped and these values are used directly
     * for {@code T(n) = c0 + c1*n + c2*n²} (ms).
     */
    private String prefillCoefficients = "";

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
