package org.flexlb.config;

import lombok.Getter;
import lombok.Setter;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.LoadBalanceStrategyEnum;
import org.flexlb.enums.ResourceMeasureIndicatorEnum;
import org.flexlb.enums.ScheduleModeEnum;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

import static org.flexlb.enums.LoadBalanceStrategyEnum.COST_BASED_DECODE;
import static org.flexlb.enums.LoadBalanceStrategyEnum.COST_BASED_PREFILL;
import static org.flexlb.enums.LoadBalanceStrategyEnum.RANDOM;
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
    private LoadBalanceStrategyEnum loadBalanceStrategy = LoadBalanceStrategyEnum.COST_BASED_PREFILL;

    /**
     * Load balancing strategy for DECODE role
     */
    private LoadBalanceStrategyEnum decodeLoadBalanceStrategy = LoadBalanceStrategyEnum.COST_BASED_DECODE;

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

    /**
     * Maximum cache-key occurrences retained by the recent cache-key pool.
     * Environment override: CACHE_HIT_MAX_CACHE_KEYS.
     */
    private long cacheHitMaxCacheKeys = 10_000_000L;

    /**
     * Whether Master writes successful requests into the recent cache-key window.
     * Environment override: CACHE_HIT_WINDOW_WRITE_ENABLED.
     */
    private boolean cacheHitWindowWriteEnabled = true;

    /**
     * Whether Master reports recent cache-key hit/total metrics.
     * Environment override: CACHE_HIT_METRIC_REPORT_ENABLED.
     */
    private boolean cacheHitMetricReportEnabled = true;

    /**
     * Whether Master logs per-request recent cache-key hit trace.
     * Environment override: CACHE_HIT_TRACE_LOG_ENABLED.
     */
    private boolean cacheHitTraceLogEnabled = false;

    /**
     * Whether Master writes aggregated theory hit counters to master_theory_hit.log.
     * Environment override: CACHE_HIT_THEORY_LOG_ENABLED.
     */
    private boolean cacheHitTheoryLogEnabled = true;

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
    private long prefillQueueSizeThreshold = 64;

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

    /**
     * Ordered traffic policy rules. A matched rule forces the whole request to a worker group.
     */
    private volatile TrafficPolicyConfig trafficPolicy = new TrafficPolicyConfig();

    // ========== FlexLB Batch Configuration ==========

    /**
     * Enables master-side request coalescing. Requests carrying a full
     * GenerateInputPB are routed once, grouped by Prefill worker,
     * and submitted through EnqueueBatch.
     */
    private boolean flexlbBatchEnabled = true;

    /**
     * Default schedule mode when the frontend doesn't specify one in the gRPC request.
     * Environment variable: DEFAULT_SCHEDULE_MODE (values: AUTO, BATCH, DIRECT).
     */
    private String defaultScheduleMode = "AUTO";

    /**
     * Maximum real requests in one EnqueueBatch request.
     */
    private int flexlbBatchSizeMax = 8;

    /**
     * Remaining-budget window in milliseconds. Outside this window the batcher
     * keeps collecting unless the batch reaches flexlbBatchSizeMax. Inside this
     * window it can dispatch once the batch has enough requests and another
     * arrival is unlikely before the latest safe dispatch point.
     */
    private long flexlbBatchWindowMs = 300;

    /**
     * Minimum useful batch size. This is not a hard immediate-dispatch trigger:
     * the batcher may keep waiting if the remaining SLO slack can likely buy
     * one more request.
     */
    private int flexlbBatchMinSize = 3;

    /**
     * Upper bound for deadline-protection dispatch. The effective guard is
     * min(flexlbBatchEmergencyBudgetMs, incrementalBatchCost + flexlbBatchDispatchGuardMs).
     */
    private long flexlbBatchEmergencyBudgetMs = 150;

    /**
     * Safety guard left before the computed SLO deadline when dispatching a batch.
     * Covers master loop jitter, gRPC enqueue overhead, and predictor error.
     */
    private long flexlbBatchDispatchGuardMs = 40;

    /**
     * EMA alpha used to estimate per-worker request inter-arrival time for batching.
     */
    private double flexlbBatchArrivalEmaAlpha = 0.2;

    /**
     * Extra slack that must remain after the next expected request arrival before
     * the latest safe dispatch point. Larger values dispatch earlier and reduce
     * deadline pressure; smaller values favor bigger batches.
     */
    private long flexlbBatchArrivalWaitGuardMs = 20;

    /**
     * Maximum in-flight prefill batches allowed per worker before the batcher
     * stops dispatching new batches and keeps requests in the master-side queue.
     * Values <= 0 disable this backpressure gate.
     */
    private int flexlbBatchSloMaxInflightBatches = 2;

    /**
     * Maximum in-flight prefill batches per worker for the fixed_window batcher.
     * When the engine already has this many batches inflight, the batcher parks
     * instead of dispatching new batches.  Default 0 disables backpressure —
     * the fixed_window batcher dispatches regardless of engine load.
     *
     * <p>Set to a small value (e.g. 2–3) to prevent engine overload when
     * using fixed_window; set to 0 to keep the original always-dispatch behavior.
     */
    private int flexlbBatchFixedMaxInflightBatches = 0;

    /**
     * Deadline in milliseconds for EnqueueBatch.
     */
    private long flexlbBatchEnqueueDeadlineMs = 5000;

    /**
     * TTL for inflight entries before eviction (used by all routing paths).
     * Only a safety net — calibrate() cleans up normally.  This catches stale
     * entries left by engine crashes, lost status reports, or bugs.
     * 5 min is generous for network/engine-report jitter but short enough
     * that stale inflight won't distort realWaitTimeMs for long.
     */
    private long flexlbInflightTtlMs = 300_000L;

    /**
     * Maximum threads in the batch dispatch executor pool.
     */
    private int flexlbBatchDispatchPoolSize = 64;

    /**
     * Maximum pending tasks in the batch dispatch executor queue.
     * Tasks submitted when both the pool and queue are full are rejected
     * and fail immediately with QUEUE_FULL.
     */
    private int flexlbBatchDispatchQueueSize = 256;

    // ========== CostBasedPrefill Strategy Configuration ==========

    /**
     * Whether to enable SLO time-budget hard filter during prefill worker selection.
     * When enabled, workers whose (waitMs + predictedPrefillMs) exceeds
     * (SLO - riskMargin) are excluded. Default false because the filter is
     * too aggressive in practice.
     */
    private boolean costSloFilterEnabled = false;

    private long costSloMs = 500;

    private long costSloRiskMarginMs = 100;

    private String costSloBuckets = "";

    private transient volatile List<long[]> parsedSloBuckets;

    public void setCostSloBuckets(String costSloBuckets) {
        this.costSloBuckets = costSloBuckets;
        this.parsedSloBuckets = null;
    }

    private double costHotspotMultiplier = 3.0;

    private double costImbalanceMultiplier = 3.0;

    /**
     * Configurable prefill-time prediction formula.
     *
     * <p>Supported variables: {@code c, p, sum_c, sum_c2, sum_cp, sum_p, n}
     * <br>Operators: {@code + - * / ^}
     * <br>Functions: {@code sqrt(x) log(x) exp(x) abs(x) max(a,b) min(a,b) pow(a,b)}
     *
     * <p>Example: {@code "205 + 1.2e-8*sum_c2 + 1.2e-8*sum_cp + 5*n"}
     */
    private String costFormula = "sum_c + 0.3*sum_p";

    // ========== SLO-Budget Batcher Configuration ==========

    private double flexlbBatchFillThreshold = 0.5;

    private int flexlbBatchMaxCapacity = 1048576;

    private int flexlbBatchSearchIter = 10;

    private int flexlbBatchScanAhead = 64;

    /**
     * Maximum queue depth per WorkerBatcher. Requests beyond this limit are
     * rejected with QUEUE_FULL.
     */
    private int flexlbBatchQueueMaxSize = 1024;

    /**
     * Maximum total in-flight requests across all batchers. Acts as a global
     * admission control gate at the FlexlbBatchScheduler entry.
     */
    private int flexlbBatchMaxInflight = 100000;

    // ========== Batcher Algorithm Selection ==========

    /**
     * Batcher algorithm name. Supported values:
     * <ul>
     *   <li>{@code fixed_window} — Fixed time window batching with optional
     *       predictor-based early dispatch. No SLO deadline tracking, no EMA,
     *       no request dropping (default).</li>
     *   <li>{@code slo_budget} — SLO-deadline-aware batching with EMA arrival
     *       rate estimation, budget-based greedy fill, and deadline-gated dispatch.</li>
     * </ul>
     */
    private String flexlbBatchAlgorithm = "fixed_window";

    /**
     * Fixed wait time in milliseconds for the {@code fixed_window} batcher
     * algorithm. After a request has waited this long, the batcher dispatches
     * whatever has accumulated regardless of batch size.
     *
     * <p>Only used when {@link #flexlbBatchAlgorithm} is {@code fixed_window}.
     */
    private long flexlbBatchFixedWaitMs = 300;

    /**
     * Predicted batch execution time threshold in milliseconds for the
     * {@code fixed_window} batcher algorithm. If the predictor estimates
     * the accumulated batch will take at least this long, the batcher
     * dispatches immediately rather than waiting for {@link #flexlbBatchFixedWaitMs}.
     *
     * <p>Set to 0 to disable predictor-based early dispatch (default).
     * Only used when {@link #flexlbBatchAlgorithm} is {@code fixed_window}.
     */
    private long flexlbBatchPredictThresholdMs = 0;

    // ========== gRPC Configuration ==========

    private long prefillLbTimeoutMs = 5000;

    // ========== Decode Load Balance Hard Filter Configuration ==========

    private double decodeHotspotMultiplier = 3.0;

    private double decodeImbalanceMultiplier = 3.0;

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
                return this.loadBalanceStrategy != null ? loadBalanceStrategy : COST_BASED_PREFILL;
            }
            case PREFILL -> {
                return this.loadBalanceStrategy != null ? loadBalanceStrategy : COST_BASED_PREFILL;
            }
            case DECODE -> {
                return this.decodeLoadBalanceStrategy != null ? decodeLoadBalanceStrategy : COST_BASED_DECODE;
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

    public long resolveSloMs(long seqLen) {
        List<long[]> buckets = getParsedSloBuckets();
        if (buckets == null || buckets.isEmpty()) {
            return costSloMs;
        }
        for (long[] bucket : buckets) {
            if (seqLen <= bucket[0]) {
                return bucket[1];
            }
        }
        return buckets.get(buckets.size() - 1)[1];
    }

    private List<long[]> getParsedSloBuckets() {
        if (parsedSloBuckets != null) {
            return parsedSloBuckets;
        }
        if (costSloBuckets == null || costSloBuckets.isBlank()) {
            return null;
        }
        List<long[]> result = new ArrayList<>();
        for (String entry : costSloBuckets.split(",")) {
            String[] kv = entry.trim().split(":");
            if (kv.length == 2) {
                try {
                    result.add(new long[]{Long.parseLong(kv[0].trim()), Long.parseLong(kv[1].trim())});
                } catch (NumberFormatException ignored) {
                }
            }
        }
        result.sort(Comparator.comparingLong(a -> a[0]));
        parsedSloBuckets = result;
        return result;
    }

    /**
     * Returns the configured default schedule mode as an enum.
     */
    public ScheduleModeEnum getDefaultScheduleModeEnum() {
        try {
            return ScheduleModeEnum.valueOf(defaultScheduleMode.toUpperCase());
        } catch (IllegalArgumentException e) {
            return ScheduleModeEnum.AUTO;
        }
    }
}
