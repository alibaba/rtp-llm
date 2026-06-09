package org.flexlb.config;

import lombok.Getter;
import lombok.Setter;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.LoadBalanceStrategyEnum;
import org.flexlb.enums.ResourceMeasureIndicatorEnum;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

import static org.flexlb.enums.LoadBalanceStrategyEnum.RANDOM;
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
     * Maximum real requests in one EnqueueBatch request.
     */
    private int flexlbBatchSizeMax = 8;

    /**
     * Maximum time in milliseconds to wait for more requests before flushing a batch.
     */
    private long flexlbBatchWindowMs = 500;

    /**
     * Deadline in milliseconds for EnqueueBatch.
     */
    private long flexlbBatchEnqueueDeadlineMs = 5000;

    /**
     * How long completed enqueued entries remain cancellable through the master endpoint.
     */
    private long flexlbBatchInflightTtlMs = 3600L * 1000L;

    // ========== CostBasedPrefill Strategy Configuration ==========

    private long costSloMs = 500;

    private long costSloRiskMarginMs = 100;

    private String costSloBuckets = "";

    private transient volatile List<long[]> parsedSloBuckets;

    private double costHotspotMultiplier = 3.0;

    private double costImbalanceMultiplier = 3.0;

    private double costAlpha0 = 0;
    private double costAlpha1 = 1.0;
    private double costAlpha2 = 0;
    private double costAlpha3 = 0;
    private double costAlpha4 = 0.3;
    private double costAlpha5 = 0;

    /**
     * Comma-separated shorthand for the 6 predictor coefficients.
     * Accepts 3 values (α₀,α₁,α₂) or 6 values (α₀–α₅).
     * Overrides the individual costAlpha* fields when set.
     * Example: "290,0.0116,1.21e-8" or "290,0.0116,1.21e-8,1.21e-8,0,0"
     */
    public void setPrefillCoefficients(String csv) {
        if (csv == null || csv.isBlank()) {
            return;
        }
        String[] parts = csv.split(",");
        if (parts.length >= 3) {
            costAlpha0 = Double.parseDouble(parts[0].trim());
            costAlpha1 = Double.parseDouble(parts[1].trim());
            costAlpha2 = Double.parseDouble(parts[2].trim());
        }
        if (parts.length >= 6) {
            costAlpha3 = Double.parseDouble(parts[3].trim());
            costAlpha4 = Double.parseDouble(parts[4].trim());
            costAlpha5 = Double.parseDouble(parts[5].trim());
        } else if (parts.length >= 3) {
            costAlpha3 = 0;
            costAlpha4 = 0;
            costAlpha5 = 0;
        }
    }

    // ========== SLO-Budget Batcher Configuration ==========

    private double flexlbBatchFillThreshold = 0.5;

    private int flexlbBatchMaxCapacity = 1048576;

    private int flexlbBatchSearchIter = 10;

    private int flexlbBatchScanAhead = 64;

    /**
     * Maximum queue depth per WorkerBatcher. Requests beyond this limit are
     * rejected with QUEUE_FULL.
     */
    private int flexlbBatchQueueMaxSize = 64;

    /**
     * Maximum total in-flight requests across all batchers. Acts as a global
     * admission control gate at the FlexlbBatchScheduler entry.
     */
    private int flexlbBatchMaxInflight = 100000;

    // ========== gRPC Configuration ==========

    private long prefillLbTimeoutMs = 5000;

    private int flexlbGrpcPort = 7003;

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
}
