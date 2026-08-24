package org.flexlb.config;

import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.LoadBalanceStrategyEnum;
import org.flexlb.enums.ResourceMeasureIndicatorEnum;
import org.flexlb.enums.ScheduleModeEnum;

import com.fasterxml.jackson.annotation.JsonAlias;
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
 * Example: defaultScheduleMode -> DEFAULT_SCHEDULE_MODE
 */
@Getter
@Setter
@Slf4j
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

    // ========== Queue Configuration ==========

    /**
     * Maximum queue capacity for the {@code QueueingComponent} (DIRECT/QUEUE
     * routing path). Requests that cannot be routed immediately (no available
     * worker) enter this bounded FIFO queue.
     *
     * <p><b>Disambiguation:</b> This is distinct from:
     * <ul>
     *   <li>{@link #flexlbBatchQueueMaxSize} — per-WorkerBatcher queue depth
     *       limit (BATCH routing path)</li>
     *   <li>{@link #flexlbBatchMaxCapacity} — total capacity across all
     *       batchers (global admission gate)</li>
     *   <li>{@link #maxPrefillQueueSize} — prefill resource water-level
     *       threshold (not a queue capacity; used for availability scoring)</li>
     * </ul>
     *
     * <p>JSON alias {@code "maxQueueSize"} is kept for backward compatibility
     * with existing deployment configs.
     * Environment variable: QUEUEING_COMPONENT_QUEUE_MAX_SIZE
     * (legacy: MAX_QUEUE_SIZE, still accepted for backward compat)
     */
    @JsonAlias({"maxQueueSize"})
    private int queueingComponentQueueMaxSize = 1000000;

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
     * Resource availability check interval (milliseconds, default 1000ms)
     */
    private long resourceCheckIntervalMs = 1000;

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

    // ========== Thread Pool Size Configuration ==========

    /**
     * gRPC Client Executor core pool size.
     * Environment variable: GRPC_CLIENT_EXECUTOR_CORE_SIZE
     */
    private int grpcClientExecutorCoreSize = 32;

    /**
     * gRPC Client Executor max pool size.
     * Set equal to core size because LinkedBlockingQueue prevents thread pool
     * expansion until the queue is full. With a large queue (10000), maxPoolSize
     * would be effectively unreachable, so we keep the pool fixed at core size.
     * Environment variable: GRPC_CLIENT_EXECUTOR_MAX_SIZE
     */
    private int grpcClientExecutorMaxSize = 32;

    /**
     * gRPC Client Executor bounded queue size.
     * Environment variable: GRPC_CLIENT_EXECUTOR_QUEUE_SIZE
     */
    private int grpcClientExecutorQueueSize = 10000;

    /**
     * gRPC Client EventLoopGroup thread count.
     * Environment variable: GRPC_CLIENT_EVENT_LOOP_THREADS
     */
    private int grpcClientEventLoopThreads = 8;

    /**
     * gRPC Server Worker EventLoopGroup thread count.
     * Environment variable: GRPC_SERVER_WORKER_EVENT_LOOP_THREADS
     */
    private int grpcServerWorkerEventLoopThreads = 4;

    /**
     * HTTP Netty NioEventLoopGroup thread count.
     * Environment variable: HTTP_NETTY_EVENT_LOOP_THREADS
     */
    private int httpNettyEventLoopThreads = 4;

    /**
     * HTTP Netty EventExecutorGroup thread count.
     * Environment variable: HTTP_NETTY_EVENT_EXECUTOR_THREADS
     */
    private int httpNettyEventExecutorThreads = 16;

    /**
     * HTTP Netty EventExecutorGroup bounded queue size.
     * Environment variable: HTTP_NETTY_EVENT_EXECUTOR_QUEUE_SIZE
     */
    private int httpNettyEventExecutorQueueSize = 1000;

    /**
     * HTTP Request Executor core pool size.
     * Environment variable: HTTP_REQUEST_EXECUTOR_CORE_SIZE
     */
    private int httpRequestExecutorCoreSize = 32;

    /**
     * HTTP Request Executor max pool size.
     * Set equal to core size because LinkedBlockingQueue prevents thread pool
     * expansion until the queue is full. With a large queue (10000), maxPoolSize
     * would be effectively unreachable, so we keep the pool fixed at core size.
     * Environment variable: HTTP_REQUEST_EXECUTOR_MAX_SIZE
     */
    private int httpRequestExecutorMaxSize = 32;

    /**
     * HTTP Request Executor bounded queue size.
     * Environment variable: HTTP_REQUEST_EXECUTOR_QUEUE_SIZE
     */
    private int httpRequestExecutorQueueSize = 10000;

    /**
     * Engine Sync Executor core pool size.
     * Environment variable: ENGINE_SYNC_EXECUTOR_CORE_SIZE
     */
    private int engineSyncExecutorCoreSize = 32;

    /**
     * Engine Sync Executor max pool size.
     * Environment variable: ENGINE_SYNC_EXECUTOR_MAX_SIZE
     */
    private int engineSyncExecutorMaxSize = 64;

    /**
     * Status Check Executor core pool size.
     * Environment variable: STATUS_CHECK_EXECUTOR_CORE_SIZE
     */
    private int statusCheckExecutorCoreSize = 32;

    /**
     * Status Check Executor max pool size.
     * Environment variable: STATUS_CHECK_EXECUTOR_MAX_SIZE
     */
    private int statusCheckExecutorMaxSize = 64;

    // ========== Engine Sync Configuration ==========

    /**
     * Interval in milliseconds between engine status sync cycles.
     * Environment variable: SYNC_STATUS_INTERVAL.
     */
    private long syncStatusInterval = 20;

    /**
     * Timeout in milliseconds for gRPC sync-request calls to engines.
     * Must be longer than {@link #syncStatusInterval} to avoid premature timeouts.
     * Environment variable: SYNC_REQUEST_TIMEOUT_MS.
     */
    private long syncRequestTimeoutMs = 5000;

    /**
     * Worker status expiration threshold in microseconds.
     * Workers whose last status update is older than this are considered expired and removed.
     * Environment variable: WORKER_TIMEOUT_US.
     */
    private long workerTimeoutUs = 3_000_000L;

    /**
     * Whether to enable verbose cache-status debug logging in gRPC sync runners.
     * Environment variable: WHALE_CACHE_DEBUG_MODE.
     */
    private boolean whaleCacheDebugMode = false;

    /**
     * Service Discovery Executor max pool size.
     * Environment variable: SERVICE_DISCOVERY_MAX_SIZE
     */
    private int serviceDiscoveryMaxSize = 32;

    /**
     * Ordered traffic policy rules. A matched rule forces the whole request to a worker group.
     */
    private volatile TrafficPolicyConfig trafficPolicy = new TrafficPolicyConfig();

    // ========== FlexLB Batch Configuration ==========

    /**
     * Default schedule mode. Controls the routing path for all requests.
     * Environment variable: DEFAULT_SCHEDULE_MODE (values: BATCH, DIRECT, QUEUE).
     */
    private String defaultScheduleMode = "BATCH";

    /**
     * Maximum real requests in one EnqueueBatch request.
     */
    private int flexlbBatchSizeMax = 8;

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
     * that stale inflight won't distort prefillEstimatedWaitTimeMs for long.
     */
    private long flexlbInflightTtlMs = 300_000L;

    /**
     * EP-level (layer-2 engineWork) inflight TTL — a separate, longer TTL
     * for engine-acknowledged entries that have migrated to layer 2.
     * Defaults to 600s (vs 300s for scheduler-level {@link #flexlbInflightTtlMs}),
     * because engine-accepted tasks legitimately run longer (decode generation)
     * and should not be prematurely evicted by the wall-clock backstop.
     * Environment variable: FLEXLB_EP_INFLIGHT_TTL_MS.
     */
    private long flexlbEpInflightTtlMs = 600_000L;

    /**
     * Tombstone retention period in the {@link org.flexlb.balance.scheduler.InflightStore}.
     * Terminal items remain as tombstones for this long after termination, so that
     * late cancel lookups return {@code false} (already terminal) rather than
     * {@code null} (not found). Environment variable: FLEXLB_TOMBSTONE_TTL_MS.
     */
    private long flexlbTombstoneTtlMs = 60_000L;

    /**
     * Number of consecutive calibrate rounds an engineWork entry can be absent
     * from both running and finished reports before being evicted as stale
     * (lost completion report). Environment variable: FLEXLB_STALE_EVICT_ROUNDS.
     */
    private int flexlbStaleEvictRounds = 3;

    /**
     * Default KV token estimate used when a cross-EP failover task is reported
     * by the engine but has no local reservation (foreign key). Falls back to
     * this value when the engine-reported {@code input_length} is 0 or missing.
     * Environment variable: DEFAULT_KV_TOKENS.
     */
    private long defaultKvTokens = 2048;

    /**
     * Maximum new tokens (generation length) added to the prompt's input length
     * to estimate the total KV demand for cross-EP failover tasks.
     * {@code expectedKv = inputLength + maxNewTokens}.
     * Environment variable: MAX_NEW_TOKENS.
     */
    private long maxNewTokens = 1024;

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
     * Whether to enable score-tie randomization among near-equal prefill candidates.
     * When enabled (default), endpoints within a threshold of the minimum score are
     * randomly selected to avoid deterministic routing bias.
     * When disabled, only endpoints with exactly the minimum score are considered.
     * Environment variable: SCORE_TIE_RANDOM_ENABLED.
     */
    private boolean scoreTieRandomEnabled = true;

    /**
     * Percentage threshold for score-tie randomization.
     * Endpoints whose score is within (minScore * scoreTieThresholdPct) of the
     * minimum score are considered "tied" and randomly selected.
     * Default 0.1 means 10% of the minimum score.
     * Environment variable: SCORE_TIE_THRESHOLD_PCT.
     */
    private double scoreTieThresholdPct = 0.1;

    /**
     * Minimum absolute threshold (in milliseconds) for score-tie randomization.
     * The effective threshold is max(minScore * scoreTieThresholdPct, scoreTieThresholdMs).
     * Default 20ms.
     * Environment variable: SCORE_TIE_THRESHOLD_MS.
     */
    private long scoreTieThresholdMs = 20;

    // ========== ShortestTTFT Strategy Configuration ==========

    /** Candidate pool mode: "RATIO" (floor(workerCount * ratio)) or "FIXED" (absolute size). */
    private String shortestTtftCandidatePoolMode = "RATIO";

    /** Ratio of workers in candidate pool (RATIO mode only, 0 < ratio <= 1.0). */
    private double shortestTtftCandidatePoolRatio = 0.3;

    /** Minimum candidate pool size (RATIO mode only). */
    private int shortestTtftCandidatePoolMinSize = 1;

    /** Fixed candidate pool size (FIXED mode only). */
    private int shortestTtftCandidatePoolSize = 1;

    public int resolveShortestTtftCandidateCount(int workerCount) {
        if ("FIXED".equalsIgnoreCase(shortestTtftCandidatePoolMode)) {
            return Math.max(1, Math.min(shortestTtftCandidatePoolSize, workerCount));
        }
        // RATIO mode
        return Math.max(1, Math.max(shortestTtftCandidatePoolMinSize,
                (int) Math.floor(workerCount * shortestTtftCandidatePoolRatio)));
    }

    /**
     * Configurable prefill-time prediction formula.
     *
     * <p>Batch-scoped variables:
     * {@code batchSize, totalInputTokens, totalHitCacheTokens, totalComputeTokens,
     * maxInputTokens, maxComputeTokens}
     * <br>Per-request variables:
     * {@code inputTokens, hitCacheTokens, computeTokens, hasHitCache}
     * <br>Operators: {@code + - * / ^}
     * <br>Functions: {@code sqrt(x) log(x) exp(x) abs(x) max(a,b) min(a,b) pow(a,b)}
     * <br>Batch aggregate: {@code sum(expr)} evaluates {@code expr} per request and sums it.
     * Batch-scoped variables are not valid inside {@code sum(expr)}. For example,
     * {@code totalComputeTokens^2} is the square of the whole batch's compute-token count,
     * whereas {@code sum(computeTokens^2)} is the sum of per-request squares.
     */
    private String costFormula = "sum(computeTokens) + 0.3*sum(hitCacheTokens)";

    /**
     * Prefill time predictor type. Supported values:
     * <ul>
     *   <li>{@code formula} — Formula-driven predictor with configurable costFormula (default)</li>
     *   <li>{@code learning} — Hardcoded linear regression predictor with online learning</li>
     * </ul>
     */
    private String prefillPredictorType = "formula";

    // ========== Batcher Configuration ==========

    private int flexlbBatchMaxCapacity = 1048576;

    /**
     * Maximum queue depth per WorkerBatcher (BATCH routing path). Requests
     * beyond this limit are rejected with QUEUE_FULL.
     *
     * <p><b>Disambiguation:</b> This is the per-batcher queue limit for the
     * BATCH routing path, distinct from {@link #queueingComponentQueueMaxSize}
     * which is the global queue for the DIRECT/QUEUE routing path.
     */
    private int flexlbBatchQueueMaxSize = 1024;

    /**
     * Maximum total in-flight requests across all batchers. Acts as a global
     * admission control gate at the BatchScheduler entry.
     */
    private int flexlbBatchMaxInflight = 100000;

    // ========== Fixed-Window Batcher Configuration ==========

    /**
     * Fixed wait time in milliseconds for the {@code fixed_window} batcher
     * algorithm. After a request has waited this long, the batcher dispatches
     * whatever has accumulated regardless of batch size.
     */
    private long flexlbBatchFixedWaitMs = 300;

    /**
     * Predicted batch execution time threshold in milliseconds for the
     * {@code fixed_window} batcher algorithm. If the predictor estimates
     * the accumulated batch will take at least this long, the batcher
     * dispatches immediately rather than waiting for {@link #flexlbBatchFixedWaitMs}.
     *
     * <p>Set to 0 to disable predictor-based early dispatch (default).
     */
    private long flexlbBatchPredictThresholdMs = 0;

    // ========== Metrics Reporting Configuration ==========

    /**
     * FlexLB state v2 shadow mode (G1): when enabled, the new flexlb-state
     * StateLedger consumes the same engine status event stream and local
     * lifecycle events in parallel with the legacy path, purely for
     * observation and terminal-state diff accounting. The legacy routing /
     * dispatch / settlement behavior is unchanged — every shadow call is
     * wrapped catch-all and can never affect the main path.
     *
     * <p>Resolved once at startup (no runtime hot-toggle). Environment
     * variable: FLEXLB_STATE_V2_SHADOW_ENABLED (also echoed in the effective
     * config dump at startup, R2).
     */
    private boolean flexlbStateV2ShadowEnabled = false;

    /**
     * FlexLB state v2 evidence-channel (F2) absence threshold: a ledger entry
     * that was engine-confirmed at least once (non-negative lastSeenRound) and
     * has been absent for more than this many complete observation rounds is
     * presumed dead and settled as VANISHED by the LedgerJanitor. Guard-rail 1
     * (debounce) is satisfied naturally by this threshold.
     *
     * <p>Only complete ticks ({@code detailCount == running.size()}, E7) count
     * toward absence; truncated reports never advance the absence tracking.
     * Environment variable: FLEXLB_STATE_V2_STALE_ROUNDS. Default: 3.
     */
    private int flexlbStateV2StaleRounds = 3;

    /**
     * FlexLB state v2 time-channel (F3) TTL in milliseconds: ledger entries
     * older than this (measured from createdAtMs, which is final and never
     * renewed by any touch/observe — R5) are settled as TTL_EXPIRED by the
     * LedgerJanitor. Default 300s, aligned with the legacy InflightStore TTL.
     *
     * <p>Fenced entries are exempt (guard-rail 3 / R4) until the fence expires
     * or is lifted. Environment variable: FLEXLB_STATE_V2_TTL_MS.
     */
    private long flexlbStateV2TtlMs = 300_000L;

    /**
     * FlexLB state v2 force-channel (F4) hard cap in milliseconds: entries
     * older than createdAtMs + this cap are settled unconditionally as
     * HARD_CAP — <b>fences do not exempt</b> (a fence that outlives the hard
     * cap is itself a leak; prefer clearing over keeping, with doubled alarm
     * accounting). Must be strictly greater than flexlbStateV2TtlMs.
     * Default 900s. Environment variable: FLEXLB_STATE_V2_HARD_CAP_MS.
     */
    private long flexlbStateV2HardCapMs = 900_000L;

    /**
     * FlexLB state v2 LedgerJanitor maintenance tick interval in milliseconds:
     * the low-frequency scheduled scan (TTL + hard cap rotation, absence-orphan
     * cleanup, tombstone/fence expiry) driven by the shadow bridge when
     * flexlbStateV2ShadowEnabled is on. Not started when shadow mode is off
     * (the legacy path keeps its own InflightEvictor).
     *
     * <p>Environment variable: FLEXLB_STATE_V2_JANITOR_INTERVAL_MS. Default 10s.
     */
    private long flexlbStateV2JanitorIntervalMs = 10_000L;

    /**
     * FlexLB state v2 settlement authority switch (G3 — 终态结算换权): when
     * enabled, terminal settlement of BATCH-path requests converges on the
     * StateLedger as the authoritative bookkeeper, while the legacy callback
     * chain keeps driving the client future unchanged (client-visible behavior
     * is identical):
     * <ul>
     *   <li>Legacy COMPLETED (engine ACK of enqueue) does NOT pre-settle the
     *       ledger — engine-execution phases and the KV billing handover
     *       (engine-reported KV takes over the local reservation only after
     *       the engine confirms allocation) stay intact. The terminal metric
     *       is parked in a pending table inside the shadow bridge and is
     *       produced at the ledger's own terminal exit (decode tombstone
     *       first, prefill tombstone as fallback).</li>
     *   <li>Legacy FAILED / TIMED_OUT / CANCELLED proactively settle both
     *       ledger sides — the master has already declared the request dead —
     *       and the terminal metric is reported immediately at the settle
     *       exit (single production point per request).</li>
     * </ul>
     *
     * <p>Prerequisite: {@link #flexlbStateV2ShadowEnabled} must be on; startup
     * fails fast otherwise (settlement authority requires the shadow ledger
     * to be running). DIRECT / QUEUE routing paths are unaffected (the shadow
     * ledger only covers the BATCH path). Resolved once at startup (no runtime
     * hot-toggle). Environment variable: FLEXLB_STATE_V2_SETTLE_ENABLED.
     */
    private boolean flexlbStateV2SettleEnabled = false;

    /**
     * Metrics report interval in milliseconds.
     * Controls the periodic reporting frequency for scheduler-level and
     * per-endpoint metrics via {@code @Scheduled} throttle.
     * Environment variable: METRICS_REPORT_INTERVAL_MS.
     */
    private long metricsReportIntervalMs = 2000;

    /**
     * Comma-separated metric-name whitelist for the Micrometer reporting path.
     * <ul>
     *   <li>Empty or {@code "*"} → report all metrics (no filtering)</li>
     *   <li>Comma-separated metric names (without the {@code flexlb.} prefix)
     *       → only those metrics are registered/reported</li>
     * </ul>
     * Replaces the former {@code flexlb.monitor.mode=critical-only} toggle +
     * hardcoded {@code CriticalMetricsFilterConfig.CRITICAL_METRICS} set.
     * KMonitor path is unaffected (production always reports all metrics).
     * Environment variable: FLEXLB_MONITOR_CRITICAL_METRICS.
     */
    private String flexlbMonitorCriticalMetrics = "";

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
     * Returns {@code true} when the effective schedule mode is BATCH.
     * Convenience method for strategy classes that need to decide whether
     * to reserve prefill inflight locally or defer to BatchScheduler.
     */
    public boolean isBatchPath() {
        return getDefaultScheduleModeEnum() == ScheduleModeEnum.BATCH;
    }

    /**
     * Returns the configured default schedule mode as an enum.
     *
     * <p>Behavior:
     * <ul>
     *   <li>null or blank → {@code BATCH} with WARN log (unconfigured, use default)</li>
     *   <li>{@code "AUTO"} (legacy, case-insensitive) → {@code BATCH} with WARN log
     *       (backward compatibility, field is corrected to avoid repeated warnings)</li>
     *   <li>Any other invalid string (e.g. typos like "queuu") → throws
     *       {@link IllegalArgumentException} (fail-fast, so misconfiguration is caught immediately)</li>
     * </ul>
     */
    public ScheduleModeEnum getDefaultScheduleModeEnum() {
        if (defaultScheduleMode == null || defaultScheduleMode.isBlank()) {
            log.warn("defaultScheduleMode is null/blank, falling back to BATCH");
            defaultScheduleMode = "BATCH";
            return ScheduleModeEnum.BATCH;
        }
        String upper = defaultScheduleMode.toUpperCase();
        // Backward compatibility: legacy AUTO mode degrades to BATCH
        if ("AUTO".equals(upper)) {
            log.warn("Legacy schedule mode 'AUTO' is deprecated, falling back to BATCH. Use BATCH/DIRECT/QUEUE instead.");
            defaultScheduleMode = "BATCH";
            return ScheduleModeEnum.BATCH;
        }
        try {
            return ScheduleModeEnum.valueOf(upper);
        } catch (IllegalArgumentException e) {
            throw new IllegalArgumentException(
                String.format("Invalid schedule mode '%s'. Valid values: BATCH, DIRECT, QUEUE", defaultScheduleMode), e);
        }
    }
}
