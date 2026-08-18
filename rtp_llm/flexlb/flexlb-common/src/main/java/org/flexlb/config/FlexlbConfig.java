package org.flexlb.config;

import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
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
     * Hard age cap for inflight ledger entries, enforced across all TTL
     * exemptions (dispatch-reconciliation fence, preemption claim, cleanup
     * ownership) and observation-refresh keep-alives.  A stuck engine that
     * keeps re-reporting a zombie task (e.g. PENDING + priority-cancel
     * overlay, never executed) must not pin ledger entries — and the
     * inflight.max.age.ms metric — forever.  Must comfortably exceed the
     * longest legitimate request lifecycle (admission SLO caps at seconds,
     * prefill execution at minutes).  {@code <= 0} disables the cap.
     * Environment variable: FLEXLB_INFLIGHT_HARD_MAX_AGE_MS.
     */
    private long flexlbInflightHardMaxAgeMs = 1_800_000L;

    /**
     * Batch-level inflight age cap (F-F, na130_4 bounded-freeze fix), in
     * milliseconds, enforced by the endpoint 60s eviction sweep
     * ({@code PrefillEndpoint.evictExpiredBatches}): a committed inflight
     * batch whose creation age ({@code now - createdAtMs}) exceeds this cap
     * <b>and</b> has gone unobserved for longer than
     * {@link #flexlbBatchInflightStaleMs} is force-settled — even while a
     * dispatch-reconciliation fence holds it (that is the point: a zombie
     * reconciliation that never receives its authoritative settlement must
     * not freeze the fixed-window inflight gate and pin
     * {@code inflight.batch.count} forever). Batches that the ~20ms worker
     * status sync still observes (running members, saturated queued
     * batches, long-generation pdFusion batches) keep refreshing
     * {@code lastObservedAtMs} and are never capped. Effective window per
     * unobserved batch: {@code [maxAge, maxAge + stale + 60s sweep)}. On
     * release the batch entry, its reconciliation fence and the request
     * counter are dropped, and each member is routed through the existing
     * handler terminal chain ({@code BatchDecisionHandler.onExpired}).
     * Auto-TPM only (the registry gates the pass with
     * {@link #autoTpmEnabled}). {@code <= 0} disables the cap.
     * Environment variable: FLEXLB_BATCH_INFLIGHT_MAX_AGE_MS.
     */
    private long flexlbBatchInflightMaxAgeMs = 120_000L;

    /**
     * No-progress staleness threshold for the batch-level inflight age cap
     * (F-F), in milliseconds, paired with
     * {@link #flexlbBatchInflightMaxAgeMs}: an over-age batch is only
     * force-settled when its last observation
     * ({@code now - lastObservedAtMs}) is also older than this threshold —
     * the progress-aware guard that keeps legitimately long batches alive
     * while they are still being observed by the worker status sync.
     * {@code <= 0} disables the staleness exemption (pure age cap, the
     * pre-review behavior).
     * Environment variable: FLEXLB_BATCH_INFLIGHT_STALE_MS.
     */
    private long flexlbBatchInflightStaleMs = 60_000L;

    /**
     * Frozen-batch audit threshold in milliseconds. Each inflight TTL sweep
     * (60s, {@code EndpointRegistry.scheduledEviction}) emits one WARN audit
     * line per still-resident batch older than this threshold — with the
     * exact fields that reveal which exemption leg kept it alive
     * (over_age_cap / stale verdicts, dispatch fence, scheduler ownership,
     * member terminal distribution) — rate-limited to 5 lines per endpoint
     * per sweep. The frozen-audit QPS metric
     * ({@code app.flexlb.batch.inflight.frozen.audit.qps}) counts the
     * audited batches. Deployed to explain the na130_4 pattern
     * "age.capped = 0 while ttl.expired > 0": over-age batches being kept
     * alive by the observed-freshness/fence exemption legs.
     * {@code <= 0} disables the audit entirely.
     * Environment variable: FLEXLB_BATCH_FROZEN_AUDIT_AFTER_MS.
     */
    private long flexlbBatchFrozenAuditAfterMs = 60_000L;

    /**
     * Post-ACK inflight audit threshold in milliseconds. The scheduler audit
     * tick force-settles an inflight ledger entry older than this when its
     * public future is already completed, no fence (preemption claim /
     * dispatch reconciliation / cleanup ownership) retains it, and neither
     * the prefill batch ledger nor the decode confirmed registry still
     * tracks the request — the post-ACK leak where an ACK-released entry
     * lingers in the ledger with nothing left that can settle it through
     * the ordinary paths. Shorter than {@link #flexlbInflightTtlMs} so such
     * leaks clear in seconds instead of minutes. {@code 0} disables the
     * audit entirely.
     * Environment variable: FLEXLB_INFLIGHT_AUDIT_AFTER_MS.
     */
    private long flexlbInflightAuditAfterMs = 30_000L;

    /**
     * ACKNOWLEDGED-lost detection threshold (Fix A, 205 pileup incident), in
     * milliseconds. A committed inflight batch that a <b>successful</b> engine
     * WorkerStatus report (calibrate round) fails to mention — neither in
     * {@code finished_task_info} nor {@code running_task_info} — for
     * {@link #flexlbPrefillLostMinMisses} consecutive rounds <b>and</b> whose
     * {@code lastObservedAtMs} is older than this threshold is declared lost
     * (the engine silently dropped it, e.g. a DeferredPrefillContext that
     * evaporated after a clean EnqueueBatch ACK) and force-settled through
     * the ordinary handler terminal chain. Detection is driven purely by the
     * master-side observation loop — no engine-side cooperation or C++
     * change is required. Must exceed the EnqueueBatch deadline
     * ({@link #flexlbBatchEnqueueDeadlineMs}, 5s) plus several status-sync
     * rounds; far below the batch age cap
     * ({@link #flexlbBatchInflightMaxAgeMs}, 120s) and the inflight TTL
     * ({@link #flexlbInflightTtlMs}, 300s) it front-runs. Auto-TPM only.
     * {@code <= 0} disables the detection.
     * Environment variable: FLEXLB_PREFILL_LOST_AFTER_MS.
     */
    private long flexlbPrefillLostAfterMs = 20_000L;

    /**
     * Minimum consecutive unobserved calibrate rounds before the
     * ACKNOWLEDGED-lost detection ({@link #flexlbPrefillLostAfterMs}) may
     * fire. Miss counting advances only on rounds backed by a real engine
     * report (version-advanced, alive) and resets the moment any member of
     * the batch is mentioned, so sync stalls or pull failures can never
     * accumulate misses. Values below 1 are clamped to 1.
     * Environment variable: FLEXLB_PREFILL_LOST_MIN_MISSES.
     */
    private int flexlbPrefillLostMinMisses = 3;

    /**
     * Ack-only release gate: when true (default), the frontend-facing fetch
     * release completes only on the Prefill EnqueueBatch ACK semantic —
     * a direct/late ACK or a Prefill WorkerStatus observation of the same
     * dispatch generation, both of which happen strictly after the engine
     * stored the deferred fetch slot. Decode WorkerStatus / DECODE_OWNED no
     * longer triggers release. Set to false to restore the legacy
     * Decode-owned shortcut release paths.
     */
    private boolean flexlbAckOnlyReleaseEnabled = true;

    /**
     * Grace window (ms) before a fenced uncertain dispatch whose Cancel
     * target has disappeared from the EndpointRegistry is force-settled as
     * an ordinary terminal (fence-leak fix A). Registry removal already lags
     * real pod death by workerTimeoutMs, so the effective settle delay is
     * roughly the sum of both. Set to 0 or below to disable the guard and
     * restore the legacy unbounded reconciliation retry loop.
     * Environment variable: FLEXLB_RECONCILE_TARGET_MISSING_TERMINAL_MS.
     */
    private long flexlbReconcileTargetMissingTerminalMs = 15_000L;

    /**
     * Maximum consecutive failed reconciliation Cancels (FAILED / NOT_FOUND /
     * UNSUPPORTED) before the entry is force-settled as an ordinary terminal
     * (fence-leak fix B, the D3 backstop). 36 tries at the 5s retry-backoff
     * ceiling is about 3 minutes — far past the EnqueueBatch deadline, so a
     * late enqueue can no longer land. Set to 0 or below to disable the cap
     * and restore the legacy unbounded reconciliation retry loop.
     * Environment variable: FLEXLB_RECONCILE_MAX_CONSECUTIVE_FAILURES.
     */
    private int flexlbReconcileMaxConsecutiveFailures = 36;

    /**
     * Flush-path top-k sort gate (task61 M1): when true (default) the
     * SLO-budget batcher picks its greedy-fill candidates with a bounded-heap
     * top-k selection instead of a per-flush full sort of the queue. Only
     * effective on the Auto-TPM path, whose queue comparator is a total order
     * (requestId tie-break) — that is what makes the top-k prefix provably
     * identical to the full-sort prefix. Set to false to restore the full
     * per-flush sort.
     */
    private boolean flexlbFlushTopKSortEnabled = true;

    /**
     * Snapshot sort placement gate (task61 M2): when true (default) the
     * prefill queue snapshot copies items and captures the queue version
     * inside the queue lock, then sorts outside the lock. The copy+version
     * pair is captured atomically under the lock and the sort is a pure
     * function of the thread-private copy, so the "version unchanged =>
     * content unchanged" optimistic-concurrency invariant is preserved.
     * Set to false to restore sorting inside the lock.
     */
    private boolean flexlbSnapshotSortOutsideLockEnabled = true;

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
     * Queue-depth penalty gate for the Auto-TPM batcher wait estimate
     * ({@code PrefillQueueManager.estimateWaitMs}, design doc 8.4). When
     * true (default), the estimate returns {@code max(jumpWait, depthWait)},
     * where the depth term is {@code (queueSize / flexlbBatchSizeMax) ×
     * avgDispatchIntervalMs × flexlbQueueDepthPenaltyFactor}. The legacy
     * jump estimate only counts items ordered ahead of the probe, so a
     * high-priority request facing an already-saturated queue reports a
     * near-zero wait and the slow engine keeps being the preferred target
     * (na130_4). The depth term exposes the true drain horizon of the full
     * queue regardless of the probe's priority. Set to false to restore the
     * legacy jump-only estimate.
     * Environment variable: FLEXLB_QUEUE_DEPTH_PENALTY_ENABLED.
     */
    private boolean flexlbQueueDepthPenaltyEnabled = true;

    /**
     * Multiplier of the depth term in the batcher wait estimate when
     * {@link #flexlbQueueDepthPenaltyEnabled} is on:
     * {@code depthWait = (queueSize / flexlbBatchSizeMax) ×
     * avgDispatchIntervalMs × thisFactor}. Default 1.0 (linear in queue
     * depth); values &gt; 1 penalize deep queues harder, 0 makes the term a
     * no-op.
     * Environment variable: FLEXLB_QUEUE_DEPTH_PENALTY_FACTOR.
     */
    private double flexlbQueueDepthPenaltyFactor = 1.0;

    /**
     * Congested-queue candidate filter gate for prefill selection
     * ({@code CostBasedPrefillStrategy}): when true (default), a prefill
     * endpoint whose batcher queue depth is at least
     * {@code flexlbCongestedQueueRatio × flexlbBatchQueueMaxSize} is
     * excluded from routing candidates ("CONGESTED_QUEUE_FILTERED"), so an
     * engine whose queue is pinned near its cap stops being the preferred
     * target (na130_4). When every feasible endpoint is congested, the
     * existing least-loaded fallback still returns one endpoint, so routing
     * never fails closed. A non-positive {@code flexlbBatchQueueMaxSize}
     * (unbounded) disables the filter. Set to false to restore the legacy
     * candidate set without the queue-depth condition.
     * Environment variable: FLEXLB_CONGESTED_QUEUE_FILTER_ENABLED.
     */
    private boolean flexlbCongestedQueueFilterEnabled = true;

    /**
     * Queue-occupancy ratio (0-1, default 0.8) of
     * {@link #flexlbBatchQueueMaxSize} at which the congested-queue filter
     * excludes an endpoint, when {@link #flexlbCongestedQueueFilterEnabled}
     * is on. An endpoint is congested when
     * {@code queueSize >= ceil(ratio × flexlbBatchQueueMaxSize)}.
     * Environment variable: FLEXLB_CONGESTED_QUEUE_RATIO.
     */
    private double flexlbCongestedQueueRatio = 0.8;

    /**
     * Engine-wait penalty gate for prefill selection
     * ({@code CostBasedPrefillStrategy}): when true (default), each
     * engine-side queued request reported by the engine
     * ({@code waitingQueryLen}, synced every ~20ms via worker status) adds
     * {@link #flexlbEngineWaitPenaltyMsPerWaitStream} ms to that endpoint's
     * Round-1 score, so engines whose engine-side admission queue keeps
     * growing lose routing attractiveness even when the master-side view
     * (inflight + batcher queue) looks clean (na130_4). Only effective in
     * Auto-TPM deployments (gated on {@link #autoTpmEnabled} like the
     * congested-queue filter); when the gate is off the term is exactly 0
     * (legacy score).
     * Environment variable: FLEXLB_ENGINE_WAIT_PENALTY_ENABLED.
     */
    private boolean flexlbEngineWaitPenaltyEnabled = true;

    /**
     * Score penalty in milliseconds per engine-side waiting request when
     * {@link #flexlbEngineWaitPenaltyEnabled} is on (and Auto-TPM is
     * enabled):
     * {@code engineWaitMs = min(reportedWaitingQueryLen × thisFactor,
     * 1L << 40)}. Default 20.0; a non-finite or non-positive value falls
     * back to 20.0 at use time (treated as the default, not as 0).
     * Environment variable: FLEXLB_ENGINE_WAIT_PENALTY_MS_PER_WAIT_STREAM.
     */
    private double flexlbEngineWaitPenaltyMsPerWaitStream = 20.0;

    /**
     * Engine-wait hard filter gate for prefill selection
     * ({@code CostBasedPrefillStrategy}): when true (default), a prefill
     * endpoint whose reported {@code waitingQueryLen} has reached
     * {@link #flexlbEngineWaitHardFilterThreshold} is excluded from routing
     * candidates outright ("ENGINE_WAIT_FILTERED"). Only effective in
     * Auto-TPM deployments (gated on {@link #autoTpmEnabled} like the
     * congested-queue filter). Same fallback semantics: when every feasible
     * endpoint exceeds the threshold, the existing least-loaded fallback
     * still returns one endpoint, so routing never fails closed. Set to
     * false to restore the legacy candidate set without the engine-wait
     * condition.
     * Environment variable: FLEXLB_ENGINE_WAIT_HARD_FILTER_ENABLED.
     */
    private boolean flexlbEngineWaitHardFilterEnabled = true;

    /**
     * Engine-reported {@code waitingQueryLen} at which the engine-wait hard
     * filter excludes an endpoint, when
     * {@link #flexlbEngineWaitHardFilterEnabled} is on (and Auto-TPM is
     * enabled). An endpoint is filtered when
     * {@code waitingQueryLen >= threshold}. Default 256; a non-positive
     * value is a misconfiguration that would filter every engine and is
     * treated at use time as disabled (filter never fires).
     * Environment variable: FLEXLB_ENGINE_WAIT_HARD_FILTER_THRESHOLD.
     */
    private int flexlbEngineWaitHardFilterThreshold = 256;

    /**
     * Pending-offer penalty gate for prefill selection (R1, 205 pileup
     * incident): when true (default), requests that route() already
     * committed to an endpoint but that the batcher has not yet accepted
     * (the route→offer blind window) add
     * {@link #flexlbPendingOfferPenaltyMsPerRequest} each to the endpoint's
     * Round-1 score, so a burst routed to one endpoint within a single
     * scoring epoch stops looking free to the followers. Only effective in
     * Auto-TPM batch-path deployments (gated on {@link #autoTpmEnabled}
     * like the other score terms); legacy scoring is bit-for-bit unchanged.
     * Environment variable: FLEXLB_PENDING_OFFER_PENALTY_ENABLED.
     */
    private boolean flexlbPendingOfferPenaltyEnabled = true;

    /**
     * Score penalty in milliseconds per pending (route-committed, not yet
     * offered) request when {@link #flexlbPendingOfferPenaltyEnabled} is on
     * (and Auto-TPM is enabled):
     * {@code pendingOfferMs = min(pendingOfferCount × thisFactor, 1L << 40)}.
     * Default 50.0; a non-finite or non-positive value falls back to 50.0
     * at use time (treated as the default, not as 0).
     * Environment variable: FLEXLB_PENDING_OFFER_PENALTY_MS_PER_REQUEST.
     */
    private double flexlbPendingOfferPenaltyMsPerRequest = 50.0;

    /**
     * Engine-untracked penalty gate for prefill selection (S3, 205 pileup
     * incident): when true (default), active engine tasks that the local
     * batch ledger does not track ({@code engineUntrackedRequestCount} —
     * e.g. requests re-routed by another master generation, or the scalar
     * lower bound when the engine omits task details) add
     * {@link #flexlbEngineUntrackedPenaltyMsPerRequest} each to the
     * endpoint's Round-1 score. Closes the scoring blind spot where an
     * engine busy with untracked work looked as attractive as an idle one.
     * Only effective in Auto-TPM deployments (gated on
     * {@link #autoTpmEnabled}); legacy scoring is bit-for-bit unchanged.
     * Environment variable: FLEXLB_ENGINE_UNTRACKED_PENALTY_ENABLED.
     */
    private boolean flexlbEngineUntrackedPenaltyEnabled = true;

    /**
     * Score penalty in milliseconds per engine-untracked active request
     * when {@link #flexlbEngineUntrackedPenaltyEnabled} is on (and Auto-TPM
     * is enabled):
     * {@code engineUntrackedMs = min(engineUntrackedRequestCount ×
     * thisFactor, 1L << 40)}. Default 20.0 (same scale as the engine-wait
     * penalty — both count engine-side work the master ledgers cannot see);
     * a non-finite or non-positive value falls back to 20.0 at use time.
     * Environment variable: FLEXLB_ENGINE_UNTRACKED_PENALTY_MS_PER_REQUEST.
     */
    private double flexlbEngineUntrackedPenaltyMsPerRequest = 20.0;

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

    // ========== SLO-Budget Batcher Configuration ==========

    private int flexlbBatchMaxCapacity = 1048576;

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
     * Upper bound on maxNewTokens used solely for local KV-reservation
     * estimation (expectedKvTokens = seqLen + clamped maxNewTokens).
     * The request's original maxNewTokens is never modified — this cap only
     * prevents over-conservative reservation when the declared generation
     * budget is very large (e.g. 393216). Default 1000 based on actual
     * generation distribution (~466-1166 tokens); 0 disables the cap.
     * Environment variable: MAX_NEW_TOKENS_CAP.
     */
    private long maxNewTokensCap = 1000;

    // ========== Auto-TPM Priority Scheduler Configuration ==========

    private boolean autoTpmEnabled = false;

    /** 默认 50，一般无需修改（unset 请求的归一目标档）。 */
    private int autoTpmDefaultPriority = 50;

    private String autoTpmSloLengthBuckets = "256:150,1024:300,4096:600,16384:1200,*:2400";

    private String autoTpmPrioritySloMultipliers = "30:2.0,40:1.5,50:1.0,60:0.75,70:0.5";

    private boolean autoTpmPrefillQueueEvictEnabled = false;

    private boolean autoTpmDecodeReservedEvictEnabled = false;

    /**
     * Upper bound (in plan-cost units) of the cache-hit benefit an eviction
     * plan may subtract from its raw cost. The effective benefit is further
     * clamped to half of the minimum adjacent-priority cost gap so cache
     * affinity can never flip a cross-priority victim choice. 0 (default)
     * disables cache benefit entirely.
     */
    private long autoTpmPlanCacheHitBenefitCap = 0;

    /**
     * Post-success soft timeout for AdmissionLease (ms). When prefill succeeds
     * but decode hasn't accepted within this window, the lease is force-closed
     * and a cancel signal is sent to the engine, releasing the pinned KV cache
     * block. 0 disables the soft timeout (legacy behavior — leaks on OOM).
     * Environment variable: AUTO_TPM_POST_SUCCESS_SOFT_TIMEOUT_MS.
     */
    private long autoTpmPostSuccessSoftTimeoutMs = 30000;

    /**
     * Backpressure limit for handed-over-but-not-accepted requests. When the
     * active lease count (handed over but not yet accepted by decode) exceeds
     * this, new prefill requests are rejected with 8502 (QUEUE_FULL). 0 disables
     * the backpressure check.
     * Environment variable: AUTO_TPM_POST_SUCCESS_BACKPRESSURE_LIMIT.
     */
    private int autoTpmPostSuccessBackpressureLimit = 200;

    // ---- Auto-TPM reserved config (design doc §18) — future phases, not wired yet ----

    /** Decode engine-owned/in-flight eviction switch. */
    private boolean autoTpmDecodeAcceptedEvictEnabled = false;

    /** Deadline for the Prefill Cancel RPC acknowledgement phase. */
    private long autoTpmCancelAckTimeoutMs = 50;

    /** Independent deadline for typed WorkerStatus CANCELED confirmation. */
    private long autoTpmCancelCompletionTimeoutMs = 1000;

    /**
     * Normal-path plan commit strategy (plan-commit concurrency redesign N3):
     * {@code lockfree} — no snapshot-version validation; the enqueue relies on
     * the queue lock's own atomicity and local capacity checks, so unrelated
     * queue churn can no longer abort a commit (the VERSION_MISMATCH storm at
     * high QPS). {@code versioned} — the legacy optimistic-concurrency
     * protocol, kept as a gray-release fallback for one version cycle.
     * Default {@code lockfree}: the versioned protocol is proven unusable at
     * production QPS (85%+ commit failures under homogeneous load).
     */
    private String autoTpmCommitStrategy = "lockfree";

    /**
     * Eviction victim guard mode (redesign N3): {@code victim_presence} —
     * victim-level guards (queue victims: atomic remove-if-present with
     * zero-side-effect VICTIM_GONE abort; decode victims: still-reserved
     * validation) instead of whole-queue/endpoint version checks, so
     * unrelated mutations no longer abort an eviction commit.
     * {@code queue_version} — the legacy snapshot-version guard (fallback).
     */
    private String autoTpmVictimGuardMode = "victim_presence";

    /**
     * Scheduling snapshot capture mode (O(1) snapshot redesign):
     * {@code summary} (default) — O(1) aggregate-only decode summaries on
     * the normal path, with the full per-entry snapshots built lazily only
     * when the eviction / failure classification paths need them;
     * {@code full} — legacy per-attempt full decode snapshots (per-endpoint
     * layered view under the admission lock), bit-for-bit identical to the
     * pre-redesign decision path.
     * Environment variable: AUTO_TPM_SNAPSHOT_MODE.
     */
    private String autoTpmSnapshotMode = "summary";

    /**
     * True when the O(1) summary snapshot path is enabled; any value other
     * than {@code summary} keeps the legacy full-capture behavior.
     */
    public boolean isAutoTpmSnapshotSummaryMode() {
        return "summary".equalsIgnoreCase(autoTpmSnapshotMode);
    }

    /**
     * TTL in milliseconds for the shared {@code ClusterSnapshot} cache on the
     * priority admission path. Capturing a snapshot walks every endpoint under
     * its admission lock and deep-copies the layered views; doing that
     * per-request×retry (~6000 captures/s at 2000 QPS) is an O(N) allocation
     * flood (~740MB/s with high inflight) that stalls the master in GC.
     * Endpoint state only refreshes on the ~3.2s sync cadence, so requests and
     * retries inside one TTL window safely share a single immutable snapshot
     * (OCC-conflict retries force a refresh — see the scheduler cache doc).
     * 0 disables the cache and falls back to per-call capture (legacy
     * behavior). Environment variable: FLEXLB_CLUSTER_SNAPSHOT_CACHE_TTL_MS.
     */
    private long flexlbClusterSnapshotCacheTtlMs = 200;

    // ========== Worker Expiration Configuration ==========

    /**
     * Worker status expiration timeout in milliseconds.
     * <p>Must be at least 2× the gRPC sync request timeout (5000 ms) to prevent
     * the ExpirationCleaner from removing endpoints that are still alive but
     * experiencing transient gRPC delays — the root cause of the decode
     * death-spiral (error_8400).
     * <p>Default: 10000 (10 seconds, 2× gRPC 5s timeout — does not excessively
     * delay dead-worker detection). Environment variable: WORKER_TIMEOUT_MS.
     */
    private long workerTimeoutMs = 10000L;

    // ========== Worker Status Sync Failure Tolerance Configuration ==========

    /**
     * Max consecutive DEADLINE_EXCEEDED (timeout) GetWorkerStatus failures before
     * a worker is marked dead and its endpoint removed.
     * <p>Slow != dead: an engine busy with long prefill batches (10-15s per batch)
     * keeps timing out on sync without being disconnected. Removing it drains the
     * batcher and fails every queued request with 8510, so this tier tolerates far
     * more consecutive failures than the connection tier.
     * <p>Environment variable: FLEXLB_SYNC_TIMEOUT_MAX_CONSECUTIVE_FAILURES.
     */
    private int flexlbSyncTimeoutMaxConsecutiveFailures = 10;

    /**
     * Max consecutive connection-class (UNAVAILABLE / IO / any non-timeout)
     * GetWorkerStatus failures before a worker is marked dead and its endpoint
     * removed. Preserves the original hard-failure semantics.
     * <p>Environment variable: FLEXLB_SYNC_HARD_MAX_CONSECUTIVE_FAILURES.
     */
    private int flexlbSyncHardMaxConsecutiveFailures = 3;

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
     * to reserve prefill inflight locally or defer to FlexlbBatchScheduler.
     */
    public boolean isBatchPath() {
        return getDefaultScheduleModeEnum() == ScheduleModeEnum.BATCH;
    }

    /**
     * Returns the effective maxNewTokens to use for local KV-reservation
     * estimation. When {@link #maxNewTokensCap} is positive, the declared
     * value is clamped to the cap; when 0, the declared value is
     * returned unchanged. The request's original maxNewTokens is never
     * modified by this method — it only computes a local estimate.
     */
    public long effectiveMaxNewTokensForReservation(long declared) {
        return maxNewTokensCap > 0 ? Math.min(declared, maxNewTokensCap) : declared;
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
