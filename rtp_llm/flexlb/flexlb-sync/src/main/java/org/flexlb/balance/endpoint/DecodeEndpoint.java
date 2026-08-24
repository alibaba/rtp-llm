package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.InflightEvictor;
import org.flexlb.balance.scheduler.InflightItem;
import org.flexlb.balance.scheduler.InflightState;
import org.flexlb.balance.scheduler.InflightStore;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.TaskPhase;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.state.DecodeEndpointCounters;
import org.flexlb.sync.shadow.StateShadowBridge;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Decode worker endpoint with two-layer inflight tracking:
 *
 * <ol>
 *   <li>layer 1 ({@link #inflightRequests}) — reserved locally, not yet
 *       reported by the engine. Acceptance boundary is unified with
 *       {@link PrefillEndpoint}: any phase appearing in runningTaskInfo means
 *       the engine has taken ownership. KV reservation counters are tied to
 *       this layer only.</li>
 *   <li>layer 2 ({@link #engineWork}) — engine-accepted tasks keyed by
 *       requestId with a phase (PENDING/RECEIVED → WAITING, KV_ALLOCATED →
 *       LOADING, RUNNING → RUNNING) and lastSeenRound. Replaces the legacy
 *       flat {@code confirmedRunningCount} counter with per-request phase
 *       visibility via {@code countEngineWorkInPhase(RUNNING)}.</li>
 * </ol>
 */
public class DecodeEndpoint extends WorkerEndpoint {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");

    /** Layer 1: reserved locally, not yet accepted by the engine. */
    private final ConcurrentHashMap<Long, RequestInflight> inflightRequests = new ConcurrentHashMap<>();
    private final AtomicLong inflightKvReservedTotal = new AtomicLong(0);
    private final AtomicLong inflightExpectedKvReservedTotal = new AtomicLong(0);
    private final AtomicLong reportedKvAvailable = new AtomicLong();

    /** Layer 2: engine-accepted tasks with phase state and lastSeenRound. */
    private final ConcurrentHashMap<Long, EngineTask<RequestInflight>> engineWork = new ConcurrentHashMap<>();

    private final InflightEvictor<Long, RequestInflight> requestEvictor;
    private final InflightEvictor<Long, EngineTask<RequestInflight>> engineWorkEvictor;
    private final InflightStore inflightStore;
    private final FlexlbConfig config;

    // ---- G4 读取换权（调度读点与 KV 记账切 StateLedger per-EP 视图） ----

    /** 影子门面（装配点注入；测试可传 null——完全退回旧双层 map 行为）。 */
    private final StateShadowBridge shadowBridge;

    /** 端点稳定 ID（ipPort 哈希——与影子 translator 的 endpointId 同映射）。 */
    private final int endpointId;

    /**
     * per-EP 账本计数缓存：calibrate tick（引擎状态报文）刷新，读点零锁
     * volatile 读——策略 select 热路径不触发按需聚合。关态恒 null。
     */
    private volatile DecodeEndpointCounters ledgerCounters;

    /** Monotonic calibrate round counter driving stale engineWork eviction. */
    private final AtomicLong calibrateRound = new AtomicLong(0);

    public DecodeEndpoint(WorkerStatus status, FlexlbConfig config, InflightStore inflightStore) {
        this(status, config, inflightStore, null);
    }

    /**
     * 读取换权装配构造：注入影子门面后（读取换权开启时）KV 预占记账与
     * 全部调度读点切 StateLedger per-EP 视图；门面为 null / 开关关时保持
     * 旧双层 map 行为（零行为变化）。
     */
    public DecodeEndpoint(WorkerStatus status, FlexlbConfig config, InflightStore inflightStore,
                          StateShadowBridge shadowBridge) {
        super(status);
        this.config = config;
        this.inflightStore = inflightStore;
        this.shadowBridge = shadowBridge;
        this.endpointId = ipPort() != null ? ipPort().hashCode() : 0;
        this.requestEvictor = new InflightEvictor<>(inflightRequests, req -> {
            inflightKvReservedTotal.addAndGet(-req.kvTokens());
            inflightExpectedKvReservedTotal.addAndGet(-req.expectedKvTokens());
        });
        // Layer-2 KV reservations were already released on acceptance, so no
        // counter adjustment is needed on eviction.
        this.engineWorkEvictor = new InflightEvictor<>(engineWork, null);
    }

    /** 读取换权模式判定（门面 null / 开关关时 false——一切走旧路径）。 */
    private boolean readAuthority() {
        return shadowBridge != null && shadowBridge.isReadAuthority();
    }

    /** per-EP 账本计数（未刷新时全零——读数退化方向为低估，不阻断调度）。 */
    private DecodeEndpointCounters ledgerCountersOrZero() {
        DecodeEndpointCounters c = ledgerCounters;
        return c != null ? c : DecodeEndpointCounters.empty();
    }

    @Override
    public void close() {
        try {
            drainInflight("EP closed");
        } finally {
            super.close();
        }
    }

    /**
     * Drain all tracked inflight entries from both layers, terminating their
     * bound {@link InflightItem}s so clients are notified immediately instead
     * of waiting for the 300s TTL safety net (review A4).
     */
    private void drainInflight(String reason) {
        List<InflightItem> toTerminate = new ArrayList<>();
        for (Long requestId : engineWork.keySet()) {
            collectItem(requestId, toTerminate);
        }
        for (Long requestId : inflightRequests.keySet()) {
            collectItem(requestId, toTerminate);
        }
        engineWork.clear();
        inflightRequests.clear();
        inflightKvReservedTotal.set(0);
        inflightExpectedKvReservedTotal.set(0);
        for (InflightItem item : toTerminate) {
            if (!item.isTerminated()) {
                item.complete(Response.error(StrategyErrorType.WORKER_EXECUTION_FAILED, reason),
                        InflightState.FAILED);
            }
        }
    }

    /**
     * Look up an {@link InflightItem} by requestId and add it to the sink if
     * found and not already terminal. Null-safe on {@code inflightStore}.
     */
    private void collectItem(long requestId, List<InflightItem> sink) {
        if (inflightStore == null) return;
        InflightItem item = inflightStore.get(String.valueOf(requestId));
        if (item != null && !item.isTerminated()) {
            sink.add(item);
        }
    }

    /**
     * Terminate a single bound {@link InflightItem} by requestId. Used by
     * STALE eviction to drive items terminal immediately (review A3).
     */
    private void terminateBoundItem(long requestId, String reason) {
        if (inflightStore == null) return;
        InflightItem item = inflightStore.get(String.valueOf(requestId));
        if (item != null && !item.isTerminated()) {
            item.complete(Response.error(StrategyErrorType.WORKER_EXECUTION_FAILED, reason),
                    InflightState.FAILED);
        }
    }

    public void reserve(long requestId, long kvTokens, long expectedKvTokens) {
        if (readAuthority()) {
            // G4 读取换权：真实记账切 ledger.decode().reserve 单入账（含世代
            // 绑定与判重）；旧 layer-1 map 停写。异常由门面吞入 shadow.error，
            // 不回写旧 map（读数与记账同源，避免双源分裂）。
            shadowBridge.decodeReserveAuthority(requestId, kvTokens, expectedKvTokens, ipPort());
            return;
        }
        RequestInflight newRi = new RequestInflight(kvTokens, expectedKvTokens);
        // Atomic compute eliminates the TOCTOU window between putIfAbsent and put:
        // if calibrate's removeInflight ran between the two steps, the old value
        // was already subtracted, and the subsequent addAndGet(-prev) would
        // double-decrement → inflightKvReservedTotal goes negative → over-admission.
        // compute holds the bin lock, so no concurrent remove can interleave.
        inflightRequests.compute(requestId, (key, prev) -> {
            if (prev != null) {
                // requestId already exists — subtract the old kvTokens before replacing,
                // otherwise the old value is silently lost and the counter stays inflated.
                inflightKvReservedTotal.addAndGet(-prev.kvTokens());
                inflightExpectedKvReservedTotal.addAndGet(-prev.expectedKvTokens());
            }
            inflightKvReservedTotal.addAndGet(kvTokens);
            inflightExpectedKvReservedTotal.addAndGet(expectedKvTokens);
            return newRi;
        });
    }

    /**
     * Release the layer-1 reservation for {@code requestId}.
     * <p>Layer 2 is intentionally untouched: engineWork mirror what the
     * engine reports as accepted, so entries leave via finishedTaskInfo or
     * stale-round eviction — same coverage the legacy flat counter had
     * (recomputed purely from reports via
     * {@code countEngineWorkInPhase(RUNNING)}).
     */
    @Override
    public void release(long requestId) {
        if (readAuthority()) {
            // G4 读取换权：释放也切 ledger（未终态主动放弃；终局归 settle
            // 单出口）。旧 layer-1 map 在此模式下恒空，removeInflight 无需执行。
            shadowBridge.decodeReleaseAuthority(requestId);
            return;
        }
        removeInflight(requestId);
    }

    @Override
    public void onWorkerStatusUpdate(WorkerStatus ws, WorkerStatusResponse resp) {
        super.onWorkerStatusUpdate(ws, resp);
        calibrate(resp.getRunningTaskInfo(), resp.getFinishedTaskInfo());
    }

    /**
     * Full calibration against worker status report, driving the two-layer
     * inflight state machine (symmetric with {@link PrefillEndpoint}):
     *
     * <ol>
     *   <li>acceptance — any request appearing in runningTaskInfo migrates
     *       from layer 1 to layer 2 with its mapped phase
     *       (WAITING/LOADING/RUNNING); already-migrated tasks get
     *       phase/lastSeenRound refreshed.
     *       Migration inserts into layer 2 <b>before</b> releasing the
     *       layer-1 KV reservation (conservative order: the transient window
     *       over-counts rather than under-counts — same rationale as the
     *       legacy two-pass scan).</li>
     *   <li>completion — finished tasks are removed from whichever layer
     *       tracks them (fast path: finished while still in layer 1).</li>
     *   <li>staleness — engineWork entries absent from reports for
     *       {@code flexlbStaleEvictRounds} consecutive rounds are evicted.</li>
     * </ol>
     */
    private void calibrate(Map<String, TaskInfo> runningTaskInfo, Map<String, TaskInfo> finishedTaskInfo) {
        this.reportedKvAvailable.set(status.getAvailableKvCacheTokens().get());
        if (readAuthority()) {
            // G4 读取换权：per-EP 账本计数缓存随引擎状态 tick 刷新
            //（旧双层 map 在此模式下不再产生调度读数）。
            this.ledgerCounters = shadowBridge.decodeEndpointCounters(endpointId);
        }
        long statusMs = System.currentTimeMillis();
        long round = calibrateRound.incrementAndGet();

        observeRunningTasks(runningTaskInfo, round, statusMs);
        processFinishedTasks(finishedTaskInfo);
        evictStaleEngineWork(round);
    }

    /**
     * Acceptance step: migrate accepted requests layer 1 → layer 2 and
     * refresh phases of already-accepted tasks.
     *
     * <p>Foreign key pre-check: when a requestId is not in layer 1
     * (inflightRequests) or layer 2 (engineWork), we consult the global
     * {@link InflightStore} to distinguish three cases:
     * <ol>
     *   <li><b>Foreign key</b> — store.get returns null: the requestId
     *       belongs to another master (multi-master failover). Do NOT create
     *       an engineWork entry — it is not ours to track.</li>
     *   <li><b>Already terminal</b> — store.get returns a terminal item: the
     *       request has finished/cancelled but the engine still reports it as
     *       running (stale report). Skip.</li>
     *   <li><b>Cross-EP failover</b> — store.get returns a RUNNING item: the
     *       request was dispatched by this master but the local reservation was
     *       lost (e.g. EP restart). Create an engineWork entry with KV estimated
     *       from TaskInfo input_length instead of (0,0) to avoid under-estimating
     *       decode load. Formula: {@code kvTokens = max(inputLength, defaultKvTokens)},
     *       {@code expectedKv = kvTokens + maxNewTokens}.</li>
     * </ol>
     */
    private void observeRunningTasks(Map<String, TaskInfo> runningTaskInfo, long round, long statusMs) {
        if (runningTaskInfo == null || runningTaskInfo.isEmpty()) {
            return;
        }
        for (TaskInfo task : runningTaskInfo.values()) {
            long requestId = task.getRequestId();
            TaskPhase reported = task.getPhase();

            EngineTask<RequestInflight> existing = engineWork.get(requestId);
            if (existing != null) {
                existing.observe(EngineTaskPhase.fromDecode(reported), round, statusMs);
                continue;
            }

            // Unified acceptance boundary (same as PrefillEndpoint): any phase
            // reported in runningTaskInfo means the engine has taken ownership,
            // so the request migrates to layer 2 and its KV reservation is
            // released (engine now accounts for it in its own reports).
            RequestInflight reserved = inflightRequests.get(requestId);
            if (reserved != null) {
                // Normal path: locally reserved, migrate to layer 2
                engineWork.putIfAbsent(requestId,
                        new EngineTask<>(reserved, EngineTaskPhase.fromDecode(reported), round, statusMs));
                removeInflight(requestId);
            } else if (inflightStore != null) {
                // Foreign key pre-check: not in layer 1 or layer 2.
                // Check the global InflightStore to decide whether to track.
                InflightItem item = inflightStore.get(String.valueOf(requestId));
                if (item == null || item.state().isTerminal()) {
                    // Foreign key (another master) or already terminal — skip
                    logger.debug("Decode calibrate: running reqId={} not in inflightRequests/engineWork "
                            + "and not in store (foreign key or terminal), skipping", requestId);
                    continue;
                }
                // Cross-EP failover: estimate KV from TaskInfo input_length
                long inputLen = task.getInputLength();
                long kvTokens = inputLen > 0 ? inputLen : config.getDefaultKvTokens();
                long expectedKv = kvTokens + config.getMaxNewTokens();
                RequestInflight fallback = new RequestInflight(kvTokens, expectedKv);
                engineWork.putIfAbsent(requestId,
                        new EngineTask<>(fallback, EngineTaskPhase.fromDecode(reported), round, statusMs));
                logger.info("Decode calibrate: cross-EP failover reqId={} estimated kvTokens={} expectedKv={}",
                        requestId, kvTokens, expectedKv);
            } else {
                // No inflightStore (tests) — fall back to (0,0) as before
                RequestInflight payload = new RequestInflight(0, 0);
                engineWork.putIfAbsent(requestId,
                        new EngineTask<>(payload, EngineTaskPhase.fromDecode(reported), round, statusMs));
            }
        }
    }

    /**
     * Completion step: a finished request leaves whichever layer tracks it.
     * Cross-round fast path: finished before ever being observed running —
     * removed straight from layer 1 (KV reservation released).
     */
    private void processFinishedTasks(Map<String, TaskInfo> finishedTaskInfo) {
        if (finishedTaskInfo == null || finishedTaskInfo.isEmpty()) {
            return;
        }
        for (TaskInfo task : finishedTaskInfo.values()) {
            long requestId = task.getRequestId();
            boolean removedTask = engineWork.remove(requestId) != null;
            boolean removedInflight = removeInflight(requestId);
            if (task.getErrorCode() != 0 && !removedTask && !removedInflight) {
                logger.debug("Decode calibrate: finished failed request reqId={} not tracked, error={}",
                        requestId, task.getErrorMessage());
            }
        }
    }

    /**
     * Staleness step: evict engineWork entries that have been absent from both
     * running and finished reports for {@code flexlbStaleEvictRounds}
     * consecutive calibrate rounds (lost completion report).
     */
    private void evictStaleEngineWork(long round) {
        for (Map.Entry<Long, EngineTask<RequestInflight>> entry : engineWork.entrySet()) {
            EngineTask<RequestInflight> task = entry.getValue();
            if (round - task.lastSeenRound() < config.getFlexlbStaleEvictRounds()) {
                continue;
            }
            if (engineWork.remove(entry.getKey(), task)) {
                logger.warn("Decode calibrate: engineWork reqId={} phase={} unseen for {} rounds, evicting as stale",
                        entry.getKey(), task.phase(), round - task.lastSeenRound());
                // A3: STALE eviction now drives the bound InflightItem to a
                // terminal state so the client future is settled in seconds,
                // not the 300s TTL safety net.
                terminateBoundItem(entry.getKey(), "engine evicted as stale");
            }
        }
    }

    /** Remove a layer-1 entry and release its KV reservation counters. */
    private boolean removeInflight(long requestId) {
        RequestInflight removed = inflightRequests.remove(requestId);
        if (removed == null) {
            return false;
        }
        inflightKvReservedTotal.addAndGet(-removed.kvTokens());
        inflightExpectedKvReservedTotal.addAndGet(-removed.expectedKvTokens());
        return true;
    }

    // ==================== 新三视图（显式接口） ====================

    /** Layer-1 entry count: dispatched, not yet accepted by the engine. */
    public int decodeInflightCount() {
        if (readAuthority()) {
            return ledgerCountersOrZero().unconfirmedCount();
        }
        return inflightRequests.size();
    }

    /**
     * Layer-1 Σ kvTokens (hard demand) — seqLen only, the minimum KV needed
     * for the prompt itself. Used for hard-capacity filtering.
     * Backed by {@code inflightKvReservedTotal} — O(1) incremental maintenance.
     */
    public long decodeInflightHardKvReserved() {
        if (readAuthority()) {
            return ledgerCountersOrZero().unconfirmedSeqKv();
        }
        return inflightKvReservedTotal.get();
    }

    /**
     * Layer-1 Σ expectedKvTokens (conservative estimate) — seqLen +
     * maxNewTokens, accounting for generation-phase KV growth. Used for
     * scoring / load balancing.
     * Backed by {@code inflightExpectedKvReservedTotal} — O(1) incremental maintenance.
     */
    public long decodeInflightExpectedKvReserved() {
        if (readAuthority()) {
            return ledgerCountersOrZero().unconfirmedExpectedKv();
        }
        return inflightExpectedKvReservedTotal.get();
    }

    /** Layer-2 task count: requests the engine has accepted (LOADING/RUNNING). */
    public int decodeEngineWorkCount() {
        if (readAuthority()) {
            return ledgerCountersOrZero().engineOwnedCount();
        }
        return engineWork.size();
    }

    /** Layer-2 tasks currently in the WAITING phase. */
    public int decodeEngineWaitingCount() {
        if (readAuthority()) {
            // 账本口径：已派发未确认（DISPATCHED）≈ 引擎侧排队窗口
            return (int) phaseCountOrZero(DecodePhaseOrdinal.DISPATCHED);
        }
        return countEngineWorkInPhase(EngineTaskPhase.WAITING);
    }

    /** Layer-2 tasks currently in the LOADING phase (remote KV loading). */
    public int decodeEngineLoadingCount() {
        if (readAuthority()) {
            return (int) phaseCountOrZero(DecodePhaseOrdinal.D_LOADING);
        }
        return countEngineWorkInPhase(EngineTaskPhase.LOADING);
    }

    /** Layer-2 tasks currently in the RUNNING phase. */
    public int decodeEngineRunningCount() {
        if (readAuthority()) {
            return (int) phaseCountOrZero(DecodePhaseOrdinal.D_RUNNING);
        }
        return countEngineWorkInPhase(EngineTaskPhase.RUNNING);
    }

    /** 账本相位人口下标（与 flexlb-state DecodePhase.ordinal 对齐）。 */
    private static final class DecodePhaseOrdinal {
        static final int RESERVED = 0;
        static final int DISPATCHED = 1;
        static final int D_LOADING = 2;
        static final int D_RUNNING = 3;
    }

    private long phaseCountOrZero(int ordinal) {
        DecodeEndpointCounters c = ledgerCountersOrZero();
        List<Long> phases = c.phaseCounts();
        return ordinal < phases.size() ? phases.get(ordinal) : 0L;
    }

    private int countEngineWorkInPhase(EngineTaskPhase phase) {
        int count = 0;
        for (EngineTask<RequestInflight> task : engineWork.values()) {
            if (task.phase() == phase) {
                count++;
            }
        }
        return count;
    }

    /**
     * Total active load: engine-accepted tasks + local inflight
     * (countEngineWorkInPhase(RUNNING) + inflight in legacy terms).
     * 读取换权模式下读账本 per-EP 活跃条目数（reserve 起至终局，全相位）。
     */
    public int decodeTotalLoad() {
        if (readAuthority()) {
            return ledgerCountersOrZero().activeTotal();
        }
        return engineWork.size() + inflightRequests.size();
    }

    /**
     * Real KV used: engine-reported used (total - available) + local inflight
     * expected reservations.
     * 读取换权模式下预占读账本 per-EP 未确认条目 Σ expectedKv（口径一致）。
     */
    public long decodeRealKvUsed() {
        long totalCap = status.getTotalKvCacheTokens().get();
        long avail = status.getAvailableKvCacheTokens().get();
        long reportedUsed = totalCap > 0 ? Math.max(0, totalCap - avail) : 0;
        if (readAuthority()) {
            return reportedUsed + ledgerCountersOrZero().unconfirmedExpectedKv();
        }
        return reportedUsed + decodeInflightExpectedKvReserved();
    }

    /**
     * Real KV available: engine-reported available - local inflight hard reservations.
     *
     * <p>Uses {@link #decodeInflightHardKvReserved()} (prompt-only KV) rather than
     * {@link #decodeInflightExpectedKvReserved()} (expected KV with generation) so
     * that the hard-capacity filter only checks whether the prompt itself fits,
     * without being overly aggressive due to other inflight requests' expected growth.
     *
     * <p><b>Approximate:</b> reads {@code reportedKvAvailable} and
     * computes {@code decodeInflightHardKvReserved()} non-atomically — the returned
     * value may reflect a slightly inconsistent snapshot. This is acceptable for
     * scheduling decisions.
     */
    public long decodeRealKvAvailable() {
        if (readAuthority()) {
            // 账本口径：硬预占 = 未确认条目 Σ seqLen（与旧 layer-1 hard 口径一致）
            return Math.max(0, reportedKvAvailable.get() - ledgerCountersOrZero().unconfirmedSeqKv());
        }
        return Math.max(0, reportedKvAvailable.get() - decodeInflightHardKvReserved());
    }

    /** Real KV total capacity reported by the engine. */
    public long decodeKvTotal() {
        return status.getTotalKvCacheTokens().get();
    }

    // ==================== Metrics ====================

    /**
     * Report per-worker decode inflight metrics via the given reporter.
     * Called periodically by {@code RouteService#triggerSchedulerMetrics()}.
     */
    public void reportBatchMetrics(BatchSchedulerReporter reporter) {
        reporter.reportInflightRequestCount(RoleType.DECODE.name(), getIp(), decodeInflightCount());
        reporter.reportDecodeTotalLoad(getIp(), decodeTotalLoad());
        reporter.reportDecodeInflightKvReserved(getIp(), decodeInflightExpectedKvReserved());
        reporter.reportDecodeInflightKvReservedHard(getIp(), decodeInflightHardKvReserved());
        // Phase-split layer-2 counts (WAITING / LOADING / RUNNING)
        reporter.reportDecodeEngineWaitingCount(getIp(), decodeEngineWaitingCount());
        reporter.reportDecodeEngineLoadingCount(getIp(), decodeEngineLoadingCount());
        reporter.reportDecodeEngineRunningCount(getIp(), decodeEngineRunningCount());
        // Two-layer breakdown
        reporter.reportDecodeInflightRequestsCount(getIp(), decodeInflightCount());
        reporter.reportDecodeEngineWorkCount(getIp(), decodeEngineWorkCount());
    }

    // ==================== Eviction ====================

    /**
     * Evict layer-1 inflight requests older than {@code ttlMs} (lost dispatch
     * backstop). Layer 2 has its own wall-clock backstop via
     * {@link #evictExpiredEngineWork(long)}.
     *
     * @return number of entries evicted
     */
    public int evictExpiredRequests(long ttlMs) {
        return requestEvictor.evictExpired(ttlMs);
    }

    /**
     * Evict layer-2 engineWork entries accepted more than {@code ttlMs} ago.
     * Backstop for a worker that stops reporting entirely: calibrate rounds
     * no longer advance, so stale-round eviction cannot fire. Decode tasks
     * legitimately run for a long time (generation), so callers should pass
     * a generous TTL relative to the worst-case generation time.
     *
     * <p>Called by {@code EndpointRegistry.scheduledEviction()} with the
     * EP-level TTL ({@code flexlbEpInflightTtlMs}, default 600s) — longer
     * than the scheduler-level TTL ({@code flexlbInflightTtlMs}, default 300s)
     * because engine-accepted decode tasks run longer than prefill dispatch.
     *
     * @return number of entries evicted
     */
    public int evictExpiredEngineWork(long ttlMs) {
        return engineWorkEvictor.evictExpired(ttlMs);
    }

    // ==================== test hooks ====================

    /** Package-private test hook: current phase of an engineWork entry, or null. */
    EngineTaskPhase engineWorkPhase(long requestId) {
        EngineTask<RequestInflight> task = engineWork.get(requestId);
        return task != null ? task.phase() : null;
    }
}
