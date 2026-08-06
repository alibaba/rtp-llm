package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.InflightEvictor;
import org.flexlb.balance.scheduler.InflightItem;
import org.flexlb.balance.scheduler.InflightStore;
import org.flexlb.balance.scheduler.TerminalReason;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.TaskPhase;
import org.flexlb.service.monitor.BatchSchedulerReporter;
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
 *   <li>layer 2 ({@link #engineTasks}) — engine-accepted tasks keyed by
 *       requestId with a phase (PENDING/RECEIVED → WAITING, KV_ALLOCATED →
 *       LOADING, RUNNING → RUNNING) and lastSeenRound. Replaces the legacy
 *       {@code confirmedRunningCount} flat counter with per-request phase
 *       visibility.</li>
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
    private final ConcurrentHashMap<Long, EngineTask<RequestInflight>> engineTasks = new ConcurrentHashMap<>();

    private final InflightEvictor<Long, RequestInflight> requestEvictor;
    private final InflightEvictor<Long, EngineTask<RequestInflight>> engineTaskEvictor;
    private final InflightStore inflightStore;

    /** Monotonic calibrate round counter driving stale engine-task eviction. */
    private final AtomicLong calibrateRound = new AtomicLong(0);

    /**
     * Evict an engine task absent from both running and finished reports for
     * this many consecutive calibrate rounds (lost completion report).
     * TODO(config): couple with the worker-status sync interval once the
     * config wiring lands with the consumer-migration phase.
     */
    private static final int STALE_EVICT_ROUNDS = 3;

    public DecodeEndpoint(WorkerStatus status, InflightStore inflightStore) {
        super(status);
        this.inflightStore = inflightStore;
        this.requestEvictor = new InflightEvictor<>(inflightRequests, req -> {
            inflightKvReservedTotal.addAndGet(-req.kvTokens());
            inflightExpectedKvReservedTotal.addAndGet(-req.expectedKvTokens());
        });
        // Layer-2 KV reservations were already released on acceptance, so no
        // counter adjustment is needed on eviction.
        this.engineTaskEvictor = new InflightEvictor<>(engineTasks, null);
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
        for (Long requestId : engineTasks.keySet()) {
            collectItem(requestId, toTerminate);
        }
        for (Long requestId : inflightRequests.keySet()) {
            collectItem(requestId, toTerminate);
        }
        engineTasks.clear();
        inflightRequests.clear();
        inflightKvReservedTotal.set(0);
        inflightExpectedKvReservedTotal.set(0);
        for (InflightItem item : toTerminate) {
            if (!item.isTerminated()) {
                item.terminate(TerminalReason.FAILED, new RuntimeException(reason));
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
            item.terminate(TerminalReason.FAILED, new RuntimeException(reason));
        }
    }

    public void reserve(long requestId, long kvTokens, long expectedKvTokens) {
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
     * <p>Layer 2 is intentionally untouched: engineTasks mirror what the
     * engine reports as accepted, so entries leave via finishedTaskInfo or
     * stale-round eviction — same coverage the legacy
     * {@code confirmedRunningCount} had (recomputed purely from reports).
     */
    @Override
    public void release(long requestId) {
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
     *   <li>staleness — engine tasks absent from reports for
     *       {@link #STALE_EVICT_ROUNDS} consecutive rounds are evicted.</li>
     * </ol>
     */
    private void calibrate(Map<String, TaskInfo> runningTaskInfo, Map<String, TaskInfo> finishedTaskInfo) {
        this.reportedKvAvailable.set(status.getAvailableKvCacheTokens().get());
        long statusMs = System.currentTimeMillis();
        long round = calibrateRound.incrementAndGet();

        observeRunningTasks(runningTaskInfo, round, statusMs);
        processFinishedTasks(finishedTaskInfo);
        evictStaleEngineTasks(round);
    }

    /**
     * Acceptance step: migrate accepted requests layer 1 → layer 2 and
     * refresh phases of already-accepted tasks.
     */
    private void observeRunningTasks(Map<String, TaskInfo> runningTaskInfo, long round, long statusMs) {
        if (runningTaskInfo == null || runningTaskInfo.isEmpty()) {
            return;
        }
        for (TaskInfo task : runningTaskInfo.values()) {
            long requestId = task.getRequestId();
            TaskPhase reported = task.getPhase();

            EngineTask<RequestInflight> existing = engineTasks.get(requestId);
            if (existing != null) {
                existing.observe(EngineTaskPhase.fromDecode(reported), round, statusMs);
                continue;
            }

            // Unified acceptance boundary (same as PrefillEndpoint): any phase
            // reported in runningTaskInfo means the engine has taken ownership,
            // so the request migrates to layer 2 and its KV reservation is
            // released (engine now accounts for it in its own reports).

            // A task reported accepted but never reserved locally (e.g. traffic
            // from another master) is still tracked with an empty reservation so
            // decodeTotalLoad keeps the legacy confirmedRunningCount coverage.
            RequestInflight reserved = inflightRequests.get(requestId);
            RequestInflight payload = reserved != null ? reserved : new RequestInflight(0, 0);
            engineTasks.putIfAbsent(requestId,
                    new EngineTask<>(payload, EngineTaskPhase.fromDecode(reported), round, statusMs));
            removeInflight(requestId);
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
            boolean removedTask = engineTasks.remove(requestId) != null;
            boolean removedInflight = removeInflight(requestId);
            if (task.getErrorCode() != 0 && !removedTask && !removedInflight) {
                logger.debug("Decode calibrate: finished failed request reqId={} not tracked, error={}",
                        requestId, task.getErrorMessage());
            }
        }
    }

    /**
     * Staleness step: evict engine tasks that have been absent from both
     * running and finished reports for {@link #STALE_EVICT_ROUNDS}
     * consecutive calibrate rounds (lost completion report).
     */
    private void evictStaleEngineTasks(long round) {
        for (Map.Entry<Long, EngineTask<RequestInflight>> entry : engineTasks.entrySet()) {
            EngineTask<RequestInflight> task = entry.getValue();
            if (round - task.lastSeenRound() < STALE_EVICT_ROUNDS) {
                continue;
            }
            if (engineTasks.remove(entry.getKey(), task)) {
                logger.warn("Decode calibrate: engine task reqId={} phase={} unseen for {} rounds, evicting as stale",
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
        return inflightRequests.size();
    }

    /**
     * Layer-1 Σ kvTokens (hard demand) — seqLen only, the minimum KV needed
     * for the prompt itself. Used for hard-capacity filtering.
     * Backed by {@code inflightKvReservedTotal} — O(1) incremental maintenance.
     */
    public long decodeInflightHardKvReserved() {
        return inflightKvReservedTotal.get();
    }

    /**
     * Layer-1 Σ expectedKvTokens (conservative estimate) — seqLen +
     * maxNewTokens, accounting for generation-phase KV growth. Used for
     * scoring / load balancing.
     * Backed by {@code inflightExpectedKvReservedTotal} — O(1) incremental maintenance.
     */
    public long decodeInflightExpectedKvReserved() {
        return inflightExpectedKvReservedTotal.get();
    }

    /** Layer-2 task count: requests the engine has accepted (LOADING/RUNNING). */
    public int decodeEngineTaskCount() {
        return engineTasks.size();
    }

    /** Layer-2 tasks currently in the WAITING phase. */
    public int decodeEngineWaitingCount() {
        return countEngineTasksInPhase(EngineTaskPhase.WAITING);
    }

    /** Layer-2 tasks currently in the LOADING phase (remote KV loading). */
    public int decodeEngineLoadingCount() {
        return countEngineTasksInPhase(EngineTaskPhase.LOADING);
    }

    /** Layer-2 tasks currently in the RUNNING phase. */
    public int decodeEngineRunningCount() {
        return countEngineTasksInPhase(EngineTaskPhase.RUNNING);
    }

    private int countEngineTasksInPhase(EngineTaskPhase phase) {
        int count = 0;
        for (EngineTask<RequestInflight> task : engineTasks.values()) {
            if (task.phase() == phase) {
                count++;
            }
        }
        return count;
    }

    /**
     * Total active load: engine-accepted tasks + local inflight
     * (confirmedRunningCount + inflight in legacy terms).
     */
    public int decodeTotalLoad() {
        return engineTasks.size() + inflightRequests.size();
    }

    /**
     * Real KV used: engine-reported used (total - available) + local inflight
     * expected reservations.
     */
    public long decodeRealKvUsed() {
        long totalCap = status.getTotalKvCacheTokens().get();
        long avail = status.getAvailableKvCacheTokens().get();
        long reportedUsed = totalCap > 0 ? Math.max(0, totalCap - avail) : 0;
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
        reporter.reportDecodeEngineTasksCount(getIp(), decodeEngineTaskCount());
    }

    // ==================== Eviction ====================

    /**
     * Evict layer-1 inflight requests older than {@code ttlMs} (lost dispatch
     * backstop). Layer 2 has its own wall-clock backstop via
     * {@link #evictExpiredEngineTasks(long)}.
     *
     * @return number of entries evicted
     */
    public int evictExpiredRequests(long ttlMs) {
        return requestEvictor.evictExpired(ttlMs);
    }

    /**
     * Evict layer-2 engine tasks accepted more than {@code ttlMs} ago.
     * Backstop for a worker that stops reporting entirely: calibrate rounds
     * no longer advance, so stale-round eviction cannot fire. Decode tasks
     * legitimately run for a long time (generation), so callers should pass
     * a generous TTL relative to the worst-case generation time.
     *
     * @return number of entries evicted
     */
    public int evictExpiredEngineTasks(long ttlMs) {
        return engineTaskEvictor.evictExpired(ttlMs);
    }

    // ==================== test hooks ====================

    /** Package-private test hook: current phase of an engine task, or null. */
    EngineTaskPhase engineTaskPhase(long requestId) {
        EngineTask<RequestInflight> task = engineTasks.get(requestId);
        return task != null ? task.phase() : null;
    }
}
