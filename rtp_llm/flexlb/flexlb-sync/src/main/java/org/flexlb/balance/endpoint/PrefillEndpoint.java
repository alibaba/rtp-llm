package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.BatchDecisionHandler;
import org.flexlb.balance.scheduler.BatchItem;
import org.flexlb.balance.scheduler.InflightEvictor;
import org.flexlb.balance.scheduler.WorkerBatcher;
import org.flexlb.balance.strategy.FormulaPredictor;
import org.flexlb.balance.strategy.LearningPredictor;
import org.flexlb.balance.strategy.PrefillTimePredictor;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.PriorityPreemptionProgress;
import org.flexlb.enums.TaskPhase;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.LongPredicate;
import java.util.stream.Collectors;

public class PrefillEndpoint extends WorkerEndpoint {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");

    private record FinishedObservation(long requestId,
                                       long executionTimeMs,
                                       long errorCode,
                                       String errorMessage) {
        static FinishedObservation from(TaskInfo task) {
            return new FinishedObservation(task.getRequestId(), task.getExecutionTimeMs(),
                    task.getErrorCode(), task.getErrorMessage());
        }

        FinishedObservation merge(FinishedObservation other) {
            long mergedErrorCode = errorCode != 0 ? errorCode : other.errorCode;
            String mergedErrorMessage = errorCode != 0 ? errorMessage : other.errorMessage;
            return new FinishedObservation(requestId,
                    Math.max(executionTimeMs, other.executionTimeMs),
                    mergedErrorCode, mergedErrorMessage);
        }
    }

    private record ReconciliationState(FinishedObservation deferredTerminal) {}

    private final PrefillTimePredictor predictor;
    private final ConcurrentHashMap<Long, BatchInflight> inflightBatches = new ConcurrentHashMap<>();
    private final ConcurrentHashMap<Long, ConcurrentHashMap<Long, ReconciliationState>>
            reconciliationRequests = new ConcurrentHashMap<>();
    private final AtomicInteger inflightRequestCount = new AtomicInteger(0);
    private final WorkerBatcher batcher;
    private final BatchSchedulerReporter reporter;
    /**
     * The batch decision handler (the FlexlbBatchScheduler in production).
     * Retained so the batch-level age cap can route each member of a
     * force-settled batch through the existing handler terminal chain
     * ({@link BatchDecisionHandler#onExpired}) instead of leaving the
     * scheduler-side inflight entries dangling after the endpoint ledger
     * entry is removed.
     */
    private final BatchDecisionHandler handler;

    /**
     * Active Engine tasks not already represented in the local batch ledger.
     * Atomic so the age-cap release (engine-untracked compensation) can add the
     * force-settled members back concurrently with the status-sync
     * recomputation.
     */
    private final AtomicLong engineUntrackedRequestCount = new AtomicLong(0);

    /**
     * Raw engine-reported waiting (queued) query count from the last worker
     * status sync — {@code WorkerStatusResponse.waiting_query_len}, clamped
     * to {@code >= 0}. The Engine reports every ~20ms, so the value is a
     * near-real-time view of the engine-side admission queue depth that the
     * master cannot see through its own ledgers (e.g. slow engines whose
     * local batcher queue was drained by dispatch but whose engine-side
     * queue keeps growing).
     *
     * <p>Last-known semantics: the value is kept as-is between syncs and is
     * NOT decayed — a stale high value keeps signaling a slow engine, and a
     * lost engine is handled by the registry removing the endpoint entirely,
     * so no extra staleness handling is needed here.
     */
    private volatile long reportedWaitingQueryLen = 0;

    private static final long WAIT_TIME_CACHE_TTL_MS = 2;
    private volatile long cachedWaitTimeMs = 0;
    private volatile long cachedWaitTimeExpireAtMs = 0;

    public PrefillEndpoint(WorkerStatus status, FlexlbConfig config,
                           BatchDecisionHandler handler,
                           BatchSchedulerReporter reporter) {
        super(status);
        this.reporter = reporter;
        this.handler = handler;
        this.predictor = createPredictor(config);
        this.batcher = createBatcher(config, handler, reporter);
        this.batcher.start();
    }

    private WorkerBatcher createBatcher(FlexlbConfig config, BatchDecisionHandler handler,
                                        BatchSchedulerReporter reporter) {
        return new WorkerBatcher(status.getIpPort(), this, config, handler, reporter);
    }

    public WorkerBatcher getBatcher() {
        return batcher;
    }

    @Override
    public void close() {
        try {
            batcher.shutdown();
        } finally {
            super.close();
        }
    }

    public long batcherWaitMs() {
        return batcher.queueWaitMs();
    }

    /**
     * Auto-TPM measured queue-wait estimate (measured queue-age design):
     * the age of the next-to-dispatch head (service-order head, i.e. the
     * oldest item of the highest-priority lane) — a direct congestion
     * measurement of how slowly this engine drains its queue, O(1) and
     * priority-blind. For a probe below the head's priority lane the
     * value is a lower bound on its true wait (the probe queues behind
     * that head); that conservative form is intentional — see
     * {@code PrefillQueueManager.estimateWaitMs} for the full argument.
     */
    public long batcherEstimatedWaitMs(int priority, long deadlineMs, long requestId) {
        return batcher.queueManager().estimateWaitMs(priority, deadlineMs, requestId);
    }

    private static PrefillTimePredictor createPredictor(FlexlbConfig cfg) {
        if ("learning".equalsIgnoreCase(cfg.getPrefillPredictorType())) {
            return new LearningPredictor();
        }
        return new FormulaPredictor(cfg.getCostFormula());
    }

    public void commitBatch(long batchId, long predictMs, List<BatchItem> requests) {
        BatchInflight newBatch = new BatchInflight(predictMs, requests);
        BatchInflight prev = inflightBatches.putIfAbsent(batchId, newBatch);
        if (prev != null) {
            // batchId already exists — subtract the old request count before overwriting,
            // otherwise the old value is silently lost and the counter stays inflated.
            inflightRequestCount.addAndGet(-prev.requests().size());
            inflightBatches.put(batchId, newBatch);
        }
        inflightRequestCount.addAndGet(requests.size());
        cachedWaitTimeExpireAtMs = 0;
    }

    public void releaseBatch(long batchId) {
        AtomicReference<BatchInflight> removedBatch = new AtomicReference<>();
        inflightBatches.compute(batchId, (id, batch) -> {
            // Keep the lock order consistent with begin/calibration:
            // inflight batch key first, reconciliation key second.
            reconciliationRequests.remove(id);
            removedBatch.set(batch);
            return null;
        });
        BatchInflight removed = removedBatch.get();
        if (removed != null) {
            inflightRequestCount.addAndGet(-removed.requests().size());
            cachedWaitTimeExpireAtMs = 0;
        }
    }

    /**
     * Handle partial batch failure: remove failed requests from a batch and recompute prediction.
     *
     */
    public void repackBatch(long batchId, Set<Long> failedRequestIds) {
        long statusMs = System.currentTimeMillis();
        inflightBatches.computeIfPresent(batchId, (id, old) -> {
            List<BatchItem> survivors = old.requests().stream()
                    .filter(r -> !failedRequestIds.contains(r.requestId()))
                    .toList();
            int removed = old.requests().size() - survivors.size();
            if (removed == 0) {
                return old;
            }
            old.touch(statusMs);
            old.observeFailure();
            if (survivors.isEmpty()) {
                inflightRequestCount.addAndGet(-old.requests().size());
                cachedWaitTimeExpireAtMs = 0;
                return null; // removes entry from map
            }
            long newPredMs = (long) predictor.predictBatchMs(survivors);
            BatchInflight repacked = old.repack(newPredMs, survivors);
            inflightRequestCount.addAndGet(-removed);
            cachedWaitTimeExpireAtMs = 0;
            return repacked;
        });
    }

    @Override
    public void onWorkerStatusUpdate(WorkerStatus ws, WorkerStatusResponse resp) {
        super.onWorkerStatusUpdate(ws, resp);
        calibrate(resp.getFinishedTaskInfo(), resp.getRunningTaskInfo());
        updateEngineUntrackedRequestCount(resp);
        reportedWaitingQueryLen = Math.max(0, resp.getWaitingQueryLen());
    }

    /**
     * Engine-reported waiting (queued) query count from the last ~20ms status
     * sync, clamped to {@code >= 0}. Last-known value between syncs (no
     * decay); a lost engine is removed by the registry instead.
     */
    public long getReportedWaitingQueryLen() {
        return reportedWaitingQueryLen;
    }

    /**
     * Full calibration against worker status report.
     */
    private void calibrate(Map<String, TaskInfo> finishedTaskInfo, Map<String, TaskInfo> runningTaskInfo) {
        long statusMs = System.currentTimeMillis();

        int finishedSize = finishedTaskInfo != null ? finishedTaskInfo.size() : 0;
        int runningSize = runningTaskInfo != null ? runningTaskInfo.size() : 0;
        if (finishedSize > 0 || !inflightBatches.isEmpty()) {
            logger.debug("Prefill calibrate: finishedTasks={}, runningTasks={}, inflightBatches={}",
                    finishedSize, runningSize, inflightBatches.size());
        }

        // Phase 1: collect request-level terminal observations and reconcile tasks whose Engine
        // status omitted batch_id. Legacy non-batch requests use requestId as
        // the inflight key; real batch members are resolved by membership.
        Map<Long, List<FinishedObservation>> finishedByBatch = new HashMap<>();

        if (finishedTaskInfo != null) {
            for (TaskInfo task : finishedTaskInfo.values()) {
                FinishedObservation observation = FinishedObservation.from(task);
                long batchId = task.getBatchId();
                if (batchId < 0) {
                    reconcileFinishedWithoutBatchId(observation, statusMs);
                    continue;
                }
                finishedByBatch.computeIfAbsent(batchId, k -> new ArrayList<>()).add(observation);
            }
        }

        // Phase 2: settle only the locally-owned finished members. WorkerStatus
        // is request-granular and the Engine may partially admit a batch, so a
        // short member finishing must not release long-running siblings or
        // reopen the fixed-window inflight gate.
        for (Map.Entry<Long, List<FinishedObservation>> entry : finishedByBatch.entrySet()) {
            settleFinishedMembers(entry.getKey(), entry.getValue(), statusMs);
        }

        // Phase 3: update progress anchors. A queued batch cannot spend
        // predicted forward time until the worker reports it as RUNNING.
        Map<Long, List<TaskInfo>> activeByBatch = new HashMap<>();
        if (runningTaskInfo != null) {
            for (TaskInfo task : runningTaskInfo.values()) {
                long batchId = task.getBatchId();
                if (batchId >= 0) {
                    activeByBatch.computeIfAbsent(batchId, ignored -> new ArrayList<>()).add(task);
                }
            }
        }
        for (Map.Entry<Long, List<TaskInfo>> entry : activeByBatch.entrySet()) {
            inflightBatches.computeIfPresent(entry.getKey(), (id, batch) -> {
                Set<Long> currentRequestIds = batch.requests().stream()
                        .map(BatchItem::requestId)
                        .collect(Collectors.toSet());
                boolean observedCurrentMember = false;
                boolean observedRunningMember = false;
                boolean observedCancelOverlay = false;
                for (TaskInfo task : entry.getValue()) {
                    if (!currentRequestIds.contains(task.getRequestId())) {
                        continue;
                    }
                    observedCurrentMember = true;
                    observedRunningMember |= task.getPhase() == TaskPhase.RUNNING;
                    observedCancelOverlay |= isPriorityCancelOverlayOnly(task);
                }
                if (!observedCurrentMember) {
                    return batch;
                }
                if (observedCancelOverlay) {
                    batch.observeCancelOverlay();
                }
                if (observedRunningMember) {
                    batch.markRunning(statusMs);
                } else {
                    batch.markQueued(statusMs);
                }
                return batch;
            });
        }

        // Phase 4: check running requests for anomalies
        if (runningTaskInfo != null) {
            for (TaskInfo task : runningTaskInfo.values()) {
                long batchId = task.getBatchId();
                if (batchId < 0) {
                    continue;
                }
                if (!inflightBatches.containsKey(batchId)) {
                    logger.debug("Prefill calibrate: running request reqId={} batchId={} not in inflight",
                            task.getRequestId(), batchId);
                }
            }
        }
    }

    private void settleFinishedMembers(long batchId,
                                       List<FinishedObservation> observations,
                                       long statusMs) {
        AtomicReference<BatchInflight> completed = new AtomicReference<>();
        inflightBatches.computeIfPresent(batchId, (id, batch) ->
                applyFinishedObservations(id, batch, observations, statusMs, true, completed));

        BatchInflight completedBatch = completed.get();
        if (completedBatch != null) {
            reportBatchCompletion(batchId, completedBatch);
        }
    }

    private BatchInflight applyFinishedObservations(long batchId,
                                                     BatchInflight batch,
                                                     List<FinishedObservation> observations,
                                                     long statusMs,
                                                     boolean deferReconciliation,
                                                     AtomicReference<BatchInflight> completed) {
        Set<Long> localRequestIds = batch.requests().stream()
                .map(BatchItem::requestId)
                .collect(Collectors.toSet());
        Set<Long> finishedIds = new HashSet<>();
        int foreignCount = 0;
        int deferredByFence = 0;

        for (FinishedObservation observation : observations) {
            long requestId = observation.requestId();
            if (!localRequestIds.contains(requestId)) {
                // Finished snapshots can repeat a member already settled
                // in a previous calibration pass. Warn only for a request
                // that never belonged to this batch generation.
                if (!batch.originalRequestIds().contains(requestId)) {
                    foreignCount++;
                }
                continue;
            }

            batch.touch(statusMs);
            batch.observeExecutionTime(observation.executionTimeMs());
            if (observation.errorCode() == 0) {
                batch.observeSuccessfulCompletion();
            } else {
                batch.observeFailure();
            }
            if (deferReconciliation && deferIfReconciling(batchId, observation)) {
                deferredByFence++;
                continue;
            }
            finishedIds.add(requestId);
            if (observation.errorCode() != 0) {
                logger.debug("Prefill calibrate: batch failure batchId={} reqId={} error={}",
                        batchId, requestId, observation.errorMessage());
            }
        }

        if (deferredByFence > 0) {
            warnSettleDeferredByFence(batchId, batch, deferredByFence,
                    observations.size(), statusMs);
        }
        if (foreignCount > 0) {
            logger.warn("Prefill calibrate: batchId={} has {} finished tasks with foreign requestIds; "
                            + "ignoring them",
                    batchId, foreignCount);
        }
        if (finishedIds.isEmpty()) {
            return batch;
        }

        List<BatchItem> survivors = batch.requests().stream()
                .filter(item -> !finishedIds.contains(item.requestId()))
                .toList();
        inflightRequestCount.addAndGet(-(batch.requests().size() - survivors.size()));
        cachedWaitTimeExpireAtMs = 0;
        if (survivors.isEmpty()) {
            completed.set(batch);
            return null;
        }

        long newPredMs = (long) predictor.predictBatchMs(survivors);
        return batch.repack(newPredMs, survivors);
    }

    /** Anti-flood rate limit for the settle-deferred fence WARN (per batch). */
    private static final long SETTLE_DEFER_WARN_RATE_MS = 60_000L;

    /**
     * Minimum batch age before the all-terminal release leg may fire.
     * Batch-mode futures complete at EnqueueBatch ACK (~100ms after batch
     * creation), while the engine's finish report arrives ~1-2s later via
     * the normal settlement path. Without this floor, the 60s eviction sweep
     * preempts the normal settlement for every freshly-ACKed batch whose
     * sweep tick lands inside the ACK→finish-report window, turning the
     * safety net into the primary gate-release path and producing the
     * synchronized per-minute dispatch pulse. 30s comfortably exceeds the
     * ACK→finish window of healthy batches while still catching
     * zombie-touched entries within one sweep cycle.
     */
    private static final long ALL_TERMINAL_MIN_AGE_MS = 30_000L;

    /**
     * Fence-deferred settle audit: finished members whose settle was skipped
     * because the dispatch-reconciliation fence still owns them (the evidence
     * is cached in the fence state and replays when the fence closes). One
     * WARN per batch per {@link #SETTLE_DEFER_WARN_RATE_MS} — an unresolved
     * fence otherwise repeats on every ~20ms calibrate round because finished
     * snapshots re-report the same members.
     */
    private void warnSettleDeferredByFence(long batchId, BatchInflight batch,
                                           int deferredMembers, int reportedMembers,
                                           long statusMs) {
        if (!batch.shouldWarnSettleDeferred(statusMs, SETTLE_DEFER_WARN_RATE_MS)) {
            return;
        }
        ConcurrentHashMap<Long, ReconciliationState> fence =
                reconciliationRequests.get(batchId);
        org.flexlb.util.Logger.warn(
                "event=inflight_settle_deferred_by_fence endpoint={} batch_id={} "
                        + "reason=dispatch_reconciliation_fence deferred_members={} "
                        + "reported_finished_members={} fence_size={} age_ms={} "
                        + "last_observed_ago_ms={} n_requests={}",
                getIp(), batchId, deferredMembers, reportedMembers,
                fence != null ? fence.size() : 0,
                statusMs - batch.createdAtMs(),
                statusMs - batch.lastObservedAtMs(),
                batch.requests().size());
    }

    /**
     * Reconcile a finished task whose Engine status omitted the original batch id.
     *
     * <p>Legacy non-batch reservations are keyed by request id and carry an empty
     * member list. A real batch is keyed by its generated batch id and always
     * carries its request members. Checking the value shape before the direct
     * removal prevents an unrelated real batch from being erased when its batch
     * id happens to equal this request id.
     *
     * <p>Production priority-cancel terminals may currently report
     * {@code batch_id=-1} even though the Master committed the request as a member
     * of a real batch. In that case scan the live ledger for the unique owning
     * batch and remove only the matching member. The member is revalidated inside
     * the map compute, so a concurrent release/repack/TTL eviction is an idempotent
     * no-op rather than a counter double-decrement. No reverse index is retained,
     * keeping every existing ledger mutation path consistent automatically.
     */
    private void reconcileFinishedWithoutBatchId(FinishedObservation observation, long statusMs) {
        long requestId = observation.requestId();
        AtomicBoolean removedNonBatch = new AtomicBoolean(false);
        inflightBatches.computeIfPresent(requestId, (id, batch) -> {
            if (!batch.requests().isEmpty()) {
                return batch;
            }
            removedNonBatch.set(true);
            reconciliationRequests.remove(id);
            inflightRequestCount.addAndGet(-batch.requests().size());
            cachedWaitTimeExpireAtMs = 0;
            return null;
        });
        if (removedNonBatch.get()) {
            return;
        }

        List<Long> matchingBatchIds = new ArrayList<>();
        for (Map.Entry<Long, BatchInflight> entry : inflightBatches.entrySet()) {
            boolean containsRequest = entry.getValue().requests().stream()
                    .anyMatch(item -> item.requestId() == requestId);
            if (containsRequest) {
                matchingBatchIds.add(entry.getKey());
            }
        }
        if (matchingBatchIds.isEmpty()) {
            logger.debug("Prefill calibrate: finished task with no batch id reqId={} not in inflight",
                    requestId);
            return;
        }
        if (matchingBatchIds.size() != 1) {
            // A request is expected to belong to exactly one live batch. Do not
            // guess when that invariant is already broken: member-scoped cleanup
            // in multiple generations could erase a newer dispatch.
            logger.warn("Prefill calibrate: finished task with no batch id reqId={} matches batches={}; "
                            + "skipping ambiguous cleanup",
                    requestId, matchingBatchIds);
            return;
        }

        long resolvedBatchId = matchingBatchIds.get(0);
        settleFinishedMembers(resolvedBatchId, List.of(observation), statusMs);
    }

    private void updateEngineUntrackedRequestCount(WorkerStatusResponse response) {
        Set<Long> localRequestIds = new HashSet<>();
        for (BatchInflight batch : inflightBatches.values()) {
            for (BatchItem request : batch.requests()) {
                localRequestIds.add(request.requestId());
            }
        }

        Set<Long> untracked = new HashSet<>();
        Map<String, TaskInfo> runningTasks = response.getRunningTaskInfo();
        if (runningTasks != null) {
            for (TaskInfo task : runningTasks.values()) {
                if (task == null || isPriorityCancelOverlayOnly(task)) {
                    continue;
                }
                if (!localRequestIds.contains(task.getRequestId())) {
                    untracked.add(task.getRequestId());
                }
            }
        }

        long reportedActive = Math.max(0, response.getWaitingQueryLen())
                + Math.max(0, response.getRunningQueryLen());
        long scalarLowerBound = Math.max(0, reportedActive - localRequestIds.size());
        // The protobuf converter represents an absent detail list as an empty map,
        // while older/newer Engine variants may still populate only the scalar
        // counts. Keep the request-id union when details exist and conservatively
        // retain the scalar lower bound when the detail list is empty or partial.
        engineUntrackedRequestCount.set(Math.max(untracked.size(), scalarLowerBound));
    }

    private static boolean isPriorityCancelOverlayOnly(TaskInfo task) {
        PriorityPreemptionProgress progress = task.getPriorityPreemptionProgress();
        return (progress == PriorityPreemptionProgress.CANCELING
                || progress == PriorityPreemptionProgress.CANCELED)
                && task.getPhase() == TaskPhase.PENDING;
    }

    // ==================== Pending Count ====================

    /**
     * Real pending count: total requests the engine will face.
     * Includes master-tracked inflight + batcher queue + active Engine tasks
     * not already represented in the local ledger.
     */
    public long realPendingCount() {
        return inflightRequestCount.get() + batcher.queueSize() + engineUntrackedRequestCount.get();
    }

    // ==================== Wait Time ====================

    /**
     * Real wait time: estimated time to drain current inflight batches.
     */
    public long realWaitTimeMs() {
        long waitMs = estimateWaitingTimeMs(System.currentTimeMillis());
        return waitMs;
    }

    public int getInflightBatchCount() {
        return inflightBatches.size();
    }

    /**
     * Whether any live inflight batch still carries {@code requestId} as a
     * member — the prefill-side visibility check for the scheduler's
     * post-ACK inflight audit. An entry invisible here and on the decode
     * confirmed registry can no longer be settled through any ordinary path,
     * so the audit may force-settle it.
     */
    public boolean tracksRequest(long requestId) {
        for (BatchInflight batch : inflightBatches.values()) {
            for (BatchItem request : batch.requests()) {
                if (request.requestId() == requestId) {
                    return true;
                }
            }
        }
        return false;
    }

    /**
     * Evict inflight batches not observed for longer than {@code ttlMs}.
     * Called periodically by the scheduler to clean up stale prefill entries.
     *
     * @return number of batches evicted
     */
    public int evictExpiredBatches(long ttlMs) {
        return evictExpiredBatches(ttlMs, 0, 0, 0, requestId -> false);
    }

    /**
     * Evict inflight batches not observed for longer than {@code ttlMs}, and
     * force-evict batches whose creation age exceeds {@code hardMaxAgeMs}
     * regardless of the reconciliation fence or observation freshness.
     *
     * <p>The hard cap defends the ledger against engine-side zombie tasks: a
     * stuck task that the engine keeps re-reporting refreshes
     * {@code lastObservedAtMs} on every calibrate round, so the TTL criterion
     * alone would retain the entry forever. Entries whose age exceeds any
     * legitimate request lifecycle are evicted even when kept "fresh".
     *
     * <p>Safety: a batch is only force-evicted when none of its requests is
     * still tracked by the scheduler ({@code schedulerOwnsRequest}), so an
     * active cancel-fence settlement is never raced. The check runs inside the
     * per-key compute so it is atomic against concurrent ledger mutation.
     *
     * @param ttlMs             max unobserved age before normal eviction
     * @param hardMaxAgeMs      hard creation-age cap; {@code <= 0} disables
     * @param schedulerOwnsRequest whether the scheduler still tracks a request
     * @return number of batches evicted (normal + forced)
     */
    public int evictExpiredBatches(long ttlMs, long hardMaxAgeMs,
                                   LongPredicate schedulerOwnsRequest) {
        return evictExpiredBatches(ttlMs, hardMaxAgeMs, 0, 0, schedulerOwnsRequest);
    }

    /**
     * Full eviction pass with the progress-aware batch-level inflight age
     * cap ({@code flexlbBatchInflightMaxAgeMs} +
     * {@code flexlbBatchInflightStaleMs}) layered on top: a committed
     * inflight batch whose creation age exceeds {@code batchInflightMaxAgeMs}
     * <b>and</b> whose last observation is older than
     * {@code batchInflightStaleMs} is force-settled — even while a
     * dispatch-reconciliation fence holds it and even when the scheduler
     * still tracks its members. That is the bounded-freeze guarantee: a
     * zombie reconciliation that never receives its authoritative
     * settlement must not pin the endpoint ledger (and the fixed-window
     * inflight gate) forever. The staleness leg is the progress-aware
     * guard: batches the ~20ms worker status sync keeps observing (running
     * members, saturated queued batches, long-generation pdFusion batches)
     * refresh {@code lastObservedAtMs} on every calibrate round and are
     * never capped — only batches that went silent for
     * {@code batchInflightStaleMs} are treated as frozen.
     * {@code batchInflightStaleMs <= 0} drops the progress guard (pure age
     * cap). On release the batch entry, its reconciliation fence and the
     * request counter are dropped, the members are re-reserved as
     * engine-untracked (the engine may still be executing
     * them, and the next status sync recomputes the exact count), each
     * member is routed through the existing handler terminal chain
     * ({@link BatchDecisionHandler#onExpired} — idempotent against entries
     * the scheduler already settled), and one WARN + age-cap metric is
     * emitted per batch tagged with the endpoint's own role. The cap is
     * checked first, so it also covers the scheduler-owned entries the
     * guarded {@code hardMaxAgeMs} branch skips; those get their
     * scheduler-side release from the same handler chain. Auto-TPM only —
     * the registry gates the pass with {@code isAutoTpmEnabled()}.
     *
     * <p>All-terminal release (18:10 fix): a batch whose members are ALL
     * terminal — scheduler-side futures done (admission timeout, client
     * cancel, scheduler-side expiry: terminals that never surface as
     * engine finished reports) — is removed regardless of observation
     * freshness. Zombie engine running entries keep touching such a batch
     * ({@code markQueued}), defeating every staleness-based release leg
     * (TTL / age cap / lost-detection) indefinitely; terminal members
     * cannot revive, so the sweep releases the ledger entry and the
     * inflight gate directly. Fenced batches are excluded (the fence owns
     * their settle: finished evidence is deferred into the fence state and
     * replays when the fence closes) and no member callbacks are needed
     * (the futures are already terminal). This leg is unconditional — it
     * is a correctness fix, not a tuning knob.
     *
     * @param ttlMs                 max unobserved age before normal eviction
     * @param hardMaxAgeMs          guarded hard creation-age cap;
     *                              {@code <= 0} disables
     * @param batchInflightMaxAgeMs batch-level age cap;
     *                              {@code <= 0} disables
     * @param batchInflightStaleMs  no-progress staleness threshold for the
     *                              age cap; {@code <= 0} drops the
     *                              progress guard (pure age cap)
     * @param schedulerOwnsRequest  whether the scheduler still tracks a
     *                              request (race guard for the guarded
     *                              hard-cap branch only)
     * @return number of batches removed (all-terminal + age-capped +
     *         guarded hard-cap + normal TTL)
     */
    public int evictExpiredBatches(long ttlMs, long hardMaxAgeMs, long batchInflightMaxAgeMs,
                                   long batchInflightStaleMs, LongPredicate schedulerOwnsRequest) {
        return evictExpiredBatchesByReason(ttlMs, hardMaxAgeMs, batchInflightMaxAgeMs,
                batchInflightStaleMs, schedulerOwnsRequest).total();
    }

    /**
     * Same eviction pass as
     * {@link #evictExpiredBatches(long, long, long, long, LongPredicate)} but
     * returns the per-exit counts, so the eviction metric can carry one
     * {@code reason} tag per exit ({@code all_terminal} / {@code age_capped}
     * / {@code hard_age_cap} / {@code ttl}) instead of folding all exits into
     * one number. The eviction logic, exit ordering and thresholds are
     * identical.
     *
     * @return per-exit eviction counts
     */
    public EvictionBreakdown evictExpiredBatchesByReason(long ttlMs, long hardMaxAgeMs,
                                                         long batchInflightMaxAgeMs,
                                                         long batchInflightStaleMs,
                                                         LongPredicate schedulerOwnsRequest) {
        long nowMs = System.currentTimeMillis();
        AtomicInteger allTerminalCount = new AtomicInteger();
        AtomicInteger ageCappedCount = new AtomicInteger();
        AtomicInteger hardCappedCount = new AtomicInteger();
        AtomicInteger ttlCount = new AtomicInteger();
        for (Long batchId : inflightBatches.keySet()) {
            AtomicReference<BatchInflight> evicted = new AtomicReference<>();
            AtomicReference<BatchInflight> forced = new AtomicReference<>();
            AtomicReference<BatchInflight> ageCapped = new AtomicReference<>();
            AtomicReference<BatchInflight> allTerminal = new AtomicReference<>();
            inflightBatches.computeIfPresent(batchId, (id, batch) -> {
                long ageMs = nowMs - batch.createdAtMs();
                // 18:10 fix: members whose scheduler-side futures are all done
                // were terminated through paths that never surface as engine
                // finished reports (admission timeout, client cancel,
                // scheduler-side expiry). Zombie engine running entries keep
                // touching the batch (markQueued), so every staleness-based
                // release leg (TTL / age cap / lost-detection) is defeated
                // indefinitely. Terminal members cannot revive — release the
                // gate now, regardless of observation freshness. Fenced
                // batches stay with the fence's own settle paths.
                // Min-age gate: batch-mode futures complete at EnqueueBatch
                // ACK (~100ms after creation), long before the engine finishes
                // computing (~1-2s) and the normal settlement path releases
                // the batch via engine finished reports. Without an age floor
                // here, every freshly-ACKed batch satisfies all-terminal and
                // the 60s sweep preempts the normal settlement, causing the
                // synchronized per-minute dispatch pulse. 30s comfortably
                // exceeds the longest expected ACK→finish-report window for
                // healthy batches while still catching zombie-touched entries
                // within one sweep cycle.
                if (!hasDispatchReconciliation(id)
                        && ageMs > ALL_TERMINAL_MIN_AGE_MS
                        && !batch.requests().isEmpty()
                        && batch.requests().stream().allMatch(
                                item -> item.future() != null && item.future().isDone())) {
                    allTerminal.set(batch);
                    return null;
                }
                if (batchInflightMaxAgeMs > 0 && ageMs > batchInflightMaxAgeMs
                        && (batchInflightStaleMs <= 0
                            || nowMs - batch.lastObservedAtMs() > batchInflightStaleMs)) {
                    ageCapped.set(batch);
                    return null;
                }
                if (hardMaxAgeMs > 0 && ageMs > hardMaxAgeMs
                        && batch.requests().stream().noneMatch(
                                item -> schedulerOwnsRequest.test(item.requestId()))) {
                    forced.set(batch);
                    return null;
                }
                if (hasDispatchReconciliation(id)
                        || nowMs - batch.lastObservedAtMs() <= ttlMs) {
                    return batch;
                }
                evicted.set(batch);
                return null;
            });
            BatchInflight terminalBatch = allTerminal.get();
            if (terminalBatch != null) {
                // No fence exists by precondition, so there is nothing to
                // clear; the futures are already terminal, so no handler
                // terminal chain is needed either — just release the ledger
                // entry and the inflight gate. Zombie engine entries keep
                // feeding updateEngineUntrackedRequestCount until the engine
                // drops them, which only biases that penalty conservatively.
                boolean schedulerOwned = terminalBatch.requests().stream()
                        .anyMatch(item -> schedulerOwnsRequest.test(item.requestId()));
                RoleType terminalRole = status.getRole();
                org.flexlb.util.Logger.warn(
                        "event=inflight_settle_forced_by_all_members_terminal role={} "
                                + "endpoint={} batch_id={} age_ms={} member_count={} "
                                + "last_observed_ago_ms={} touch_source={} "
                                + "scheduler_owned={} observation_misses={} "
                                + "reason=all_members_terminal_zombie_touch",
                        terminalRole != null ? terminalRole.name() : RoleType.PREFILL.name(),
                        getIp(), batchId,
                        nowMs - terminalBatch.createdAtMs(),
                        terminalBatch.requests().size(),
                        nowMs - terminalBatch.lastObservedAtMs(),
                        terminalBatch.running() ? "engine_running_report"
                                : "engine_queued_report_or_init",
                        schedulerOwned, terminalBatch.observationMisses());
                inflightRequestCount.addAndGet(-terminalBatch.requests().size());
                cachedWaitTimeExpireAtMs = 0;
                allTerminalCount.incrementAndGet();
                continue;
            }
            BatchInflight cappedBatch = ageCapped.get();
            if (cappedBatch != null) {
                forceSettleAgeCappedBatch(batchId, cappedBatch,
                        nowMs - cappedBatch.createdAtMs(),
                        nowMs - cappedBatch.lastObservedAtMs(), batchInflightMaxAgeMs);
                ageCappedCount.incrementAndGet();
                continue;
            }
            BatchInflight forcedBatch = forced.get();
            if (forcedBatch != null) {
                // The engine never settled these members, so the fence's
                // authoritative settlement will never arrive — drop it too.
                boolean hadFence = reconciliationRequests.remove(batchId) != null;
                org.flexlb.util.Logger.warn(
                        "event=inflight_hard_age_eviction role=PREFILL endpoint={} "
                                + "batch_id={} age_ms={} hard_max_age_ms={} created_at_ms={} "
                                + "last_observed_ago_ms={} running={} cancel_overlay_observed={} "
                                + "fence={} n_requests={} request_ids={}",
                        getIp(), batchId, nowMs - forcedBatch.createdAtMs(), hardMaxAgeMs,
                        forcedBatch.createdAtMs(), nowMs - forcedBatch.lastObservedAtMs(),
                        forcedBatch.running(), forcedBatch.cancelOverlayObserved(),
                        hadFence, forcedBatch.requests().size(),
                        forcedBatch.originalRequestIds());
            }
            BatchInflight removed = forcedBatch != null ? forcedBatch : evicted.get();
            if (removed != null) {
                inflightRequestCount.addAndGet(-removed.requests().size());
                cachedWaitTimeExpireAtMs = 0;
                if (forcedBatch != null) {
                    hardCappedCount.incrementAndGet();
                } else {
                    ttlCount.incrementAndGet();
                }
            }
        }
        return new EvictionBreakdown(allTerminalCount.get(), ageCappedCount.get(),
                hardCappedCount.get(), ttlCount.get());
    }

    /**
     * Bounded-freeze release for one age-capped inflight batch, called
     * after the per-key compute already removed it from {@code inflightBatches}
     * (so the visible {@code inflight.batch.count} drops immediately).
     * Drops the reconciliation fence (the authoritative settlement this
     * fence is waiting for will never arrive — that is why the batch
     * reached the cap), decrements the request counter, compensates the
     * engine-untracked counter (the engine may still be executing the
     * members — re-reserve them so the fixed-window inflight gate does not
     * oversell in the short window before the next status sync recomputes
     * the exact count), emits one WARN line per batch, and routes every
     * member through the
     * existing handler terminal chain ({@code onExpired} — the scheduler
     * completes/times out its own inflight entry and any later engine
     * terminal for these requests lands on already-removed keys, i.e.
     * idempotent no-ops). Member callbacks run outside any ledger critical
     * section.
     */
    private void forceSettleAgeCappedBatch(long batchId, BatchInflight batch,
                                           long ageMs, long lastObservedAgoMs, long maxAgeMs) {
        boolean hadFence = reconciliationRequests.remove(batchId) != null;
        int members = batch.requests().size();
        inflightRequestCount.addAndGet(-members);
        // Re-reserve the members as engine-untracked so the
        // released inflight gate does not oversell before the next worker
        // status sync recomputes the count.
        engineUntrackedRequestCount.addAndGet(members);
        cachedWaitTimeExpireAtMs = 0;
        // Tag with the endpoint's own role, not a hard-coded PREFILL.
        RoleType role = status.getRole();
        String roleName = role != null ? role.name() : RoleType.PREFILL.name();
        org.flexlb.util.Logger.warn(
                "event=inflight_batch_age_capped role={} endpoint={} batch_id={} "
                        + "age_ms={} last_observed_ago_ms={} max_age_ms={} fenced={} "
                        + "n_requests={} request_ids={}",
                roleName, getIp(), batchId, ageMs, lastObservedAgoMs, maxAgeMs, hadFence,
                members, batch.originalRequestIds());
        if (handler != null) {
            for (BatchItem request : batch.requests()) {
                handler.onExpired(request);
            }
        }
    }

    /** Protect an ACK-ambiguous batch from age-only eviction. */
    public void beginDispatchReconciliation(long batchId, long requestId) {
        long nowMs = System.currentTimeMillis();
        inflightBatches.computeIfPresent(batchId, (id, batch) -> {
            boolean owned = batch.requests().stream()
                    .anyMatch(item -> item.requestId() == requestId);
            if (!owned) {
                return batch;
            }
            reconciliationRequests.compute(id, (ignored, requests) -> {
                ConcurrentHashMap<Long, ReconciliationState> states = requests != null
                        ? requests : new ConcurrentHashMap<>();
                states.putIfAbsent(requestId, new ReconciliationState(null));
                return states;
            });
            batch.touch(nowMs);
            return batch;
        });
    }

    /** Release one request's reconciliation fence after authoritative settlement. */
    public void endDispatchReconciliation(long batchId, long requestId) {
        long statusMs = System.currentTimeMillis();
        AtomicReference<BatchInflight> completed = new AtomicReference<>();
        inflightBatches.compute(batchId, (id, batch) -> {
            AtomicReference<FinishedObservation> deferredTerminal = new AtomicReference<>();
            reconciliationRequests.computeIfPresent(id, (ignored, requests) -> {
                ReconciliationState state = requests.remove(requestId);
                if (state != null) {
                    deferredTerminal.set(state.deferredTerminal());
                }
                return requests.isEmpty() ? null : requests;
            });
            if (batch == null) {
                return null;
            }
            FinishedObservation observation = deferredTerminal.get();
            if (observation == null) {
                batch.touch(statusMs);
                return batch;
            }
            // With the engine reporting a Prefill member finished as
            // soon as its stream terminates (no fetch wait), the fence's own
            // ack-only release path closes this fence on the same response
            // that carried the finished evidence, so the cached
            // deferredTerminal replays here without re-deferring — the fence
            // is already gone by construction.
            return applyFinishedObservations(
                    id, batch, List.of(observation), statusMs, false, completed);
        });
        BatchInflight completedBatch = completed.get();
        if (completedBatch != null) {
            reportBatchCompletion(batchId, completedBatch);
        }
    }

    /**
     * Defer a fenced member's finished observation: the dispatch-reconciliation
     * fence owns its settle decision, so the evidence is cached in the fence
     * state (merged with any earlier one) instead of settling immediately.
     * {@link #endDispatchReconciliation} replays the cached terminal once the
     * fence closes — under event-driven finish promotion that is the
     * ack-only release triggered by the same finished report, so the defer
     * window is one fence lifetime, not an
     * unbounded wait on the lost Enqueue ACK.
     */
    private boolean deferIfReconciling(long batchId, FinishedObservation observation) {
        AtomicBoolean deferred = new AtomicBoolean(false);
        reconciliationRequests.computeIfPresent(batchId, (ignored, requests) -> {
            requests.computeIfPresent(observation.requestId(), (requestId, state) -> {
                deferred.set(true);
                FinishedObservation existing = state.deferredTerminal();
                return new ReconciliationState(existing == null
                        ? observation : existing.merge(observation));
            });
            return requests.isEmpty() ? null : requests;
        });
        return deferred.get();
    }

    private boolean hasDispatchReconciliation(long batchId) {
        ConcurrentHashMap<Long, ReconciliationState> requests = reconciliationRequests.get(batchId);
        return requests != null && !requests.isEmpty();
    }

    @Override
    public long getLoadMetric() {
        return realWaitTimeMs();
    }

    public PrefillTimePredictor getPredictor() {
        return predictor;
    }

    // ==================== Metrics ====================

    /**
     * Report per-worker batch metrics via the given reporter.
     * Called periodically by {@link org.flexlb.balance.scheduler.FlexlbBatchScheduler}.
     */
    public void reportBatchMetrics(BatchSchedulerReporter reporter) {
        int queueSize = batcher.queueSize();
        reporter.reportBatcherQueueSize(RoleType.PREFILL.name(), getIp(), queueSize);
        // Priority-bucketed batch queue length — single-report with priority tag.
        // Empty queue fallback: report priority=0 depth=0 so tagged panels don't gap.
        Map<Integer, Integer> sizeByPriority = batcher.queueSizeByPriority();
        if (sizeByPriority.isEmpty()) {
            reporter.reportBatcherQueueDepthByPriority(RoleType.PREFILL.name(), getIp(), 0, 0);
        } else {
            sizeByPriority.forEach((priority, size) ->
                    reporter.reportBatcherQueueDepthByPriority(RoleType.PREFILL.name(), getIp(), priority, size));
        }
        reporter.reportInflightBatchCount(RoleType.PREFILL.name(), getIp(), getInflightBatchCount());
        reporter.reportInflightRequestCount(RoleType.PREFILL.name(), getIp(), inflightRequestCount.get());
        reporter.reportInflightMaxAgeMs(RoleType.PREFILL.name(), getIp(),
                InflightEvictor.maxAgeMs(inflightBatches, System.currentTimeMillis()));
    }

    /**
     * On batch completion, compare the formula-predicted execution time against the
     * engine-reported actual execution time (max across the batch's finished tasks),
     * then log and emit prediction-accuracy metrics.
     */
    private void reportBatchCompletion(long batchId, BatchInflight batch) {
        long actualMs = batch.maxExecutionTimeMs();
        if (!batch.successfulCompletionObserved() || actualMs <= 0) {
            logger.debug("batch completion not reportable: batchId={} success={} actualMs={}",
                    batchId, batch.successfulCompletionObserved(), actualMs);
            return;
        }

        long predictedMs = batch.originalPredictTimeMs();
        long gapMs = actualMs - predictedMs;
        org.flexlb.util.Logger.debug(
                "flexlb_batch_complete batch_id={} predicted_ms={} actual_ms={} gap_ms={} batch_size={} engine={}",
                batchId, predictedMs, actualMs, gapMs, batch.originalFeatures().batchSize(), getIp());

        // A failed/removed member makes the original batch an invalid learning
        // sample even if another member completed successfully.
        if (batch.learningEligible()) {
            predictor.learn(batch.originalFeatures(), predictedMs, actualMs);
        }

        reporter.reportBatchPredictedTimeMs(RoleType.PREFILL.name(), getIp(), predictedMs);
        reporter.reportBatchActualTimeMs(RoleType.PREFILL.name(), getIp(), actualMs);
        reporter.reportBatchPredictGapMs(RoleType.PREFILL.name(), getIp(), gapMs);
    }

    private long estimateWaitingTimeMs(long nowMs) {
        if (nowMs < cachedWaitTimeExpireAtMs) {
            return cachedWaitTimeMs;
        }
        if (inflightBatches.isEmpty()) {
            cachedWaitTimeMs = 0;
            cachedWaitTimeExpireAtMs = nowMs + WAIT_TIME_CACHE_TTL_MS;
            return 0;
        }
        long totalPredMs = 0;
        long earliestProgressBaseMs = Long.MAX_VALUE;
        for (BatchInflight batch : inflightBatches.values()) {
            totalPredMs += Math.max(0, batch.predictTimeMs());
            earliestProgressBaseMs = Math.min(earliestProgressBaseMs, batch.progressBaseMs());
        }
        long result;
        if (earliestProgressBaseMs == Long.MAX_VALUE) {
            result = 0;
        } else {
            long elapsedMs = Math.max(0, nowMs - earliestProgressBaseMs);
            result = Math.max(0, totalPredMs - elapsedMs);
        }
        cachedWaitTimeMs = result;
        cachedWaitTimeExpireAtMs = nowMs + WAIT_TIME_CACHE_TTL_MS;
        return result;
    }

}
