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
     * Retained so the F-F batch-level age cap can route each member of a
     * force-settled batch through the existing handler terminal chain
     * ({@link BatchDecisionHandler#onExpired}) instead of leaving the
     * scheduler-side inflight entries dangling after the endpoint ledger
     * entry is removed.
     */
    private final BatchDecisionHandler handler;
    /**
     * Runtime config, retained for the calibrate-driven ACKNOWLEDGED-lost
     * detection (Fix A: {@code flexlbPrefillLostAfterMs} /
     * {@code flexlbPrefillLostMinMisses}) — the constructor is the only
     * config injection point of this class.
     */
    private final FlexlbConfig config;

    /**
     * Active Engine tasks not already represented in the local batch ledger.
     * Atomic so the F-F age-cap release (R5 compensation) can add the
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
        this.config = config;
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
     * Auto-TPM priority-aware queue wait estimate (design doc 8.4):
     * counts only items ordered ahead of the incoming request.
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

        // Phase 5 (Fix A): ACKNOWLEDGED-lost detection. This calibrate round
        // is backed by a real, version-advanced engine report (the status
        // runner gates onWorkerStatusUpdate on versionAdvanced && isAlive),
        // so a committed batch that the report mentions nowhere was silently
        // dropped by the engine — count the miss and, past the configured
        // thresholds, force-settle it through the handler terminal chain.
        detectLostBatches(finishedByBatch.keySet(), activeByBatch.keySet(), statusMs);
    }

    /**
     * Fix A (ACKNOWLEDGED-lost detection, 205 pileup incident): sweep the
     * inflight ledger against the batch ids this calibrate round observed.
     * A batch mentioned anywhere (finished or running, any member — even a
     * stale one) resets its miss counter. A batch mentioned nowhere records
     * one miss; once it accumulates {@code flexlbPrefillLostMinMisses}
     * consecutive misses <b>and</b> its {@code lastObservedAtMs} is older
     * than {@code flexlbPrefillLostAfterMs}, it is removed inside the
     * per-key compute (atomic against concurrent settle/repack) and
     * force-settled.
     *
     * <p>False-kill guards, in order:
     * <ul>
     * <li>misses only advance here — i.e. on rounds backed by a real
     *     version-advanced engine report; sync stalls, pull failures or a
     *     frozen report version never accumulate misses (detection then
     *     degrades to the 120s age cap / 300s TTL, never fires early);</li>
     * <li>empty-member entries (legacy non-batch placeholders keyed by
     *     requestId — the engine reports those with {@code batch_id=-1},
     *     so they would always look unobserved) are skipped;</li>
     * <li>fenced batches ({@code hasDispatchReconciliation}) are skipped —
     *     an ACK-ambiguous dispatch awaits its authoritative settlement and
     *     has its own zombie guards (reconcile-target-missing, age cap);</li>
     * <li>the time leg anchors on {@code lastObservedAtMs} (init =
     *     createdAtMs), so a just-committed batch cannot be settled before
     *     {@code lostAfterMs} even if the engine report races the commit
     *     (miss counting may start on the same tick as commitBatch, but
     *     the 20s time leg dominates the 3-round miss leg by two orders
     *     of magnitude at the ~20ms-2s sync cadence).</li>
     * </ul>
     * Auto-TPM only; {@code flexlbPrefillLostAfterMs <= 0} disables.
     */
    private void detectLostBatches(Set<Long> finishedBatchIds, Set<Long> activeBatchIds,
                                   long statusMs) {
        if (config == null || !config.isAutoTpmEnabled()) {
            return;
        }
        long lostAfterMs = config.getFlexlbPrefillLostAfterMs();
        if (lostAfterMs <= 0) {
            return;
        }
        int minMisses = Math.max(1, config.getFlexlbPrefillLostMinMisses());
        for (Long batchId : inflightBatches.keySet()) {
            AtomicReference<BatchInflight> lost = new AtomicReference<>();
            inflightBatches.computeIfPresent(batchId, (id, batch) -> {
                if (finishedBatchIds.contains(id) || activeBatchIds.contains(id)) {
                    batch.resetObservationMisses();
                    return batch;
                }
                if (batch.requests().isEmpty() || hasDispatchReconciliation(id)) {
                    return batch;
                }
                int misses = batch.recordObservationMiss();
                if (misses < minMisses
                        || statusMs - batch.lastObservedAtMs() <= lostAfterMs) {
                    return batch;
                }
                lost.set(batch);
                return null;
            });
            BatchInflight lostBatch = lost.get();
            if (lostBatch != null) {
                forceSettleLostBatch(batchId, lostBatch, statusMs, lostAfterMs);
            }
        }
    }

    /**
     * Fix A release for one lost (engine-dropped) inflight batch, called
     * after the per-key compute already removed it from
     * {@code inflightBatches}. Mirrors the F-F age-cap release shape but
     * with the opposite engine-side premise: the engine just told us — via
     * a successful report that mentions the batch nowhere — that it is NOT
     * executing these members, so unlike {@code forceSettleAgeCappedBatch}
     * there is no R5 engine-untracked compensation, and no fence to drop
     * (fenced batches are never selected). Each member goes through the
     * existing handler terminal chain ({@code onExpired} — idempotent:
     * entries the scheduler already settled reduce to no-ops, a preempted
     * entry defers to its preemption fence). Member callbacks run outside
     * any ledger critical section.
     */
    private void forceSettleLostBatch(long batchId, BatchInflight batch,
                                      long statusMs, long lostAfterMs) {
        int members = batch.requests().size();
        inflightRequestCount.addAndGet(-members);
        cachedWaitTimeExpireAtMs = 0;
        RoleType role = status.getRole();
        String roleName = role != null ? role.name() : RoleType.PREFILL.name();
        org.flexlb.util.Logger.warn(
                "event=inflight_batch_lost_settled role={} endpoint={} batch_id={} "
                        + "last_observed_ago_ms={} lost_after_ms={} misses={} age_ms={} "
                        + "running={} n_requests={} request_ids={}",
                roleName, getIp(), batchId, statusMs - batch.lastObservedAtMs(),
                lostAfterMs, batch.observationMisses(), statusMs - batch.createdAtMs(),
                batch.running(), members, batch.originalRequestIds());
        if (reporter != null) {
            reporter.reportEndpointInflightTtlExpired(roleName, getIp(), "post_ack_lost", 1);
        }
        if (handler != null) {
            for (BatchItem request : batch.requests()) {
                handler.onExpired(request);
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
     * Fence-deferred settle audit: finished members whose settle was skipped
     * because the dispatch-reconciliation fence still owns them. One WARN
     * per batch per {@link #SETTLE_DEFER_WARN_RATE_MS} — an unresolved fence
     * otherwise repeats on every ~20ms calibrate round because finished
     * snapshots re-report the same members. Pinpoints the "fence postpones
     * settle" path when batches outlive their members' engine terminals.
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

    /**
     * Active engine tasks the local batch ledger does not track — the S3
     * scoring input ({@code CostBasedPrefillStrategy} engine-untracked
     * penalty). Refreshed on every worker status sync.
     */
    public long getEngineUntrackedRequestCount() {
        return engineUntrackedRequestCount.get();
    }

    // ==================== Pending Offers (R1) ====================

    /**
     * Requests route() already committed to this endpoint but that the
     * batcher has not yet accepted — the route→offer blind window the
     * Round-1 score cannot see through any other ledger (R1, 205 pileup
     * incident). requestId → record timestamp.
     *
     * <p>Correctness over completeness: recording is idempotent (re-route
     * overwrites the timestamp), releasing is idempotent (remove of an
     * absent key is a no-op), and any entry leaked by a path that never
     * reaches the batcher (e.g. a decode-side rollback that skips the
     * prefill strategy rollback) self-heals via the lazy
     * {@link #PENDING_OFFER_TTL_MS} expiry in the read path — a stale
     * pending entry can overstate the score for at most a few seconds.
     */
    private final ConcurrentHashMap<Long, Long> pendingOffers = new ConcurrentHashMap<>();

    /** Lazy expiry for leaked pending-offer entries; route→offer is normally sub-ms. */
    private static final long PENDING_OFFER_TTL_MS = 5_000L;

    /** Route committed this request here; the batcher has not seen it yet. */
    public void recordPendingOffer(long requestId) {
        pendingOffers.put(requestId, System.currentTimeMillis());
    }

    /** The batcher took over (or the route was rolled back) — idempotent. */
    public void releasePendingOffer(long requestId) {
        pendingOffers.remove(requestId);
    }

    /**
     * Live pending-offer count for Round-1 scoring; prunes entries older
     * than {@link #PENDING_OFFER_TTL_MS} so leaked entries self-heal.
     */
    public int getPendingOfferCount() {
        long cutoffMs = System.currentTimeMillis() - PENDING_OFFER_TTL_MS;
        pendingOffers.values().removeIf(recordedAtMs -> recordedAtMs < cutoffMs);
        return pendingOffers.size();
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
     * post-ACK inflight audit (F1). An entry invisible here and on the decode
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
     * cap (F-F, {@code flexlbBatchInflightMaxAgeMs} +
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
     * engine-untracked (R5 compensation — the engine may still be executing
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
     * <p>Frozen-batch audit: batches that <b>survive</b> this sweep while
     * older than {@code flexlbBatchFrozenAuditAfterMs} (default 60s, aligned
     * with the 60s sweep cadence) emit one WARN audit line each — carrying
     * the exact verdict fields (over_age_cap / stale, dispatch fence,
     * scheduler ownership, member terminal distribution) so the next freeze
     * immediately reveals which exemption leg kept the batch alive.
     * Rate-limited to {@value #FROZEN_BATCH_AUDIT_MAX_LINES} lines per
     * endpoint per sweep; the audited-batch count is reported through
     * {@code reportBatchInflightFrozenAudit} ({@code <= 0} threshold
     * disables the audit entirely).
     *
     * @param ttlMs                 max unobserved age before normal eviction
     * @param hardMaxAgeMs          guarded hard creation-age cap;
     *                              {@code <= 0} disables
     * @param batchInflightMaxAgeMs batch-level age cap (F-F);
     *                              {@code <= 0} disables
     * @param batchInflightStaleMs  no-progress staleness threshold for the
     *                              age cap; {@code <= 0} drops the
     *                              progress guard (pure age cap)
     * @param schedulerOwnsRequest  whether the scheduler still tracks a
     *                              request (race guard for the guarded
     *                              hard-cap branch only)
     * @return number of batches removed (age-capped + guarded hard-cap +
     *         normal TTL)
     */
    public int evictExpiredBatches(long ttlMs, long hardMaxAgeMs, long batchInflightMaxAgeMs,
                                   long batchInflightStaleMs, LongPredicate schedulerOwnsRequest) {
        long nowMs = System.currentTimeMillis();
        AtomicInteger evictedCount = new AtomicInteger();
        AtomicInteger frozenAuditLines = new AtomicInteger(FROZEN_BATCH_AUDIT_MAX_LINES);
        for (Long batchId : inflightBatches.keySet()) {
            AtomicReference<BatchInflight> evicted = new AtomicReference<>();
            AtomicReference<BatchInflight> forced = new AtomicReference<>();
            AtomicReference<BatchInflight> ageCapped = new AtomicReference<>();
            inflightBatches.computeIfPresent(batchId, (id, batch) -> {
                long ageMs = nowMs - batch.createdAtMs();
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
            BatchInflight cappedBatch = ageCapped.get();
            if (cappedBatch != null) {
                forceSettleAgeCappedBatch(batchId, cappedBatch,
                        nowMs - cappedBatch.createdAtMs(),
                        nowMs - cappedBatch.lastObservedAtMs(), batchInflightMaxAgeMs);
                evictedCount.incrementAndGet();
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
                evictedCount.incrementAndGet();
            } else {
                // Retained by one of the exemption legs (fence / observed
                // freshness / under every threshold) — frozen-batch audit.
                auditFrozenBatch(batchId, nowMs, batchInflightMaxAgeMs,
                        batchInflightStaleMs, schedulerOwnsRequest, frozenAuditLines);
            }
        }
        int auditedCount = FROZEN_BATCH_AUDIT_MAX_LINES - frozenAuditLines.get();
        if (auditedCount > 0 && reporter != null) {
            RoleType auditRole = status.getRole();
            reporter.reportBatchInflightFrozenAudit(
                    auditRole != null ? auditRole.name() : RoleType.PREFILL.name(),
                    getIp(), auditedCount);
        }
        return evictedCount.get();
    }

    /** Max frozen-batch audit WARN lines per endpoint per sweep (anti-flood). */
    private static final int FROZEN_BATCH_AUDIT_MAX_LINES = 5;

    /**
     * One frozen-batch audit line for a batch that survived the sweep while
     * older than {@code flexlbBatchFrozenAuditAfterMs}. Returns whether a
     * line was emitted (consumes one unit of the per-sweep budget; batches
     * past the budget or under the threshold are skipped silently).
     *
     * <p>Field semantics — the line answers "why is this batch still alive":
     * <ul>
     * <li>{@code over_age_cap} / {@code stale}: the exact F-F cap verdict
     *     legs — {@code over_age_cap=true, stale=false} is the observed-
     *     freshness exemption (the "kept alive by observation" freeze), a
     *     fence shows as {@code fenced=true};</li>
     * <li>{@code scheduler_owned}: at least one member is still tracked by
     *     the scheduler ledger;</li>
     * <li>{@code members_terminal/pending}: how many member futures already
     *     reached a terminal state — a fully-terminal member set on a live
     *     batch is the smoking gun for a lost settle;</li>
     * <li>{@code observation_misses}: consecutive engine reports that
     *     mentioned this batch nowhere (Fix A counter).</li>
     * </ul>
     */
    private boolean auditFrozenBatch(long batchId, long nowMs, long batchInflightMaxAgeMs,
                                     long batchInflightStaleMs, LongPredicate schedulerOwnsRequest,
                                     AtomicInteger linesLeft) {
        if (linesLeft.get() <= 0) {
            return false;
        }
        long auditAfterMs = config != null ? config.getFlexlbBatchFrozenAuditAfterMs() : 0;
        if (auditAfterMs <= 0) {
            return false;
        }
        BatchInflight batch = inflightBatches.get(batchId);
        if (batch == null) {
            return false;
        }
        long ageMs = nowMs - batch.createdAtMs();
        if (ageMs <= auditAfterMs) {
            return false;
        }
        linesLeft.decrementAndGet();
        long lastObservedAgoMs = nowMs - batch.lastObservedAtMs();
        boolean overAgeCap = batchInflightMaxAgeMs > 0 && ageMs > batchInflightMaxAgeMs;
        boolean stale = batchInflightStaleMs > 0 && lastObservedAgoMs > batchInflightStaleMs;
        boolean fenced = hasDispatchReconciliation(batchId);
        boolean schedulerOwned = batch.requests().stream()
                .anyMatch(item -> schedulerOwnsRequest.test(item.requestId()));
        int terminalMembers = 0;
        for (BatchItem request : batch.requests()) {
            if (request.future() != null && request.future().isDone()) {
                terminalMembers++;
            }
        }
        int members = batch.requests().size();
        RoleType role = status.getRole();
        org.flexlb.util.Logger.warn(
                "event=inflight_batch_frozen_audit role={} endpoint={} batch_id={} "
                        + "age_ms={} last_observed_ago_ms={} fenced={} scheduler_owned={} "
                        + "member_count={} members_terminal={} members_pending={} "
                        + "inflight_request_count={} over_age_cap={} stale={} "
                        + "age_cap_ms={} stale_ms={} observation_misses={} running={}",
                role != null ? role.name() : RoleType.PREFILL.name(), getIp(), batchId,
                ageMs, lastObservedAgoMs, fenced, schedulerOwned,
                members, terminalMembers, members - terminalMembers,
                inflightRequestCount.get(), overAgeCap, stale,
                batchInflightMaxAgeMs, batchInflightStaleMs, batch.observationMisses(),
                batch.running());
        return true;
    }

    /**
     * F-F bounded-freeze release for one age-capped inflight batch, called
     * after the per-key compute already removed it from {@code inflightBatches}
     * (so the visible {@code inflight.batch.count} drops immediately).
     * Drops the reconciliation fence (the authoritative settlement this
     * fence is waiting for will never arrive — that is why the batch
     * reached the cap), decrements the request counter, compensates the
     * engine-untracked counter (R5: the engine may still be executing the
     * members — re-reserve them so the fixed-window inflight gate does not
     * oversell in the short window before the next status sync recomputes
     * the exact count), emits one WARN line + the age-cap metric tagged
     * with the endpoint's own role, and routes every member through the
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
        // R5 compensation: re-reserve the members as engine-untracked so the
        // released inflight gate does not oversell before the next worker
        // status sync recomputes the count.
        engineUntrackedRequestCount.addAndGet(members);
        cachedWaitTimeExpireAtMs = 0;
        // W2: tag with the endpoint's own role, not a hard-coded PREFILL.
        RoleType role = status.getRole();
        String roleName = role != null ? role.name() : RoleType.PREFILL.name();
        org.flexlb.util.Logger.warn(
                "event=inflight_batch_age_capped role={} endpoint={} batch_id={} "
                        + "age_ms={} last_observed_ago_ms={} max_age_ms={} fenced={} "
                        + "n_requests={} request_ids={}",
                roleName, getIp(), batchId, ageMs, lastObservedAgoMs, maxAgeMs, hadFence,
                members, batch.originalRequestIds());
        if (reporter != null) {
            reporter.reportBatchInflightAgeCapped(roleName, getIp(), 1);
        }
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
            // The fence was removed under the same inflight-key critical section,
            // so apply the cached terminal directly instead of trying to defer it again.
            return applyFinishedObservations(
                    id, batch, List.of(observation), statusMs, false, completed);
        });
        BatchInflight completedBatch = completed.get();
        if (completedBatch != null) {
            reportBatchCompletion(batchId, completedBatch);
        }
    }

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
