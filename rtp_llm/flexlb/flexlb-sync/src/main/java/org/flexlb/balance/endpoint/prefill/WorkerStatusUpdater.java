package org.flexlb.balance.endpoint.prefill;

import org.flexlb.balance.endpoint.prefill.batch.BatchPrefill;
import org.flexlb.balance.endpoint.prefill.batch.BatchProgress;
import org.flexlb.balance.endpoint.prefill.batch.FinishedBatch;
import org.flexlb.balance.endpoint.prefill.batch.LiveBatchMembers;
import org.flexlb.balance.estimate.WorkView.Phase;
import org.flexlb.balance.scheduler.ScheduledRequest;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.enums.PriorityPreemptionProgress;
import org.flexlb.enums.TaskPhase;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.ToLongFunction;

/**
 * 根据引擎上报的状态和心跳更新 Prefill 的请求记录：删除已完成的请求，更新排队中和运行中的进度，
 * 更新引擎上有、本地不认识的请求数，并生成交给调度器的状态事件。请求表和锁仍归 PrefillState。
 */
public final class WorkerStatusUpdater {
    private final PrefillState prefillState;

    public WorkerStatusUpdater(PrefillState prefillState) {
        this.prefillState = prefillState;
    }

    /**
     * Reconcile ownership and publish its matching WorkerStatus before the
     * canonical queue lock is released. Load view readers therefore observe
     * either the previous pair or the fully reduced new pair.
     */
    public WorkerStatusResult updateFromWorkerStatus(WorkerStatus.StatusObservation observation,
                                                     ToLongFunction<List<ScheduledRequest>> predictBatchTime,
                                                     Runnable publishStatus,
                                                     Runnable onPublishFailure) {
        return reconcileEngineStatus(observation.engine(), observation.finishedTasks(), predictBatchTime, publishStatus,
                onPublishFailure);
    }

    public HeartbeatResult updateFromHeartbeat(WorkerStatus.StatusObservation observation) {
        List<WorkerStatusFact> facts = new ArrayList<>(
                observation.activeTasks().size());
        boolean capacityReleased = false;
        boolean schedulingInputsChanged = false;
        prefillState.lock.lock();
        try {
            long nowMs = prefillState.clock.getAsLong();
            WorkerStatus.EngineObservation engine = observation.engine();
            IdentityHashMap<RequestEntry, Phase> individualPhases =
                    new IdentityHashMap<>();
            IdentityHashMap<BatchProgress, Phase> batchPhases = new IdentityHashMap<>();
            long nextUnknown = collectRunningTasks(engine.runningTaskList(), Map.of(), PrefillState
                    .saturatedAdd(Math.max(0L, engine.waitingQueryLen()), Math.max(0L, engine.runningQueryLen())),
                    individualPhases, batchPhases, facts);
            schedulingInputsChanged = nextUnknown != prefillState.unknownEngineRequestCount;
            for (var observed : individualPhases.entrySet()) {
                schedulingInputsChanged |= observed.getKey().singleRequestProgress.phase() != observed.getValue();
                observed.getKey().singleRequestProgress.updateProgressFromWorkerStatus(observed.getValue(), nowMs);
            }
            for (var observed : batchPhases.entrySet()) {
                schedulingInputsChanged |= observed.getKey().phase() != observed.getValue();
                observed.getKey().updateProgressFromWorkerStatus(observed.getValue(), nowMs);
            }
            capacityReleased = nextUnknown < prefillState.unknownEngineRequestCount;
            prefillState.unknownEngineRequestCount = nextUnknown;
            if (schedulingInputsChanged) {
                prefillState.committedWorkCapture = null;
                prefillState.recordMutationUnderLock();
            }
        } finally {
            prefillState.lock.unlock();
            prefillState.notifyCapacityAvailable(capacityReleased);
        }
        return new HeartbeatResult(facts, schedulingInputsChanged);
    }

    private WorkerStatusResult reconcileEngineStatus(WorkerStatus.EngineObservation engine,
                                                     Map<String, WorkerStatus.TaskObservation> finishedTasks,
                                                     ToLongFunction<List<ScheduledRequest>> predictBatchTime,
                                                     Runnable publishStatus,
                                                     Runnable onPublishFailure) {
        boolean capacityReleased = false;
        boolean canonicalMutationStarted = false;
        WorkerStatusResult outcome = null;
        prefillState.lock.lock();
        try {
            long nowMs = prefillState.clock.getAsLong();
            IdentityHashMap<BatchProgress, LiveBatchMembers> membersByBatch =
                    prefillState.batchPrefill.liveBatchMembersUnderLock();
            Map<String, TerminalObservation> terminals = collectFinishedTasks(finishedTasks);
            Map<String, WorkerStatus.TaskObservation> activeTasks =
                    engine.runningTaskList();
            IdentityHashMap<RequestEntry, TerminalObservation> settlements =
                    new IdentityHashMap<>(terminals.size());
            IdentityHashMap<BatchProgress, Phase> batchPhases = new IdentityHashMap<>();
            for (TerminalObservation terminal : terminals.values()) {
                RequestEntry entry = prefillState.requests.get(terminal.requestId());
                if (entry == null
                        || settlements.put(entry, terminal) != null) {
                    throw new IllegalStateException(
                            "status terminal lost its exact owner request_id=" + terminal.requestId());
                }
            }
            Set<LiveBatchMembers> changedBatches =
                    BatchPrefill.collectChangedBatches(settlements.keySet(), membersByBatch);
            long reportedActive = PrefillState.saturatedAdd(
                    Math.max(0L, engine.waitingQueryLen()),
                    Math.max(0L, engine.runningQueryLen()));
            IdentityHashMap<RequestEntry, Phase> individualPhases =
                    new IdentityHashMap<>();
            long nextUnknown =
                    collectRunningTasks(activeTasks, terminals, reportedActive, individualPhases, batchPhases, null);
            IdentityHashMap<BatchProgress, Long> predictions = new IdentityHashMap<>();
            prefillState.batchPrefill.prepareBatchPredictionsUnderLock(changedBatches, terminals, batchPhases,
                    predictBatchTime, predictions);
            outcome = buildResult(terminals, activeTasks, membersByBatch);

            // Everything below this boundary is assignment/removal against
            // exact prevalidated identities. Any invariant failure is captured
            // into the already materialized outcome and forces retirement.
            canonicalMutationStarted = true;
            prefillState.committedWorkCapture = null;
            prefillState.recordMutationUnderLock();
            for (Map.Entry<RequestEntry, TerminalObservation> settlement
                    : settlements.entrySet()) {
                RequestEntry entry = settlement.getKey();
                capacityReleased |= prefillState.removeCommittedEntry(entry, settlement.getValue(),
                        entry.batchProgress == null ? null : membersByBatch.get(entry.batchProgress), null, nowMs);
            }
            individualPhases.forEach(
                    (entry, phase) -> entry.singleRequestProgress.updateProgressFromWorkerStatus(phase, nowMs));
            batchPhases.forEach((batch, phase) -> batch.updateProgressFromWorkerStatus(phase, nowMs));
            predictions.forEach((batch, prediction) -> batch.resetRemainingWork(prediction, nowMs));
            capacityReleased |= nextUnknown < prefillState.unknownEngineRequestCount;
            prefillState.unknownEngineRequestCount = nextUnknown;
            // Status reduction changes counts after invalidating the old work revision.
            prefillState.publishRequestCountUnderLock();
            try {
                publishStatus.run();
            } catch (Throwable failure) {
                outcome = withPublicationFailure(outcome, failure);
                try {
                    onPublishFailure.run();
                } catch (Throwable failClosedFailure) {
                    // The first publication failure is the canonical outcome.
                }
            }
        } catch (RuntimeException | Error failure) {
            if (canonicalMutationStarted && outcome != null) {
                outcome = withPublicationFailure(outcome, failure);
                try {
                    onPublishFailure.run();
                } catch (Throwable failClosedFailure) {
                    // The first reduction failure is the canonical outcome.
                }
            } else {
                try {
                    onPublishFailure.run();
                } catch (Throwable failClosedFailure) {
                    failure.addSuppressed(failClosedFailure);
                }
                throw failure;
            }
        } finally {
            prefillState.lock.unlock();
            prefillState.notifyCapacityAvailable(capacityReleased);
        }
        return outcome;
    }

    private static WorkerStatusResult withPublicationFailure(WorkerStatusResult outcome, Throwable failure) {
        return outcome.publicationFailure() != null
                ? outcome
                : new WorkerStatusResult(outcome.schedulerFacts(), outcome.finishedBatches(), failure);
    }

    private long collectRunningTasks(Map<String, WorkerStatus.TaskObservation> activeTasks,
                                     Map<String, TerminalObservation> terminals,
                                     long reportedActive,
                                     IdentityHashMap<RequestEntry, Phase> individualPhases,
                                     IdentityHashMap<BatchProgress, Phase> batchPhases,
                                     List<WorkerStatusFact> activeFacts) {
        prefillState.requireLock();
        Set<String> unknownDetailed = new HashSet<>();
        Set<String> knownObserved = new HashSet<>();
        for (WorkerStatus.TaskObservation task : activeTasks.values()) {
            if (terminals.containsKey(task.requestId())) {
                continue;
            }
            RequestEntry entry = prefillState.requests.get(task.requestId());
            if (entry == null || !BatchPrefill.matchesObservedBatch(entry, task.batchId())) {
                if (!task.isPriorityCancelOverlayOnly()) {
                    unknownDetailed.add(task.requestId());
                }
                continue;
            }
            if (!task.isPriorityCancelOverlayOnly()) {
                knownObserved.add(task.requestId());
                if (activeFacts != null && !entry.isActive()) {
                    activeFacts.add(WorkerStatusFact.active(entry.committedItem));
                }
            }
            if (entry.isActive()) {
                continue;
            }
            Phase observed = task.phase() == TaskPhase.RUNNING
                    ? Phase.ENGINE_RUNNING : Phase.ENGINE_QUEUED;
            if (entry.batchProgress == null) {
                individualPhases.put(entry, observed);
            } else {
                batchPhases.merge(entry.batchProgress, observed, BatchPrefill::strongerEnginePhase);
            }
        }
        long scalarUnknown = Math.max(
                0L, reportedActive - knownObserved.size());
        return Math.max(unknownDetailed.size(), scalarUnknown);
    }

    private Map<String, TerminalObservation> collectFinishedTasks(
            Map<String, WorkerStatus.TaskObservation> finishedTasks) {
        prefillState.requireLock();
        Map<String, TerminalObservation> terminals = new HashMap<>();
        for (WorkerStatus.TaskObservation task : finishedTasks.values()) {
            RequestEntry entry = prefillState.requests.get(task.requestId());
            if (entry == null || !BatchPrefill.matchesObservedBatch(entry, task.batchId())) {
                continue;
            }
            TerminalObservation terminal = TerminalObservation.from(task, entry.batchProgress != null);
            terminals.merge(
                    task.requestId(), terminal, TerminalObservation::merge);
        }
        return terminals;
    }

    /**
     * Materialize every externally observable status fact before the first
     * canonical entry is settled. The returned object is the exact outcome
     * later published by the endpoint; reduction only binds a callback failure.
     */
    private WorkerStatusResult buildResult(Map<String, TerminalObservation> terminals,
                                           Map<String, WorkerStatus.TaskObservation> activeTasks,
                                           IdentityHashMap<BatchProgress, LiveBatchMembers> membersByBatch) {
        prefillState.requireLock();
        List<WorkerStatusFact> schedulerFacts = new ArrayList<>(
                terminals.size() + activeTasks.size());
        for (TerminalObservation terminal : terminals.values()) {
            WorkerStatusFact fact = finishedFact(prefillState.requests.get(terminal.requestId()), terminal);
            if (fact != null) {
                schedulerFacts.add(fact);
            }
        }
        for (WorkerStatus.TaskObservation task : activeTasks.values()) {
            if (terminals.containsKey(task.requestId())) {
                continue;
            }
            WorkerStatusFact fact = runningFact(task);
            if (fact != null) {
                schedulerFacts.add(fact);
            }
        }

        List<FinishedBatch> finishedBatches = BatchPrefill.collectFinishedBatches(membersByBatch, terminals);
        return new WorkerStatusResult(schedulerFacts, finishedBatches, null);
    }

    /** Resolve activity only through the exact committed ledger identity. */
    private WorkerStatusFact runningFact(WorkerStatus.TaskObservation task) {
        prefillState.requireLock();
        RequestEntry entry = prefillState.requests.get(task.requestId());
        if (entry == null || entry.isActive() || !BatchPrefill.matchesObservedBatch(entry, task.batchId())
                || entry.committedItem == null || task.isPriorityCancelOverlayOnly()) {
            return null;
        }
        return WorkerStatusFact.active(entry.committedItem);
    }

    private static WorkerStatusFact finishedFact(RequestEntry entry, TerminalObservation terminal) {
        if (entry == null || entry.committedItem == null || !terminal.workerObserved()) {
            return null;
        }
        WorkerStatusFact.Kind kind = terminal.errorCode() == 0L
                ? WorkerStatusFact.Kind.COMPLETED
                : terminal.preemptionProgress() == PriorityPreemptionProgress.CANCELED
                        && terminal.errorCode() == StrategyErrorType.PRIORITY_PREEMPTED.getErrorCode()
                                ? WorkerStatusFact.Kind.PRIORITY_CANCELED
                                : WorkerStatusFact.Kind.FAILED;
        return WorkerStatusFact.terminal(entry.committedItem, kind, terminal.errorCode());
    }
}
