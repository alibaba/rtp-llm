package org.flexlb.balance.endpoint;

import org.flexlb.balance.projection.WorkSnapshot.Phase;
import org.flexlb.balance.scheduler.RequestRoute;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.enums.PriorityPreemptionProgress;
import org.flexlb.enums.TaskPhase;
import java.util.*;
import static com.google.common.base.Preconditions.checkState;
import static com.google.common.math.LongMath.saturatedAdd;
import static org.flexlb.balance.endpoint.PrefillState.*;

/** Builds independent resource facts under the owner lock; it never commits or runs effects. */
public final class PrefillObservation {
    private PrefillObservation() { }

    public static StatusReduction prepareLocked(PrefillState state, WorkerStatus.StatusObservation observation, boolean full) {
        checkState(state.lock.isHeldByCurrentThread(), "Prefill observation requires ownership lock");
        long nowMs = full ? state.clock.getAsLong() : 0L;
        var reduction = new StatusReduction(state, full ? terminals(state, observation.finishedTasks()) : Map.of());
        List<PrefillRequestStatus> statuses = new ArrayList<>(reduction.terminals.size() + observation.engine().runningTaskList().size());
        for (var terminal : reduction.terminals.values()) {
            var kind = terminal.errorCode() == 0L ? PrefillRequestStatus.Kind.COMPLETED
                    : terminal.preemptionProgress() == PriorityPreemptionProgress.CANCELED
                    && terminal.errorCode() == StrategyErrorType.PRIORITY_PREEMPTED.getErrorCode()
                    ? PrefillRequestStatus.Kind.PRIORITY_CANCELED : PrefillRequestStatus.Kind.FAILED;
            statuses.add(new PrefillRequestStatus(terminal.owner().route, kind, terminal.errorCode()));
            var batch = terminal.owner().batch;
            if (batch == null) { continue; }
            var outcome = reduction.batchOutcomes.get(batch);
            if (outcome == null) {
                outcome = new BatchOutcome();
                outcome.maxExecutionTimeMs = batch.outcome.maxExecutionTimeMs;
                outcome.successfulCompletion = batch.outcome.successfulCompletion;
                outcome.learningEligible = batch.outcome.learningEligible;
                outcome.executionStarted = batch.outcome.executionStarted;
                reduction.batchOutcomes.put(batch, outcome);
            }
            boolean success = terminal.errorCode() == 0L;
            outcome.executionStarted |= success || terminal.executionTimeMs() > 0L;
            outcome.maxExecutionTimeMs = Math.max(outcome.maxExecutionTimeMs, terminal.executionTimeMs());
            outcome.successfulCompletion |= success;
            outcome.learningEligible &= success;
        }
        active(state, observation.engine(), reduction, statuses);
        boolean released = !reduction.terminals.isEmpty() || reduction.unknownRequests < state.unknownEngineRequestCount;
        reduction.nowMs = full ? nowMs : state.clock.getAsLong();
        Map<Long, List<RequestRoute>> inputs = reduction.batchOutcomes.isEmpty() ? Map.of() : new HashMap<>();
        List<BatchCompletion> completions = reduction.batchOutcomes.isEmpty() ? List.of() : new ArrayList<>();
        for (var change : reduction.batchOutcomes.entrySet()) {
            var batch = change.getKey(); var outcome = change.getValue();
            List<RequestRoute> survivors = new ArrayList<>(batch.members.size());
            for (var member : batch.members) { if (!reduction.terminals.containsKey(member.route.requestId())) { survivors.add(member.route); } }
            if (survivors.isEmpty()) {
                completions.add(new BatchCompletion(batch.batchId, batch.originalFeatures, batch.originalPredictionMs,
                        outcome.maxExecutionTimeMs, outcome.successfulCompletion, outcome.learningEligible));
            } else if (!outcome.executionStarted && reduction.batchPhases.get(batch) != Phase.ENGINE_RUNNING) {
                survivors.sort(Comparator.comparingLong(RequestRoute::enqueueSeq).thenComparingLong(RequestRoute::requestId));
                inputs.put(batch.batchId, List.copyOf(survivors));
            }
        }
        boolean changed = reduction.unknownRequests != state.unknownEngineRequestCount;
        for (var phase : reduction.individualPhases.entrySet()) { changed |= phase.getKey().individualPhase != phase.getValue(); }
        for (var phase : reduction.batchPhases.entrySet()) { changed |= phase.getKey().servicePhase != phase.getValue(); }
        reduction.version = state.progress().ownershipVersion();
        reduction.predictionInputs = Map.copyOf(inputs);
        reduction.result = new StatusReconciliation(statuses, completions,
                changed || !reduction.terminals.isEmpty() || !inputs.isEmpty(), released);
        return reduction;
    }

    private static Map<Long, TerminalObservation> terminals(PrefillState state, Map<String, WorkerStatus.TaskObservation> tasks) {
        Map<Long, TerminalObservation> facts = new HashMap<>();
        for (var task : tasks.values()) {
            var entry = state.requests.get(task.requestId());
            if (entry == null || entry.ownership != OwnershipStage.COMMITTED
                    || entry.batch != null && (task.batchId() <= 0L || entry.batch.batchId != task.batchId())) { continue; }
            var next = new TerminalObservation(entry, task.executionTimeMs(), task.errorCode(), task.priorityPreemptionProgress());
            facts.merge(task.requestId(), next, (first, other) -> new TerminalObservation(first.owner(),
                    Math.max(first.executionTimeMs(), other.executionTimeMs()), first.errorCode() != 0L ? first.errorCode() : other.errorCode(),
                    PriorityPreemptionProgress.merge(first.preemptionProgress(), other.preemptionProgress())));
        }
        return facts;
    }

    private static void active(PrefillState state, WorkerStatus.EngineObservation engine,
            StatusReduction reduction, List<PrefillRequestStatus> statuses) {
        Set<Long> unknown = new HashSet<>(), known = new HashSet<>();
        for (var task : engine.runningTaskList().values()) {
            if (reduction.terminals.containsKey(task.requestId())) { continue; }
            var entry = state.requests.get(task.requestId());
            if (entry == null || entry.batch != null && (task.batchId() <= 0L || entry.batch.batchId != task.batchId())) {
                if (!task.isPriorityCancelOverlayOnly()) { unknown.add(task.requestId()); }
                continue;
            }
            if (!task.isPriorityCancelOverlayOnly()) {
                known.add(task.requestId());
                if (entry.ownership == OwnershipStage.COMMITTED) { statuses.add(new PrefillRequestStatus(entry.route, PrefillRequestStatus.Kind.ACTIVE, 0L)); }
            }
            if (entry.ownership != OwnershipStage.COMMITTED) { continue; }
            Phase phase = task.phase() == TaskPhase.RUNNING ? Phase.ENGINE_RUNNING : Phase.ENGINE_QUEUED;
            if (entry.batch == null) { reduction.individualPhases.put(entry, phase); }
            else { reduction.batchPhases.merge(entry.batch, phase, (left, right) -> left == Phase.ENGINE_RUNNING || right == Phase.ENGINE_RUNNING ? Phase.ENGINE_RUNNING : Phase.ENGINE_QUEUED); }
        }
        long total = saturatedAdd(Math.max(0L, engine.waitingQueryLen()), Math.max(0L, engine.runningQueryLen()));
        reduction.unknownRequests = Math.max(unknown.size(), Math.max(0L, total - known.size()));
    }
}
