package org.flexlb.balance.endpoint.prefill;

import org.flexlb.balance.delivery.DeliveryBlockReason;
import org.flexlb.balance.endpoint.EndpointGenerationLifecycle;
import org.flexlb.balance.endpoint.prefill.batch.BatchPrefill;
import org.flexlb.balance.endpoint.prefill.batch.BatchProgress;
import org.flexlb.balance.endpoint.prefill.batch.BatchReservation;
import org.flexlb.balance.endpoint.prefill.batch.FinishedBatch;
import org.flexlb.balance.endpoint.prefill.batch.LiveBatchMembers;
import org.flexlb.balance.endpoint.prefill.single.QueuedPreemption;
import org.flexlb.balance.endpoint.prefill.single.SinglePrefill;
import org.flexlb.balance.endpoint.prefill.single.SingleReservation;
import org.flexlb.balance.estimate.WorkView;
import org.flexlb.balance.estimate.WorkView.Phase;
import org.flexlb.balance.scheduler.ScheduledRequest;
import org.flexlb.dao.loadbalance.AdmissionRejectReason;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.enums.PriorityPreemptionProgress;
import org.flexlb.enums.TaskPhase;
import org.flexlb.util.PriorityNormalizer;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.OptionalLong;
import java.util.Set;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.LongSupplier;
import java.util.function.ToLongFunction;

/**
 * Canonical Prefill request ownership for one worker generation.
 *
 * <p>Every known request id has exactly one {@link RequestEntry}. The worker
 * queue is only an ordered index over entries waiting for worker delivery;
 * an immediate admission owns the same resource record without a queue entry.
 * Callback and Engine progress mutate the same entry instead of moving ownership
 * between containers. All methods which end in {@code UnderLock} require the
 * worker's queue lock, which is the sole Prefill ownership lock.
 */
public final class PrefillState {

    public record WorkerStatusFact(
            ScheduledRequest item,
            Kind kind,
            long errorCode) {
        public WorkerStatusFact {
            Objects.requireNonNull(kind, "kind");
            if (kind == Kind.ACTIVE && errorCode != 0L) {
                throw new IllegalArgumentException(
                        "an active Prefill fact cannot carry an error code");
            }
        }

        public static WorkerStatusFact active(ScheduledRequest item) {
            return new WorkerStatusFact(item, Kind.ACTIVE, 0L);
        }

        public static WorkerStatusFact terminal(
                ScheduledRequest item, Kind kind, long errorCode) {
            if (kind == Kind.ACTIVE) {
                throw new IllegalArgumentException(
                        "terminal Prefill fact requires a terminal kind");
            }
            return new WorkerStatusFact(item, kind, errorCode);
        }

        public enum Kind {
            ACTIVE,
            COMPLETED,
            FAILED,
            PRIORITY_CANCELED
        }
    }

    public record StatusReconciliation(List<WorkerStatusFact> schedulerFacts,
                                       List<FinishedBatch> finishedBatches,
                                       Throwable publicationFailure) {
        public StatusReconciliation {
            schedulerFacts = List.copyOf(schedulerFacts);
            finishedBatches = List.copyOf(finishedBatches);
        }
    }

    public record Retirement(List<ScheduledRequest> ownedItems,
                             List<FinishedBatch> finishedBatches,
                             Throwable invariantFailure) {
        public Retirement {
            ownedItems = List.copyOf(ownedItems);
            finishedBatches = List.copyOf(finishedBatches);
        }
    }

    public record Stats(
            long inflightRequests,
            int locallyOwnedRequests,
            int workerStatusUnconfirmedRequests,
            int individuallyOwnedRequests,
            int batchCount,
            long maxObservedAgeMs) {
        public Stats {
            if (inflightRequests < 0L || locallyOwnedRequests < 0 || workerStatusUnconfirmedRequests < 0
                    || individuallyOwnedRequests < 0
                    || batchCount < 0 || maxObservedAgeMs < 0L) {
                throw new IllegalArgumentException(
                        "Prefill state stats must be non-negative");
            }
        }
    }

    /** One-shot capability returned when an OPEN admission commits. */
    public static final class CommittedHandoff implements AutoCloseable {
        public EndpointGenerationLifecycle.HandoffPermit generationHandoff;
        private final WorkView precedingWork;

        public CommittedHandoff(EndpointGenerationLifecycle.HandoffPermit generationHandoff, WorkView precedingWork) {
            this.generationHandoff = generationHandoff;
            this.precedingWork = precedingWork;
        }

        /** Other reserved work at the same ownership boundary that committed this admission. */
        public WorkView precedingWork() {
            return precedingWork;
        }

        @Override
        public synchronized void close() {
            EndpointGenerationLifecycle.HandoffPermit exact = generationHandoff;
            if (exact == null) {
                return;
            }
            generationHandoff = null;
            exact.close();
        }
    }

    /** Queue and committed work captured at one ownership linearization point. */
    public record Snapshot(long capturedAtMs,
                           PrefillActiveIndex.Capture active,
                           WorkCapture work) {

        public List<ScheduledRequest> activeItems() {
            return active.items();
        }
    }

    /** Immutable leaves copied under the ownership lock; load view is built outside it. */
    public static final class WorkCapture {
        private final long capturedAtMs;
        private final List<WorkView.RequestWork> requests;
        private final List<WorkView.BatchWork> batches;
        private final long unknownRequestCount;
        private volatile WorkView materialized;

        private WorkCapture(long capturedAtMs, List<WorkView.RequestWork> requests,
                            List<WorkView.BatchWork> batches, long unknownRequestCount) {
            this.capturedAtMs = capturedAtMs;
            this.requests = List.copyOf(requests);
            this.batches = List.copyOf(batches);
            this.unknownRequestCount = unknownRequestCount;
        }

        /** Shared lazy load view; callers never hold the endpoint ownership lock. */
        public WorkView materialize() {
            WorkView snapshot = materialized;
            if (snapshot != null) {
                return snapshot;
            }
            synchronized (this) {
                if (materialized == null) {
                    List<WorkView.RequestWork> orderedRequests = requests.stream()
                            .sorted(Comparator.comparing(WorkView.RequestWork::requestId)).toList();
                    List<WorkView.BatchWork> orderedBatches = batches.stream()
                            .sorted(Comparator.comparingLong(WorkView.BatchWork::batchId))
                            .map(batch -> new WorkView.BatchWork(batch.batchId(),
                                    batch.requestIds().stream().sorted().toList(),
                                    batch.phase(), batch.remainingWorkMs())).toList();
                    materialized = new WorkView(capturedAtMs, orderedRequests, orderedBatches, unknownRequestCount);
                }
                return materialized;
            }
        }
    }

    public final ReentrantLock lock;
    /** Non-owning index containing only ACTIVE ScheduledRequest identities. */
    public final PrefillActiveIndex activeIndex;
    /** Canonical request ownership, changed only under the endpoint lock. */
    public final Map<String, RequestEntry> requests = new HashMap<>();
    public final LongSupplier clock;
    public final Runnable capacityAvailable;
    /** Monotonic ownership/work revision used by load views. */
    private volatile long mutationVersion;
    /** Published at ownership mutation boundaries; admission still checks under the lock. */
    private volatile long outstandingRequestCount;
    /** Committed routes and unrecognized Engine work; ACTIVE queue items do not occupy delivery seats. */
    private volatile long dispatchedRequestCount;
    private long requestCountsVersion = -1;
    private long[] requestCountsByPriority;
    /** Derived immutable work; ACTIVE queue mutations leave committed work unchanged. */
    public WorkCapture committedWorkCapture;
    public long unknownEngineRequestCount;
    public final BatchPrefill batchPrefill = new BatchPrefill(this);
    public final SinglePrefill singlePrefill = new SinglePrefill(this);
    public final QueuedPreemption queuedPreemption = new QueuedPreemption(this);

    /** Publish the capacity summary before readers observe a new ownership revision. */
    public void recordMutationUnderLock() {
        publishRequestCountUnderLock();
        mutationVersion++;
    }

    private void publishRequestCountUnderLock() {
        requireLock();
        long count = saturatedAdd(requests.size(), unknownEngineRequestCount);
        if (outstandingRequestCount != count) {
            outstandingRequestCount = count;
        }
        dispatchedRequestCount = saturatedAdd(
                requests.size() - activeIndex.size(), unknownEngineRequestCount);
    }

    public PrefillState(ReentrantLock lock, PrefillActiveIndex activeIndex,
                        Runnable capacityAvailable) {
        this(lock, activeIndex, System::currentTimeMillis, capacityAvailable);
    }

    public PrefillState(ReentrantLock lock,
                        PrefillActiveIndex activeIndex,
                        LongSupplier clock,
                        Runnable capacityAvailable) {
        this.lock = lock;
        this.activeIndex = activeIndex;
        this.clock = clock;
        this.capacityAvailable = capacityAvailable;
    }

    public RequestEntry putRequestUnderLock(String requestId, RequestEntry entry) {
        requireLock();
        return requests.put(requestId, entry);
    }

    public boolean removeRequestUnderLock(String requestId, RequestEntry entry) {
        requireLock();
        if (!requests.remove(requestId, entry)) {
            return false;
        }
        if (entry.queueMembership == QueueMembership.UNINDEXED) {
            committedWorkCapture = null;
        }
        return true;
    }

    /** Caller must hold the ownership lock. */
    public long mutationVersionUnderLock() {
        requireLock();
        return mutationVersion;
    }

    /** Advisory revision read for the lock-free load-view cache fast path. */
    public long mutationVersion() {
        return mutationVersion;
    }

    public boolean enqueueActiveUnderLock(ScheduledRequest item, long maxOutstandingRequests) {
        requireLock();
        if (requests.containsKey(item.requestId())
                || (maxOutstandingRequests > 0L && !canAcceptRequestUnderLock(maxOutstandingRequests))) {
            return false;
        }
        RequestEntry entry = new RequestEntry(item);
        putRequestUnderLock(item.requestId(), entry);
        try {
            activeIndex.add(item);
        } catch (RuntimeException | Error failure) {
            removeRequestUnderLock(item.requestId(), entry);
            throw failure;
        }
        recordMutationUnderLock();
        return true;
    }

    public boolean removeQueuedRequest(ScheduledRequest item) {
        requireLock();
        RequestEntry entry = requests.get(item.requestId());
        if (entry == null || !entry.activeIdentity(item)) {
            return false;
        }
        Reservation lease = entry.reservation;
        requireState(lease == null || lease.state == LeaseState.OPEN,
                "ACTIVE request owns a non-OPEN Prefill lease request_id="
                        + item.requestId());
        detachAdmissionIndexUnderLock(entry, item);
        // BATCH preparation owns any OPEN batch lease together with a
        // generation handoff. NON_BATCH item-owned single reservations use the
        // atomic counterpart below so retirement can never observe an orphan.
        removeRequestUnderLock(item.requestId(), entry);
        recordMutationUnderLock();
        return true;
    }

    /** Only failed admission needs priority provenance; reuse one summary per revision. */
    public AdmissionRejectReason admissionRejectReason(int priority, long requestLimit) {
        long[] counts;
        long occupiedRequests;
        lock.lock();
        try {
            if (requestCountsByPriority == null || requestCountsVersion != mutationVersion) {
                long[] updatedCounts = new long[PriorityNormalizer.MAX_PRIORITY + 1];
                updatedCounts[0] = unknownEngineRequestCount;
                for (RequestEntry entry : requests.values()) {
                    ScheduledRequest item = entry.activeItem != null ? entry.activeItem : entry.committedItem;
                    int occupant = item == null ? 0 : item.priority();
                    updatedCounts[occupant > 0 && occupant <= PriorityNormalizer.MAX_PRIORITY ? occupant : 0]++;
                }
                requestCountsByPriority = updatedCounts;
                requestCountsVersion = mutationVersion;
            }
            counts = requestCountsByPriority;
            occupiedRequests = outstandingRequestCount;
        } finally {
            lock.unlock();
        }
        long residual = requestLimit > 0 ? Math.max(0L, occupiedRequests + 1L - requestLimit) : 0L;
        long higher = 0L;
        long same = 0L;
        for (int occupant = 1; occupant <= PriorityNormalizer.MAX_PRIORITY; occupant++) {
            if (occupant < priority) { residual = Math.max(0L, residual - counts[occupant]); }
            else if (occupant == priority) { same += counts[occupant]; } else { higher += counts[occupant]; }
        }
        if (residual > higher + same && counts[0] > 0) {
            return AdmissionRejectReason.UNSPECIFIED;
        }
        if (residual > 0 && higher + same >= residual) {
            return higher > 0 ? AdmissionRejectReason.HIGHER_PRIORITY_AHEAD
                    : AdmissionRejectReason.SAME_PRIORITY_AHEAD;
        }
        return AdmissionRejectReason.RESOURCE_EXHAUSTED;
    }

    public boolean canPreemptQueuedRequest(int priority, long requestLimit) {
        return queuedPreemption.canPreemptQueuedRequest(priority, requestLimit);
    }

    public List<ScheduledRequest> queuedPreemptionVictimsUnderLock(int priority, long requestLimit) {
        return queuedPreemption.queuedPreemptionVictimsUnderLock(priority, requestLimit);
    }

    public ScheduledRequest oneQueuedPreemptionVictimUnderLock(int priority) {
        return queuedPreemption.oneQueuedPreemptionVictimUnderLock(priority);
    }

    public SingleReservation replaceQueuedRequests(List<ScheduledRequest> victims,
                                                         List<SingleReservation> reservations,
                                                         ScheduledRequest incoming) {
        return queuedPreemption.replaceQueuedRequests(victims, reservations, incoming);
    }

    public boolean removeQueuedSingleRequest(ScheduledRequest item, SingleReservation exactReservation) {
        return singlePrefill.removeQueuedSingleRequest(item, exactReservation);
    }

    /**
     * Detach one exact queue head for the stop callback without discarding its
     * canonical request owner. A failed callback therefore remains visible to
     * generation retirement, while a successful callback must explicitly
     * acknowledge the exact pending identity below.
     */
    public ScheduledRequest detachNextActiveForStopUnderLock() {
        requireLock();
        ScheduledRequest item = activeIndex.peek();
        if (item == null) {
            return null;
        }
        RequestEntry entry = requests.get(item.requestId());
        requireState(entry != null && entry.activeIdentity(item),
                "stopped queue head has no canonical ACTIVE owner request_id="
                        + item.requestId());
        Reservation lease = entry.reservation;
        requireState(lease == null || lease.state == LeaseState.OPEN,
                "stopped ACTIVE request owns a non-OPEN Prefill lease request_id="
                        + item.requestId());
        boolean removed = activeIndex.remove(item);
        requireState(removed,
                "stopped canonical ACTIVE request has no queue index request_id="
                        + item.requestId());
        // The queue-index removal is the PNR. Every fallible validation is
        // complete; the sole remaining commit is this private field store.
        entry.queueMembership = QueueMembership.STOP_DETACHED;
        recordMutationUnderLock();
        return item;
    }

    /** Remove only the exact stop-pending owner whose callback completed. */
    public boolean acknowledgeStopTerminalUnderLock(ScheduledRequest item) {
        requireLock();
        RequestEntry entry = requests.get(item.requestId());
        if (entry == null
                || entry.activeItem != item
                || entry.queueMembership != QueueMembership.STOP_DETACHED
                || activeIndex.contains(item)) {
            return false;
        }
        boolean removed = removeRequestUnderLock(item.requestId(), entry);
        if (removed) {
            recordMutationUnderLock();
        }
        return removed;
    }

    public ReservationResult<SingleReservation> reserveSingle(ScheduledRequest exactItem, long predictedMs) {
        return singlePrefill.reserveSingle(exactItem, predictedMs);
    }

    public CommittedHandoff commitSingleGroup(
            List<ScheduledRequest> items,
            List<SingleReservation> exactReservations,
            EndpointGenerationLifecycle.HandoffPermit generationHandoff) {
        return singlePrefill.commitSingleGroup(items, exactReservations, generationHandoff);
    }

    public ReservationResult<BatchReservation> reserveBatch(
            ScheduledRequest exactHead,
            long batchId,
            int maximum,
            EndpointGenerationLifecycle.HandoffPermit generationHandoff) {
        return batchPrefill.reserveBatch(exactHead, batchId, maximum, generationHandoff);
    }

    public int batchLeasesInUseUnderLock() {
        return batchPrefill.batchLeasesInUseUnderLock();
    }

    public DeliveryBlockReason.Availability batchAvailability(int maximum) {
        return batchPrefill.batchAvailability(maximum);
    }

    public int evictExpiredBatches(long ttlMs, java.util.function.Predicate<String> retainForSchedulerCleanup) {
        return batchPrefill.evictExpiredBatches(ttlMs, retainForSchedulerCleanup);
    }

    public boolean canDeliver(long maxDispatchLimit, int newReqNum) {
        return dispatchedRequestCount < maxDispatchLimit
                && newReqNum <= maxDispatchLimit - dispatchedRequestCount;
    }

    public long dispatchedRequestCount() {
        return dispatchedRequestCount;
    }

    public DeliveryBlockReason.Availability routeAvailability(long maximum) {
        return new DeliveryAvailability(this, maximum);
    }

    public List<ScheduledRequest> validateActiveGroup(List<ScheduledRequest> items) {
        requireState(!items.isEmpty(), "committed group requires members");
        for (ScheduledRequest item : items) {
            RequestEntry entry = requests.get(item.requestId());
            requireState(entry != null && entry.activeIdentity(item),
                    "group member is not canonical ACTIVE request_id="
                            + item.requestId());
            requireState(activeIndex.contains(item),
                    "canonical ACTIVE request has no queue index request_id="
                            + item.requestId());
        }
        return items;
    }

    public void removeValidatedActiveIndex(ScheduledRequest item) {
        boolean removed = activeIndex.remove(item);
        requireState(removed,
                "validated ACTIVE queue index disappeared request_id="
                        + item.requestId());
    }

    public void validateAdmissionIndexUnderLock(RequestEntry entry, ScheduledRequest item) {
        requireState(entry.queueMembership != QueueMembership.STOP_DETACHED,
                "stopped admission cannot commit request_id=" + item.requestId());
        requireState(activeIndex.contains(item) == (entry.queueMembership == QueueMembership.WAITING),
                "canonical admission and waiting index disagree request_id=" + item.requestId());
    }

    public void detachAdmissionIndexUnderLock(RequestEntry entry, ScheduledRequest item) {
        validateAdmissionIndexUnderLock(entry, item);
        if (entry.queueMembership == QueueMembership.WAITING) {
            removeValidatedActiveIndex(item);
        }
        // The caller removes this entry or commits it before releasing the lock.
        // Keep its admission origin until then: removing queued-only work must
        // not invalidate the immutable committed-work view.
    }

    /**
     * Total counterpart cleanup bound to one exact committed ScheduledRequest. A
     * reused request id or an ACTIVE item is a no-op.
     */
    public boolean removeCommittedRequest(ScheduledRequest exactItem) {
        boolean capacityReleased = false;
        lock.lock();
        try {
            RequestEntry entry = requests.get(exactItem.requestId());
            if (entry == null || entry.isActive()
                    || entry.committedItem != exactItem) {
                return false;
            }
            LiveBatchMembers liveMembers =
                    entry.batchProgress == null ? null : batchPrefill.liveBatchMembersUnderLock(entry.batchProgress);
            // Local cleanup does not prove that Engine omitted this member's work.
            capacityReleased = settleUnderLock(entry, TerminalObservation.external(entry.requestId), liveMembers, null);
            return true;
        } finally {
            lock.unlock();
            notifyCapacityAvailable(capacityReleased);
        }
    }

    /**
     * Reconcile ownership and publish its matching WorkerStatus before the
     * canonical queue lock is released. Load view readers therefore observe
     * either the previous pair or the fully reduced new pair.
     */
    public StatusReconciliation reconcileWorkerStatus(
            WorkerStatus.StatusObservation observation,
            ToLongFunction<List<ScheduledRequest>> repredictor,
            Runnable committedPublication,
            Runnable failedReduction) {
        return reconcileEngineStatus(
                observation.engine(),
                observation.finishedTasks(),
                repredictor,
                committedPublication,
                failedReduction);
    }

    public record HeartbeatReconciliation(List<WorkerStatusFact> schedulerFacts,
                                          boolean schedulingInputsChanged) {
        public HeartbeatReconciliation {
            schedulerFacts = List.copyOf(schedulerFacts);
        }
    }

    public HeartbeatReconciliation reconcileHeartbeat(WorkerStatus.StatusObservation observation) {
        List<WorkerStatusFact> facts = new ArrayList<>(
                observation.activeTasks().size());
        boolean capacityReleased = false;
        boolean schedulingInputsChanged = false;
        lock.lock();
        try {
            long nowMs = clock.getAsLong();
            WorkerStatus.EngineObservation engine = observation.engine();
            IdentityHashMap<RequestEntry, Phase> individualPhases =
                    new IdentityHashMap<>();
            IdentityHashMap<BatchProgress, Phase> batchPhases = new IdentityHashMap<>();
            long nextUnknown =
                    prepareActiveObservationsUnderLock(
                            engine.runningTaskList(),
                            Map.of(),
                            saturatedAdd(
                                    Math.max(0L, engine.waitingQueryLen()),
                                    Math.max(0L, engine.runningQueryLen())),
                            individualPhases,
                            batchPhases,
                            facts);
            schedulingInputsChanged = nextUnknown != unknownEngineRequestCount;
            for (var observed : individualPhases.entrySet()) {
                schedulingInputsChanged |= observed.getKey().singleRequestProgress.phase() != observed.getValue();
                observed.getKey().singleRequestProgress.updateProgressFromWorkerStatus(observed.getValue(), nowMs);
            }
            for (var observed : batchPhases.entrySet()) {
                schedulingInputsChanged |= observed.getKey().phase() != observed.getValue();
                observed.getKey().updateProgressFromWorkerStatus(observed.getValue(), nowMs);
            }
            capacityReleased = nextUnknown < unknownEngineRequestCount;
            unknownEngineRequestCount = nextUnknown;
            if (schedulingInputsChanged) {
                committedWorkCapture = null;
                recordMutationUnderLock();
            }
        } finally {
            lock.unlock();
            notifyCapacityAvailable(capacityReleased);
        }
        return new HeartbeatReconciliation(facts, schedulingInputsChanged);
    }

    /**
     * Materialize every externally observable status fact before the first
     * canonical entry is settled. The returned object is the exact outcome
     * later published by the endpoint; reduction only binds a callback failure.
     */
    private StatusReconciliation prepareStatusReconciliationUnderLock(
            Map<String, TerminalObservation> terminals,
            Map<String, WorkerStatus.TaskObservation> activeTasks,
            IdentityHashMap<BatchProgress, LiveBatchMembers> membersByBatch) {
        requireLock();
        List<WorkerStatusFact> schedulerFacts = new ArrayList<>(
                terminals.size() + activeTasks.size());
        for (TerminalObservation terminal : terminals.values()) {
            WorkerStatusFact fact = terminalFact(requests.get(terminal.requestId()), terminal);
            if (fact != null) {
                schedulerFacts.add(fact);
            }
        }
        for (WorkerStatus.TaskObservation task : activeTasks.values()) {
            if (terminals.containsKey(task.requestId())) {
                continue;
            }
            WorkerStatusFact fact = activeStatusFactUnderLock(task);
            if (fact != null) {
                schedulerFacts.add(fact);
            }
        }

        List<FinishedBatch> finishedBatches = new ArrayList<>(Math.min(terminals.size(), membersByBatch.size()));
        for (LiveBatchMembers liveMembers : membersByBatch.values()) {
            FinishedBatch finishedBatch = liveMembers.finishedBatchIfAllFinish(terminals);
            if (finishedBatch != null) {
                finishedBatches.add(finishedBatch);
            }
        }
        return new StatusReconciliation(schedulerFacts, finishedBatches, null);
    }

    /** Resolve activity only through the exact committed ledger identity. */
    private WorkerStatusFact activeStatusFactUnderLock(WorkerStatus.TaskObservation task) {
        requireLock();
        RequestEntry entry = requests.get(task.requestId());
        if (entry == null || entry.isActive() || !BatchPrefill.matchesObservedBatch(entry, task.batchId())
                || entry.committedItem == null || task.isPriorityCancelOverlayOnly()) {
            return null;
        }
        return WorkerStatusFact.active(entry.committedItem);
    }

    private long prepareActiveObservationsUnderLock(Map<String, WorkerStatus.TaskObservation> activeTasks,
                                                    Map<String, TerminalObservation> terminals,
                                                    long reportedActive,
                                                    IdentityHashMap<RequestEntry, Phase> individualPhases,
                                                    IdentityHashMap<BatchProgress, Phase> batchPhases,
                                                    List<WorkerStatusFact> activeFacts) {
        requireLock();
        Set<String> unknownDetailed = new HashSet<>();
        Set<String> knownObserved = new HashSet<>();
        for (WorkerStatus.TaskObservation task : activeTasks.values()) {
            if (terminals.containsKey(task.requestId())) {
                continue;
            }
            RequestEntry entry = requests.get(task.requestId());
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

    private StatusReconciliation reconcileEngineStatus(WorkerStatus.EngineObservation engine,
                                                       Map<String, WorkerStatus.TaskObservation> finishedTasks,
                                                       ToLongFunction<List<ScheduledRequest>> repredictor,
                                                       Runnable committedPublication,
                                                       Runnable failedReduction) {
        boolean capacityReleased = false;
        boolean canonicalMutationStarted = false;
        StatusReconciliation outcome = null;
        lock.lock();
        try {
            long nowMs = clock.getAsLong();
            IdentityHashMap<BatchProgress, LiveBatchMembers> membersByBatch = batchPrefill.liveBatchMembersUnderLock();
            Map<String, TerminalObservation> terminals = terminalObservationsUnderLock(
                    finishedTasks);
            Map<String, WorkerStatus.TaskObservation> activeTasks =
                    engine.runningTaskList();
            IdentityHashMap<RequestEntry, TerminalObservation> settlements =
                    new IdentityHashMap<>(terminals.size());
            Set<LiveBatchMembers> changedBatches = java.util.Collections.newSetFromMap(new IdentityHashMap<>());
            IdentityHashMap<BatchProgress, Phase> batchPhases = new IdentityHashMap<>();
            for (TerminalObservation terminal : terminals.values()) {
                RequestEntry entry = requests.get(terminal.requestId());
                if (entry == null
                        || settlements.put(entry, terminal) != null) {
                    throw new IllegalStateException(
                            "status terminal lost its exact owner request_id=" + terminal.requestId());
                }
                LiveBatchMembers liveMembers =
                        entry.batchProgress == null ? null : membersByBatch.get(entry.batchProgress);
                if (liveMembers != null) {
                    changedBatches.add(liveMembers);
                }
            }
            long reportedActive = saturatedAdd(
                    Math.max(0L, engine.waitingQueryLen()),
                    Math.max(0L, engine.runningQueryLen()));
            IdentityHashMap<RequestEntry, Phase> individualPhases =
                    new IdentityHashMap<>();
            long nextUnknown =
                    prepareActiveObservationsUnderLock(
                            activeTasks, terminals, reportedActive,
                            individualPhases, batchPhases, null);
            IdentityHashMap<BatchProgress, Long> predictions = new IdentityHashMap<>();
            batchPrefill.prepareBatchPredictionsUnderLock(changedBatches, terminals, batchPhases, repredictor,
                    predictions);
            outcome = prepareStatusReconciliationUnderLock(terminals, activeTasks, membersByBatch);

            // Everything below this boundary is assignment/removal against
            // exact prevalidated identities. Any invariant failure is captured
            // into the already materialized outcome and forces retirement.
            canonicalMutationStarted = true;
            committedWorkCapture = null;
            recordMutationUnderLock();
            for (Map.Entry<RequestEntry, TerminalObservation> settlement
                    : settlements.entrySet()) {
                RequestEntry entry = settlement.getKey();
                capacityReleased |= settleUnderLock(entry, settlement.getValue(),
                        entry.batchProgress == null ? null : membersByBatch.get(entry.batchProgress), null, nowMs);
            }
            individualPhases.forEach(
                    (entry, phase) -> entry.singleRequestProgress.updateProgressFromWorkerStatus(phase, nowMs));
            batchPhases.forEach((batch, phase) -> batch.updateProgressFromWorkerStatus(phase, nowMs));
            predictions.forEach((batch, prediction) -> batch.resetRemainingWork(prediction, nowMs));
            capacityReleased |= nextUnknown < unknownEngineRequestCount;
            unknownEngineRequestCount = nextUnknown;
            // Status reduction changes counts after invalidating the old work revision.
            publishRequestCountUnderLock();
            try {
                committedPublication.run();
            } catch (Throwable failure) {
                outcome = withPublicationFailure(outcome, failure);
                try {
                    failedReduction.run();
                } catch (Throwable failClosedFailure) {
                    // The first publication failure is the canonical outcome.
                }
            }
        } catch (RuntimeException | Error failure) {
            if (canonicalMutationStarted && outcome != null) {
                outcome = withPublicationFailure(outcome, failure);
                try {
                    failedReduction.run();
                } catch (Throwable failClosedFailure) {
                    // The first reduction failure is the canonical outcome.
                }
            } else {
                try {
                    failedReduction.run();
                } catch (Throwable failClosedFailure) {
                    failure.addSuppressed(failClosedFailure);
                }
                throw failure;
            }
        } finally {
            lock.unlock();
            notifyCapacityAvailable(capacityReleased);
        }
        return outcome;
    }

    private static StatusReconciliation withPublicationFailure(StatusReconciliation outcome, Throwable failure) {
        return outcome.publicationFailure() != null
                ? outcome
                : new StatusReconciliation(outcome.schedulerFacts(), outcome.finishedBatches(), failure);
    }

    /**
     * End every owner for this retired Prefill generation in one queue-lock
     * transaction. All callback facts and completion DTOs are constructed
     * before the canonical table, queue index, and capacity counters are
     * cleared. Once clearing starts, the remaining operations are allocation-
     * free field updates; no failure can leave a partially retired registry.
     *
     * <p>Ordinarily {@link PrefillQueue#stopAndAwait()} has already reduced every
     * ACTIVE item. Including a defensively remaining ACTIVE identity here makes
     * endpoint close total if that earlier invariant check failed.</p>
     */
    public Retirement retireGenerationOwnership() {
        List<ScheduledRequest> ownedItems = new ArrayList<>();
        List<FinishedBatch> finishedBatches = new ArrayList<>();
        List<EndpointGenerationLifecycle.HandoffPermit> orphanedHandoffs =
                new ArrayList<>();
        Set<Reservation> leases = java.util.Collections.newSetFromMap(
                new IdentityHashMap<>());
        Set<BatchProgress> batches = java.util.Collections.newSetFromMap(new IdentityHashMap<>());
        Throwable invariantFailure = null;
        Retirement plannedRetirement;
        lock.lock();
        try {
            Set<ScheduledRequest> canonicalActive = java.util.Collections.newSetFromMap(
                    new IdentityHashMap<>());
            for (Map.Entry<String, RequestEntry> canonical : requests.entrySet()) {
                RequestEntry entry = canonical.getValue();
                if (!Objects.equals(canonical.getKey(), entry.requestId)) {
                    invariantFailure = appendRetirementInvariant(
                            invariantFailure,
                            "request table key does not match request entry");
                }
                ScheduledRequest item = entry.isActive() ? entry.activeItem : entry.committedItem;
                if (item == null) {
                    invariantFailure = appendRetirementInvariant(invariantFailure,
                            "retirement owner has no ScheduledRequest request_id=" + entry.requestId);
                } else {
                    ownedItems.add(item);
                    if (entry.isActive() && entry.queueMembership == QueueMembership.WAITING) {
                        canonicalActive.add(item);
                    } else if (activeIndex.contains(item)) {
                        invariantFailure = appendRetirementInvariant(invariantFailure,
                                "unindexed owner remains in waiting index request_id=" + entry.requestId);
                    }
                }

                if (entry.reservation != null) {
                    leases.add(entry.reservation);
                }
                if (entry.batchProgress != null) {
                    batches.add(entry.batchProgress);
                    leases.add(entry.batchProgress.lease());
                }
            }

            if (activeIndex.size() != canonicalActive.size()) {
                invariantFailure = appendRetirementInvariant(
                        invariantFailure,
                        "ACTIVE index size does not match canonical ACTIVE owners");
            }
            for (ScheduledRequest indexed : activeIndex) {
                if (!canonicalActive.contains(indexed)) {
                    invariantFailure = appendRetirementInvariant(
                            invariantFailure,
                            "ACTIVE index contains a non-canonical item request_id="
                                    + indexed.requestId());
                }
            }

            int observedBatchLeases = 0;
            for (Reservation lease : leases) {
                if (lease instanceof SingleReservation) {
                    if (lease.state != LeaseState.OPEN
                            && lease.state != LeaseState.OWNED) {
                        invariantFailure = appendRetirementInvariant(
                                invariantFailure,
                                "canonical retirement owner has an invalid single-reservation state");
                    }
                } else if (lease instanceof BatchReservation) {
                    observedBatchLeases++;
                    BatchReservation batch = (BatchReservation) lease;
                    if (batch.state == LeaseState.OPEN) {
                        if (batch.generationHandoff == null) {
                            invariantFailure = appendRetirementInvariant(
                                    invariantFailure,
                                    "OPEN Prefill batch lease lost its generation handoff");
                        } else {
                            orphanedHandoffs.add(batch.generationHandoff);
                            invariantFailure = appendRetirementInvariant(
                                    invariantFailure,
                                    "retirement reached an OPEN Prefill batch lease");
                        }
                    } else if (batch.state != LeaseState.OWNED
                            || batch.generationHandoff != null) {
                        invariantFailure = appendRetirementInvariant(
                                invariantFailure,
                                "canonical retirement owner has an invalid batch lease state");
                    }
                } else {
                    invariantFailure = appendRetirementInvariant(
                            invariantFailure,
                            "retirement found an unknown Prefill lease type");
                }
            }
            if (batchPrefill.batchLeasesInUseUnderLock() != observedBatchLeases) {
                invariantFailure = appendRetirementInvariant(
                        invariantFailure,
                        "batch capacity does not match canonical retirement leases");
            }

            for (BatchProgress batch : batches) {
                finishedBatches.add(batch.finishedBatchOnRetirement());
            }
            ownedItems.sort(Comparator.comparingLong(ScheduledRequest::enqueueSeq)
                    .thenComparing(ScheduledRequest::requestId));
            finishedBatches.sort(Comparator.comparingLong(FinishedBatch::batchId));
            plannedRetirement = new Retirement(ownedItems, finishedBatches, invariantFailure);

            // Canonical retirement commit. Everything which may allocate or
            // validate has completed above this line.
            for (RequestEntry entry : requests.values()) {
                entry.activeItem = null;
                entry.committedItem = null;
                entry.reservation = null;
                entry.batchProgress = null;
                entry.singleRequestProgress = null;
                entry.queueMembership = QueueMembership.UNINDEXED;
            }
            for (Reservation lease : leases) {
                if (lease instanceof BatchReservation batch) {
                    batch.generationHandoff = null;
                }
                lease.state = LeaseState.CLOSED;
            }
            requests.clear();
            activeIndex.clear();
            batchPrefill.resetBatchLeasesInUseUnderLock();
            unknownEngineRequestCount = 0L;
            committedWorkCapture = null;
            recordMutationUnderLock();
        } finally {
            lock.unlock();
        }

        for (EndpointGenerationLifecycle.HandoffPermit handoff
                : orphanedHandoffs) {
            try {
                handoff.close();
            } catch (Throwable ignoredHandoffFailure) {
                // The OPEN-state invariant above is the fixed primary failure.
                // Canonical retirement has committed, so aggregation must not
                // allocate or prevent later exact handoffs from being closed.
            }
        }
        notifyCapacityAvailable(!leases.isEmpty());
        return plannedRetirement;
    }

    private static Throwable appendRetirementInvariant(
            Throwable first,
            String message) {
        IllegalStateException next = new IllegalStateException(message);
        if (first == null) {
            return next;
        }
        first.addSuppressed(next);
        return first;
    }

    public int evictExpiredIndividuals(long ttlMs, java.util.function.Predicate<String> retainForSchedulerCleanup) {
        int evicted = 0;
        boolean capacityReleased = false;
        lock.lock();
        try {
            long nowMs = clock.getAsLong();
            List<RequestEntry> candidates = new ArrayList<>();
            for (RequestEntry entry : requests.values()) {
                if (entry.batchProgress == null && !entry.isActive()
                        && nowMs - entry.singleRequestProgress.lastObservedAtMs() >= Math.max(0L, ttlMs)
                        && !retainForSchedulerCleanup.test(entry.requestId)) {
                    candidates.add(entry);
                }
            }
            for (RequestEntry entry : candidates) {
                capacityReleased |= settleUnderLock(
                        entry,
                        TerminalObservation.external(entry.requestId),
                        null,
                        new ArrayList<>(0));
            }
            evicted = candidates.size();
        } finally {
            lock.unlock();
            notifyCapacityAvailable(capacityReleased);
        }
        return evicted;
    }

    public Stats stats() {
        lock.lock();
        try {
            int locallyOwned = 0;
            int workerStatusUnconfirmed = 0;
            int individual = 0;
            long maxAgeMs = 0L;
            Set<BatchProgress> batches = java.util.Collections.newSetFromMap(new IdentityHashMap<>());
            long nowMs = clock.getAsLong();
            for (RequestEntry entry : requests.values()) {
                if (entry.isActive()) {
                    continue;
                }
                locallyOwned++;
                Phase phase =
                        entry.batchProgress == null ? entry.singleRequestProgress.phase() : entry.batchProgress.phase();
                if (phase == Phase.COMMITTED && (entry.batchProgress != null
                        || entry.reservation != null && entry.reservation.state == LeaseState.OWNED)) {
                    workerStatusUnconfirmed++;
                }
                if (entry.batchProgress == null) {
                    individual++;
                    maxAgeMs = Math.max(maxAgeMs, Math.max(0L, nowMs - entry.singleRequestProgress.lastObservedAtMs()));
                } else {
                    batches.add(entry.batchProgress);
                }
            }
            for (BatchProgress batch : batches) {
                maxAgeMs = Math.max(maxAgeMs, Math.max(0L, nowMs - batch.lastObservedAtMs()));
            }
            return new Stats(
                    saturatedAdd(requests.size(), unknownEngineRequestCount),
                    locallyOwned,
                    workerStatusUnconfirmed,
                    individual,
                    batches.size(),
                    maxAgeMs);
        } finally {
            lock.unlock();
        }
    }

    /** Advisory selection check; a positive result does not reserve capacity. */
    public boolean canAcceptRequest(long requestLimit) {
        return outstandingRequestCount < requestLimit;
    }

    private boolean canAcceptRequestUnderLock(long maxOutstandingRequests) {
        requireLock();
        return requests.size() < maxOutstandingRequests
                && unknownEngineRequestCount < maxOutstandingRequests - requests.size();
    }

    public long observedRequestCount() {
        lock.lock();
        try {
            return saturatedAdd(
                    requests.size(),
                    unknownEngineRequestCount);
        } finally {
            lock.unlock();
        }
    }

    public Snapshot snapshotUnderLock() {
        requireLock();
        long nowMs = clock.getAsLong();
        return new Snapshot(
                nowMs,
                activeIndex.capture(),
                captureWorkUnderLock(nowMs));
    }

    public WorkView committedSnapshot() {
        WorkCapture capture;
        lock.lock();
        try {
            capture = captureCurrentWorkUnderLock(clock.getAsLong());
        } finally {
            lock.unlock();
        }
        return capture.materialize();
    }

    private WorkCapture captureWorkUnderLock(long nowMs) {
        requireLock();
        if (committedWorkCapture == null || committedWorkCapture.capturedAtMs > nowMs) {
            committedWorkCapture = captureCurrentWorkUnderLock(nowMs);
        }
        return committedWorkCapture;
    }

    public WorkView capturePrecedingWorkUnderLock(List<ScheduledRequest> members, long nowMs) {
        requireLock();
        Set<RequestEntry> excluded = java.util.Collections.newSetFromMap(new IdentityHashMap<>());
        for (ScheduledRequest member : members) {
            excluded.add(requests.get(member.requestId()));
        }
        return captureCurrentWorkUnderLock(nowMs, excluded).materialize();
    }

    private WorkCapture captureCurrentWorkUnderLock(long nowMs) {
        return captureCurrentWorkUnderLock(nowMs, Set.of());
    }

    private WorkCapture captureCurrentWorkUnderLock(long nowMs, Set<RequestEntry> excluded) {
        requireLock();
        List<WorkView.RequestWork> individual = new ArrayList<>();
        IdentityHashMap<BatchProgress, List<String>> batchMembers = new IdentityHashMap<>();
        for (RequestEntry entry : requests.values()) {
            if (excluded.contains(entry)) {
                continue;
            }
            if (entry.isActive()) {
                continue;
            }
            if (entry.batchProgress == null) {
                individual.add(new WorkView.RequestWork(entry.requestId, entry.singleRequestProgress.phase(),
                        entry.singleRequestProgress.remainingWorkMsAt(nowMs)));
            } else {
                batchMembers.computeIfAbsent(entry.batchProgress, ignored -> new ArrayList<>()).add(entry.requestId);
            }
        }
        List<WorkView.BatchWork> batches =
                new ArrayList<>(batchMembers.size());
        for (Map.Entry<BatchProgress, List<String>> observed : batchMembers.entrySet()) {
            BatchProgress batch = observed.getKey();
            batches.add(new WorkView.BatchWork(batch.batchId(), observed.getValue(), batch.phase(),
                    OptionalLong.of(batch.remainingWorkMsAt(nowMs))));
        }
        return new WorkCapture(
                nowMs,
                individual,
                batches,
                unknownEngineRequestCount);
    }

    public boolean settleUnderLock(RequestEntry entry,
                            TerminalObservation terminal,
                            LiveBatchMembers liveMembers,
                            List<FinishedBatch> finishedBatches) {
        return settleUnderLock(entry, terminal, liveMembers, finishedBatches, clock.getAsLong());
    }

    public boolean settleUnderLock(RequestEntry entry,
                            TerminalObservation terminal,
                            LiveBatchMembers liveMembers,
                            List<FinishedBatch> finishedBatches,
                            long nowMs) {
        requireLock();
        BatchProgress batch = entry.batchProgress;
        boolean lastBatchMember = false;
        Reservation lease;
        if (batch != null) {
            if (liveMembers == null || liveMembers.batch() != batch || !liveMembers.members().contains(entry)) {
                throw new IllegalStateException("missing live batch members for request_id=" + entry.requestId);
            }
            lastBatchMember = liveMembers.members().size() == 1;
            lease = lastBatchMember ? batch.lease() : null;
        } else {
            lease = entry.reservation;
        }
        if (lease != null && lease.state != LeaseState.OWNED) {
            throw new IllegalStateException(
                    "terminal request owns a non-committed Prefill lease"
                            + " request_id=" + entry.requestId);
        }
        if (batch != null) {
            batch.updateProgressFromFinishedTask(terminal.workerObserved(), terminal.errorCode(),
                    terminal.executionTimeMs(), nowMs);
            liveMembers.remove(entry);
        }
        if (!removeRequestUnderLock(entry.requestId, entry)) {
            throw new IllegalStateException(
                    "terminal request is not canonical request_id="
                            + entry.requestId);
        }
        committedWorkCapture = null;
        recordMutationUnderLock();
        if (lease != null) {
            closeOwnedLeaseUnderLock(lease);
        }
        if (lastBatchMember && finishedBatches != null) {
            finishedBatches.add(batch.finishedBatch());
        }
        return true;
    }

    private Map<String, TerminalObservation> terminalObservationsUnderLock(
            Map<String, WorkerStatus.TaskObservation> finishedTasks) {
        requireLock();
        Map<String, TerminalObservation> terminals = new HashMap<>();
        for (WorkerStatus.TaskObservation task : finishedTasks.values()) {
            RequestEntry entry = requests.get(task.requestId());
            if (entry == null || !BatchPrefill.matchesObservedBatch(entry, task.batchId())) {
                continue;
            }
            TerminalObservation terminal = TerminalObservation.from(task, entry.batchProgress != null);
            terminals.merge(
                    task.requestId(), terminal, TerminalObservation::merge);
        }
        return terminals;
    }

    private static WorkerStatusFact terminalFact(RequestEntry entry, TerminalObservation terminal) {
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

    /**
     * Roll back only a provisional lease. Once committed, the request table is
     * the sole owner and only its terminal reducer may release the capacity.
     */
    void releaseOpenLease(Reservation lease) {
        if (lock.isHeldByCurrentThread()) {
            throw new IllegalStateException(
                    "Prefill lease rollback cannot run under queueLock");
        }
        EndpointGenerationLifecycle.HandoffPermit generationHandoff = null;
        boolean capacityReleased = false;
        lock.lock();
        try {
            if (lease.state == LeaseState.OPEN) {
                generationHandoff = closeOpenLeaseUnderLock(lease);
                capacityReleased = true;
            }
        } finally {
            lock.unlock();
        }
        if (!capacityReleased) {
            return;
        }
        Throwable failure = null;
        try {
            if (generationHandoff != null) {
                generationHandoff.close();
            }
        } catch (Throwable handoffFailure) {
            failure = handoffFailure;
        } finally {
            notifyCapacityAvailable(true);
        }
        rethrowCleanupFailure(failure);
    }

    public void notifyCapacityAvailable(boolean capacityReleased) {
        if (!capacityReleased) {
            return;
        }
        try {
            capacityAvailable.run();
        } catch (Throwable notificationFailure) {
            try {
                org.flexlb.util.Logger.error(
                        "Prefill capacity notification failed",
                        notificationFailure);
            } catch (Throwable ignoredLoggingFailure) {
                // Capacity ownership is already settled; diagnostics cannot
                // make the caller lose its prebuilt terminal facts.
            }
        }
    }

    /** OPEN rollback closes quota and any batch-owned generation handoff. */
    public EndpointGenerationLifecycle.HandoffPermit
            closeOpenLeaseUnderLock(Reservation lease) {
        requireLock();
        if (lease.state != LeaseState.OPEN) {
            throw new IllegalStateException(
                    "Prefill OPEN rollback lost its exact lease");
        }
        RequestEntry owner = openLeaseOwnerUnderLock(lease);
        if (owner != null
                && !owner.isActive()
                && owner.queueMembership != QueueMembership.STOP_DETACHED) {
            throw new IllegalStateException(
                    "OPEN Prefill lease has a non-ACTIVE canonical owner");
        }
        EndpointGenerationLifecycle.HandoffPermit generationHandoff = null;
        if (lease instanceof BatchReservation batch) {
            generationHandoff = batch.generationHandoff;
            if (generationHandoff == null) {
                throw new IllegalStateException(
                        "Prefill batch rollback lost its generation handoff");
            }
            batch.generationHandoff = null;
        }
        lease.state = LeaseState.CLOSED;
        releaseBatchSlotUnderLock(lease);
        if (owner != null) {
            owner.reservation = null;
        }
        recordMutationUnderLock();
        return generationHandoff;
    }

    /** Close exact ownership and return a batch slot when this is the last member. */
    private void closeOwnedLeaseUnderLock(Reservation lease) {
        requireLock();
        if (lease.state != LeaseState.OWNED
                || lease instanceof BatchReservation batch
                && batch.generationHandoff != null) {
            throw new IllegalStateException(
                    "Prefill OWNED terminal still owns an admission handoff");
        }
        lease.state = LeaseState.CLOSED;
        releaseBatchSlotUnderLock(lease);
    }

    private void releaseBatchSlotUnderLock(Reservation lease) {
        requireLock();
        if (lease instanceof BatchReservation) {
            batchPrefill.releaseBatchSlotUnderLock();
        } else if (!(lease instanceof SingleReservation)) {
            throw new IllegalStateException("unknown Prefill reservation type");
        }
    }

    private static void rethrowCleanupFailure(Throwable failure) {
        if (failure instanceof RuntimeException runtimeFailure) {
            throw runtimeFailure;
        }
        if (failure instanceof Error error) {
            throw error;
        }
        if (failure != null) {
            throw new IllegalStateException(
                    "Prefill exact capacity cleanup failed", failure);
        }
    }

    private RequestEntry openLeaseOwnerUnderLock(Reservation lease) {
        requireLock();
        RequestEntry originalOwner = lease.originalOwner;
        RequestEntry current = requests.get(originalOwner.requestId);
        if (current == originalOwner) {
            if (current.reservation != lease) {
                throw new IllegalStateException(
                        "canonical Prefill lease owner lost its exact reservation"
                                + " request_id=" + originalOwner.requestId);
            }
            return current;
        }
        if (current != null && current.reservation == lease) {
            throw new IllegalStateException(
                    "replacement request cannot own an earlier Prefill lease"
                            + " request_id=" + originalOwner.requestId);
        }
        return null;
    }

    public void requireLock() {
        requireState(lock.isHeldByCurrentThread(),
                "Prefill ownership requires queueLock");
    }

    public static void requireState(boolean condition, String message) {
        if (!condition) {
            throw new IllegalStateException(message);
        }
    }

    public static long saturatedAdd(long left, long right) {
        return left > Long.MAX_VALUE - right ? Long.MAX_VALUE : left + right;
    }
}
