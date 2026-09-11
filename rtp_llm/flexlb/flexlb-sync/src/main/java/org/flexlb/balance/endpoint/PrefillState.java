package org.flexlb.balance.endpoint;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.prediction.PrefillBatchFeatures;
import org.flexlb.balance.projection.WorkSnapshot;
import org.flexlb.balance.projection.WorkSnapshot.Phase;
import org.flexlb.balance.scheduler.ScheduledRequest;
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

    public enum CapacityStatus {
        ACQUIRED,
        CAPACITY_FULL,
        REQUEST_NOT_ACTIVE,
        REQUEST_ALREADY_RESERVED,
        BATCH_ID_ALREADY_RESERVED,
        ENDPOINT_RETIRED
    }

    public record ReservationResult<R extends AutoCloseable>(
            CapacityStatus status,
            R reservation) {
        public ReservationResult {
            Objects.requireNonNull(status, "status");
            if ((status == CapacityStatus.ACQUIRED) != (reservation != null)) {
                throw new IllegalArgumentException(
                        "only ACQUIRED may carry a reservation");
            }
        }
    }

    public record WorkerStatusFact(
            ScheduledRequest item,
            Kind kind,
            long errorCode) {
        public WorkerStatusFact {
            Objects.requireNonNull(item, "item");
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

    public record StatusReconciliation(
            List<WorkerStatusFact> schedulerFacts,
            List<BatchCompletion> batchCompletions,
            Throwable publicationFailure) {
        public StatusReconciliation {
            schedulerFacts = List.copyOf(schedulerFacts);
            batchCompletions = List.copyOf(batchCompletions);
        }
    }

    public record BatchCompletion(
            long batchId,
            PrefillBatchFeatures originalFeatures,
            long predictedWorkMs,
            long actualWorkMs,
            boolean successfulCompletion,
            boolean learningEligible) {
        public BatchCompletion {
            Objects.requireNonNull(originalFeatures, "originalFeatures");
        }
    }

    public record Retirement(
            List<ScheduledRequest> ownedItems,
            List<BatchCompletion> batchCompletions,
            Throwable invariantFailure) {
        public Retirement {
            ownedItems = List.copyOf(ownedItems);
            batchCompletions = List.copyOf(batchCompletions);
        }
    }

    public record Stats(
            int locallyOwnedRequests,
            int individuallyOwnedRequests,
            int batchCount,
            long maxObservedAgeMs) {
        public Stats {
            if (locallyOwnedRequests < 0 || individuallyOwnedRequests < 0
                    || batchCount < 0 || maxObservedAgeMs < 0L) {
                throw new IllegalArgumentException(
                        "Prefill state stats must be non-negative");
            }
        }
    }

    enum LeaseState {
        OPEN,
        OWNED,
        CLOSED
    }

    /** One-shot capability returned when an OPEN admission commits. */
    public static final class CommittedHandoff implements AutoCloseable {
        private EndpointGenerationLifecycle.HandoffPermit generationHandoff;
        private final WorkSnapshot precedingWork;

        private CommittedHandoff(
                EndpointGenerationLifecycle.HandoffPermit generationHandoff,
                WorkSnapshot precedingWork) {
            this.generationHandoff = generationHandoff;
            this.precedingWork = Objects.requireNonNull(precedingWork, "precedingWork");
        }

        /** Other reserved work at the same ownership boundary that committed this admission. */
        public WorkSnapshot precedingWork() {
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

    /** Exact request or batch ownership passed through admission and callback. */
    public abstract class Reservation implements AutoCloseable {
        /* guarded by PrefillState.lock */ LeaseState state =
                LeaseState.OPEN;
        private final RequestEntry originalOwner;

        private Reservation(RequestEntry originalOwner) {
            this.originalOwner = Objects.requireNonNull(
                    originalOwner, "originalOwner");
        }

        /** Roll back an OPEN lease. Committed capacity stays Registry-owned. */
        @Override
        public final void close() {
            releaseOpenLease(this);
        }
    }

    public final class RouteReservation extends Reservation {
        private final PrefillState owner = PrefillState.this;
        private final long requestId;
        /* guarded by PrefillState.lock until the reservation commits */
        private long predictedWorkMs;

        private RouteReservation(RequestEntry originalOwner,
                                 long requestId,
                                 long predictedWorkMs) {
            super(originalOwner);
            this.requestId = requestId;
            this.predictedWorkMs = boundedPrediction(predictedWorkMs);
        }

        /** Bind the fresh delivery prediction to this exact OPEN reservation. */
        public void updatePrediction(
                ScheduledRequest exactItem,
                long predictedMs) {
            lock.lock();
            try {
                RequestEntry entry = requests.get(requestId);
                if (state != LeaseState.OPEN
                        || entry == null
                        || !entry.activeIdentity(exactItem)
                        || entry.reservation != this) {
                    throw new IllegalStateException(
                            "route prediction no longer owns ACTIVE request_id="
                                    + requestId);
                }
                predictedWorkMs = boundedPrediction(predictedMs);
                if (entry.queueMembership == QueueMembership.UNINDEXED) {
                    committedWorkCapture = null;
                }
                recordMutationUnderLock();
            } finally {
                lock.unlock();
            }
        }

    }

    public final class BatchReservation extends Reservation {
        private final long headRequestId;
        private final long batchId;

        private BatchReservation(RequestEntry originalOwner,
                                 long headRequestId,
                                 long batchId,
                                 EndpointGenerationLifecycle.HandoffPermit generationHandoff) {
            super(originalOwner);
            this.headRequestId = headRequestId;
            this.batchId = batchId;
            this.generationHandoff = Objects.requireNonNull(
                    generationHandoff, "generationHandoff");
        }

        /* guarded by PrefillState.lock; non-null only while OPEN */
        private EndpointGenerationLifecycle.HandoffPermit generationHandoff;

        public long batchId() {
            return batchId;
        }

        /** Atomically commit this exact batch lease. */
        public CommittedHandoff commit(
                List<ScheduledRequest> items,
                long predictedMs) {
            lock.lock();
            try {
                return commitBatchUnderLock(this, items, predictedMs);
            } finally {
                lock.unlock();
            }
        }
    }

    /** Shared execution estimate referenced by every live member of one batch. */
    private static final class BatchWork {
        private final BatchReservation lease;
        private final long originalPredictionMs;
        private final PrefillBatchFeatures originalFeatures;
        private long remainingWorkMs;
        private Phase servicePhase = Phase.COMMITTED;
        /** A later queued observation cannot make started batch work repackable. */
        private boolean executionStarted;
        private long phaseBaseMs;
        private long lastObservedAtMs;
        private long maxExecutionTimeMs;
        private boolean successfulCompletion;
        private boolean learningEligible = true;

        private BatchWork(BatchReservation lease,
                          long predictedWorkMs,
                          PrefillBatchFeatures originalFeatures,
                          long nowMs) {
            long batchId = lease.batchId;
            if (batchId < 0L) {
                throw new IllegalArgumentException("batchId must be non-negative");
            }
            if (predictedWorkMs < 0L) {
                throw new IllegalArgumentException(
                        "predicted batch work must be non-negative");
            }
            this.lease = lease;
            this.originalPredictionMs = predictedWorkMs;
            this.originalFeatures = originalFeatures;
            this.remainingWorkMs = predictedWorkMs;
            this.phaseBaseMs = nowMs;
            this.lastObservedAtMs = nowMs;
        }

        private long remainingAt(long nowMs) {
            if (servicePhase != Phase.ENGINE_RUNNING) {
                return remainingWorkMs;
            }
            return Math.max(0L, remainingWorkMs - Math.max(0L, nowMs - phaseBaseMs));
        }

        private void observeTerminal(TerminalObservation terminal, long nowMs) {
            touch(nowMs);
            executionStarted |= terminal.workerObserved
                    && (terminal.errorCode == 0L || terminal.executionTimeMs > 0L);
            if (terminal.executionTimeMs >= 0L) {
                maxExecutionTimeMs = Math.max(
                        maxExecutionTimeMs, terminal.executionTimeMs);
            }
            successfulCompletion |= terminal.workerObserved
                    && terminal.errorCode == 0L;
            learningEligible &= terminal.workerObserved
                    && terminal.errorCode == 0L;
        }

        private void touch(long nowMs) {
            lastObservedAtMs = Math.max(lastObservedAtMs, nowMs);
        }

        private void observePhase(Phase phase, long nowMs) {
            remainingWorkMs = remainingAt(nowMs);
            phaseBaseMs = Math.max(phaseBaseMs, nowMs);
            touch(nowMs);
            executionStarted |= phase == Phase.ENGINE_RUNNING;
            servicePhase = phase;
        }

        private BatchCompletion completion() {
            return new BatchCompletion(
                    lease.batchId,
                    originalFeatures,
                    originalPredictionMs,
                    maxExecutionTimeMs,
                    successfulCompletion,
                    learningEligible);
        }

        /** Retirement is external termination and must never train prediction. */
        private BatchCompletion retirementCompletion() {
            return new BatchCompletion(
                    lease.batchId,
                    originalFeatures,
                    originalPredictionMs,
                    maxExecutionTimeMs,
                    successfulCompletion,
                    false);
        }
    }

    /** One transaction-local view derived by a single scan of the request table. */
    private static final class BatchReduction {
        private final BatchWork batch;
        private final Set<RequestEntry> members =
                java.util.Collections.newSetFromMap(new IdentityHashMap<>());

        private BatchReduction(BatchWork batch) {
            this.batch = batch;
        }

        private void add(RequestEntry entry) {
            boolean added = members.add(entry);
            requireState(added,
                    "duplicate batch reduction request_id=" + entry.requestId);
        }

        private void remove(RequestEntry entry) {
            if (!members.remove(entry)) {
                throw new IllegalStateException(
                        "batch reduction lost request_id=" + entry.requestId);
            }
        }

        private boolean isEmpty() {
            return members.isEmpty();
        }

        /** Simulate terminal metrics without mutating canonical batch work. */
        private BatchCompletion projectedCompletion(
                Map<Long, TerminalObservation> terminals) {
            long maxExecutionTimeMs = batch.maxExecutionTimeMs;
            boolean successfulCompletion = batch.successfulCompletion;
            boolean learningEligible = batch.learningEligible;
            for (RequestEntry member : members) {
                TerminalObservation terminal = terminals.get(member.requestId);
                if (terminal == null) {
                    return null;
                }
                if (terminal.executionTimeMs >= 0L) {
                    maxExecutionTimeMs = Math.max(
                            maxExecutionTimeMs, terminal.executionTimeMs);
                }
                successfulCompletion |= terminal.workerObserved
                        && terminal.errorCode == 0L;
                learningEligible &= terminal.workerObserved
                        && terminal.errorCode == 0L;
            }
            return new BatchCompletion(
                    batch.lease.batchId,
                    batch.originalFeatures,
                    batch.originalPredictionMs,
                    maxExecutionTimeMs,
                    successfulCompletion,
                    learningEligible);
        }
    }

    /** Waiting-index membership is independent of admission and Engine ownership. */
    private enum QueueMembership { UNINDEXED, WAITING, STOP_DETACHED }

    /** The sole mutable request lifecycle record. Guarded by {@link #lock}. */
    private static final class RequestEntry {
        private final long requestId;
        private QueueMembership queueMembership;
        private ScheduledRequest activeItem;
        private Phase individualPhase;
        private long remainingWorkMs;
        private long phaseBaseMs;
        private long lastObservedAtMs;
        private BatchWork batchWork;
        /** Canonical callback identity after admission commits. */
        private ScheduledRequest committedItem;
        private Reservation reservation;
        private RequestEntry(ScheduledRequest item) {
            this(item, QueueMembership.WAITING);
        }

        private RequestEntry(ScheduledRequest item, QueueMembership queueMembership) {
            this.requestId = item.requestId();
            this.queueMembership = queueMembership;
            this.activeItem = item;
        }

        private boolean isActive() {
            return activeItem != null;
        }

        private boolean activeIdentity(ScheduledRequest item) {
            return activeItem != null
                    && activeItem == item
                    && queueMembership != QueueMembership.STOP_DETACHED;
        }

        private void commitIndividual(RouteReservation lease, long nowMs) {
            if (!isActive() || reservation != lease) {
                throw new IllegalStateException(
                        "request is not an ACTIVE route request_id=" + requestId);
            }
            remainingWorkMs = lease.predictedWorkMs;
            phaseBaseMs = nowMs;
            lastObservedAtMs = nowMs;
            committedItem = activeItem;
            activeItem = null;
            queueMembership = QueueMembership.UNINDEXED;
            individualPhase = Phase.COMMITTED;
        }

        private void commitBatch(BatchWork work) {
            if (!isActive()) {
                throw new IllegalStateException(
                        "request is not an ACTIVE batch member request_id=" + requestId);
            }
            batchWork = work;
            committedItem = activeItem;
            reservation = null;
            activeItem = null;
            queueMembership = QueueMembership.UNINDEXED;
        }

        private void observeIndividualPhase(Phase next, long nowMs) {
            if (next != Phase.ENGINE_QUEUED && next != Phase.ENGINE_RUNNING) {
                throw new IllegalArgumentException("invalid Engine phase " + next);
            }
            if (batchWork != null || individualPhase == null) {
                throw new IllegalStateException(
                        "request is not individual request_id=" + requestId);
            }
            if (individualPhase == Phase.ENGINE_RUNNING) {
                remainingWorkMs = Math.max(
                        0L, remainingWorkMs - Math.max(0L, nowMs - phaseBaseMs));
            }
            phaseBaseMs = Math.max(phaseBaseMs, nowMs);
            lastObservedAtMs = Math.max(lastObservedAtMs, nowMs);
            individualPhase = next;
        }
    }

    /** Queue and committed work captured at one ownership linearization point. */
    public record Snapshot(long capturedAtMs,
                           List<ScheduledRequest> activeItems,
                           WorkCapture work) {
        public Snapshot {
            activeItems = List.copyOf(activeItems);
            Objects.requireNonNull(
                    work, "missing committed work snapshot");
        }
    }

    /** Immutable leaves copied under the ownership lock; projection runs outside it. */
    public static final class WorkCapture {
        private final long capturedAtMs;
        private final List<WorkSnapshot.RequestWork> requests;
        private final List<WorkSnapshot.BatchWork> batches;
        private final long unknownRequestCount;
        private volatile WorkSnapshot materialized;

        private WorkCapture(long capturedAtMs, List<WorkSnapshot.RequestWork> requests,
                            List<WorkSnapshot.BatchWork> batches, long unknownRequestCount) {
            this.capturedAtMs = capturedAtMs;
            this.requests = List.copyOf(requests);
            this.batches = List.copyOf(batches);
            this.unknownRequestCount = unknownRequestCount;
        }

        /** Shared lazy projection; callers never hold the endpoint ownership lock. */
        public WorkSnapshot materialize() {
            WorkSnapshot snapshot = materialized;
            if (snapshot != null) {
                return snapshot;
            }
            synchronized (this) {
                if (materialized == null) {
                    List<WorkSnapshot.RequestWork> orderedRequests = requests.stream()
                            .sorted(Comparator.comparingLong(WorkSnapshot.RequestWork::requestId)).toList();
                    List<WorkSnapshot.BatchWork> orderedBatches = batches.stream()
                            .sorted(Comparator.comparingLong(WorkSnapshot.BatchWork::batchId))
                            .map(batch -> new WorkSnapshot.BatchWork(batch.batchId(),
                                    batch.requestIds().stream().sorted().toList(),
                                    batch.phase(), batch.remainingWorkMs())).toList();
                    materialized = new WorkSnapshot(capturedAtMs, orderedRequests, orderedBatches, unknownRequestCount);
                }
                return materialized;
            }
        }
    }

    private record TerminalObservation(long requestId,
                                       long batchId,
                                       long executionTimeMs,
                                       long errorCode,
                                       PriorityPreemptionProgress preemptionProgress,
                                       boolean workerObserved) {
        private static TerminalObservation from(
                WorkerStatus.TaskObservation task,
                boolean preserveBatchId) {
            return new TerminalObservation(
                    task.requestId(),
                    preserveBatchId ? task.batchId() : -1L,
                    task.executionTimeMs(),
                    task.errorCode(),
                    task.priorityPreemptionProgress(),
                    true);
        }

        private static TerminalObservation external(long requestId) {
            return new TerminalObservation(
                    requestId, -1L, -1L, 0L,
                    PriorityPreemptionProgress.NONE, false);
        }

        private TerminalObservation merge(TerminalObservation other) {
            if (requestId != other.requestId) {
                throw new IllegalArgumentException(
                        "cannot merge different terminal requests");
            }
            if (batchId >= 0L && other.batchId >= 0L
                    && batchId != other.batchId) {
                throw new IllegalArgumentException(
                        "cannot merge different terminal batches");
            }
            long mergedError = errorCode != 0L ? errorCode : other.errorCode;
            return new TerminalObservation(
                    requestId,
                    batchId >= 0L ? batchId : other.batchId,
                    Math.max(executionTimeMs, other.executionTimeMs),
                    mergedError,
                    strongerPreemptionProgress(
                            preemptionProgress, other.preemptionProgress),
                    workerObserved || other.workerObserved);
        }
    }

    private final ReentrantLock lock;
    /** Non-owning index containing only ACTIVE ScheduledRequest identities. */
    private final PrefillActiveIndex activeIndex;
    /** Canonical request ownership, changed only under the endpoint lock. */
    private Map<Long, RequestEntry> requests = new HashMap<>();
    private final LongSupplier clock;
    private final Runnable capacityAvailable;
    /** Monotonic ownership/work revision used by projection snapshots. */
    private volatile long mutationVersion;
    /** Published at ownership mutation boundaries; admission still checks under the lock. */
    private volatile long outstandingRequestCount;
    /** Derived immutable work; ACTIVE queue mutations leave committed work unchanged. */
    private WorkCapture committedWorkCapture;
    private long unknownEngineRequestCount;
    private int batchLeasesInUse;

    /** Publish the capacity summary before readers observe a new ownership revision. */
    private void recordMutationUnderLock() {
        publishRequestCountUnderLock();
        mutationVersion++;
    }

    private void publishRequestCountUnderLock() {
        requireLock();
        long count = saturatedAdd(requests.size(), unknownEngineRequestCount);
        if (outstandingRequestCount != count) {
            outstandingRequestCount = count;
        }
    }

    public PrefillState(ReentrantLock lock, PrefillActiveIndex activeIndex,
                        Runnable capacityAvailable) {
        this(lock, activeIndex, System::currentTimeMillis, capacityAvailable);
    }

    public PrefillState(ReentrantLock lock, PrefillActiveIndex activeIndex,
                        LongSupplier clock, Runnable capacityAvailable) {
        this.lock = Objects.requireNonNull(lock, "lock");
        this.activeIndex = Objects.requireNonNull(activeIndex, "activeIndex");
        this.clock = Objects.requireNonNull(clock, "clock");
        this.capacityAvailable = Objects.requireNonNull(
                capacityAvailable, "capacityAvailable");
    }

    private RequestEntry putRequestUnderLock(long requestId, RequestEntry entry) {
        requireLock();
        return requests.put(requestId, entry);
    }

    private boolean removeRequestUnderLock(long requestId, RequestEntry entry) {
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

    /** Advisory revision read for the lock-free projection-cache fast path. */
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

    public boolean terminalizeActiveUnderLock(ScheduledRequest item) {
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
        // generation handoff. NON_BATCH item-owned route reservations use the
        // atomic counterpart below so retirement can never observe an orphan.
        removeRequestUnderLock(item.requestId(), entry);
        recordMutationUnderLock();
        return true;
    }

    public boolean canPreemptQueuedRequest(int priority, long requestLimit) {
        lock.lock();
        try {
            long required = requestSlotsToReleaseUnderLock(requestLimit);
            if (!PriorityNormalizer.hasPriority(priority) || required == 0L || required > activeIndex.size()) {
                return false;
            }
            for (ScheduledRequest item : activeIndex) {
                if (isQueuedPreemptionCandidate(item, priority) && --required == 0L) { return true; }
            }
            return false;
        } finally {
            lock.unlock();
        }
    }

    /** Select only uncommitted requests; Engine work cannot release a local queue seat. */
    public List<ScheduledRequest> queuedPreemptionVictimsUnderLock(int priority, long requestLimit) {
        requireLock();
        long required = requestSlotsToReleaseUnderLock(requestLimit);
        if (!PriorityNormalizer.hasPriority(priority) || required == 0L || required > activeIndex.size()) {
            return List.of();
        }
        List<ScheduledRequest> candidates = new ArrayList<>();
        for (ScheduledRequest item : activeIndex) {
            if (isQueuedPreemptionCandidate(item, priority)) { candidates.add(item); }
        }
        if (candidates.size() < required) { return List.of(); }
        candidates.sort(Comparator.comparingInt(ScheduledRequest::priority)
                .thenComparing(Comparator.comparingLong(ScheduledRequest::enqueueSeq).reversed()));
        return candidates.subList(0, (int) required);
    }

    private boolean isQueuedPreemptionCandidate(ScheduledRequest item, int priority) {
        requireLock();
        RequestEntry entry = requests.get(item.requestId());
        return PriorityNormalizer.hasPriority(item.priority()) && item.priority() < priority
                && !item.future().isDone() && entry != null && entry.activeIdentity(item)
                && entry.reservation instanceof RouteReservation && entry.reservation.state == LeaseState.OPEN;
    }

    private long requestSlotsToReleaseUnderLock(long requestLimit) {
        requireLock();
        return requestLimit <= 0L ? 0L
                : Math.max(0L, saturatedAdd(requests.size(), unknownEngineRequestCount) - requestLimit + 1L);
    }

    /** Transfer exact queued request seats and OPEN route reservations in one ownership transaction. */
    public RouteReservation replaceQueuedRoutesUnderLock(
            List<ScheduledRequest> victims, List<RouteReservation> reservations,
            ScheduledRequest incoming, long requestLimit) {
        requireLock();
        if (victims.isEmpty() || victims.size() != reservations.size()
                || victims.size() != requestSlotsToReleaseUnderLock(requestLimit)
                || requests.containsKey(incoming.requestId())) {
            return null;
        }
        Set<Long> ids = new HashSet<>();
        for (int index = 0; index < victims.size(); index++) {
            ScheduledRequest victim = victims.get(index);
            RequestEntry entry = requests.get(victim.requestId());
            RouteReservation reservation = reservations.get(index);
            if (!ids.add(victim.requestId()) || entry == null || !entry.activeIdentity(victim)
                    || !activeIndex.contains(victim) || reservation == null || reservation.owner != this
                    || entry.reservation != reservation || reservation.state != LeaseState.OPEN
                    || !PriorityNormalizer.hasPriority(victim.priority()) || victim.priority() >= incoming.priority()) {
                return null;
            }
        }
        RequestEntry incomingEntry = new RequestEntry(incoming);
        RouteReservation incomingReservation = new RouteReservation(incomingEntry, incoming.requestId(), 0L);
        incomingEntry.reservation = incomingReservation;
        activeIndex.add(incoming);
        try {
            putRequestUnderLock(incoming.requestId(), incomingEntry);
        } catch (RuntimeException | Error failure) {
            activeIndex.remove(incoming);
            throw failure;
        }
        for (int index = 0; index < victims.size(); index++) {
            ScheduledRequest victim = victims.get(index);
            RequestEntry entry = requests.get(victim.requestId());
            closeOpenLeaseUnderLock(reservations.get(index));
            activeIndex.remove(victim);
            removeRequestUnderLock(victim.requestId(), entry);
        }
        recordMutationUnderLock();
        return incomingReservation;
    }

    /**
     * Remove one NON_BATCH ACTIVE owner and its item-owned OPEN route reservation
     * in the same ownership transaction.
     */
    public boolean terminalizeActiveRouteUnderLock(
            ScheduledRequest item,
            RouteReservation exactReservation) {
        requireLock();
        RequestEntry entry = requests.get(item.requestId());
        if (entry == null || !entry.activeIdentity(item)) {
            return false;
        }
        if (exactReservation == null
                || exactReservation.owner != this
                || entry.reservation != exactReservation
                || exactReservation.state != LeaseState.OPEN) {
            throw new IllegalStateException(
                    "ACTIVE route request lost its exact Prefill reservation request_id="
                            + item.requestId());
        }
        validateAdmissionIndexUnderLock(entry, item);

        // Every operation below is allocation-free and runs under the same
        // lock used by generation retirement.
        detachAdmissionIndexUnderLock(entry, item);
        closeOpenLeaseUnderLock(exactReservation);
        removeRequestUnderLock(item.requestId(), entry);
        recordMutationUnderLock();
        return true;
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

    ReservationResult<RouteReservation> reserveRoute(
            ScheduledRequest exactItem,
            long predictedMs) {
        ScheduledRequest item = exactItem;
        lock.lock();
        try {
            RequestEntry entry = requests.get(item.requestId());
            if (entry == null || !entry.activeIdentity(item)) {
                return new ReservationResult<>(
                        CapacityStatus.REQUEST_NOT_ACTIVE, null);
            }
            if (entry.reservation != null) {
                return new ReservationResult<>(
                        CapacityStatus.REQUEST_ALREADY_RESERVED, null);
            }
            RouteReservation lease = new RouteReservation(
                    entry, item.requestId(), predictedMs);
            entry.reservation = lease;
            recordMutationUnderLock();
            return new ReservationResult<>(CapacityStatus.ACQUIRED, lease);
        } finally {
            lock.unlock();
        }
    }

    ReservationResult<BatchReservation> reserveBatch(
            ScheduledRequest exactHead,
            long batchId,
            int maximum,
            EndpointGenerationLifecycle.HandoffPermit generationHandoff) {
        requirePositiveBatchLimit(maximum);
        ScheduledRequest head = exactHead;
        lock.lock();
        try {
            RequestEntry entry = requests.get(head.requestId());
            if (entry == null || !entry.activeIdentity(head)) {
                return new ReservationResult<>(
                        CapacityStatus.REQUEST_NOT_ACTIVE, null);
            }
            if (entry.reservation != null) {
                return new ReservationResult<>(
                        CapacityStatus.REQUEST_ALREADY_RESERVED, null);
            }
            if (findBatchReservationUnderLock(batchId) != null) {
                return new ReservationResult<>(
                        CapacityStatus.BATCH_ID_ALREADY_RESERVED, null);
            }
            if (batchLeasesInUse >= maximum) {
                return new ReservationResult<>(
                        CapacityStatus.CAPACITY_FULL, null);
            }
            BatchReservation lease = new BatchReservation(
                    entry, head.requestId(), batchId, generationHandoff);
            entry.reservation = lease;
            batchLeasesInUse++;
            recordMutationUnderLock();
            return new ReservationResult<>(CapacityStatus.ACQUIRED, lease);
        } finally {
            lock.unlock();
        }
    }

    private boolean batchCapacityAvailable(int maximum) {
        lock.lock();
        try {
            return batchLeasesInUse < maximum;
        } finally {
            lock.unlock();
        }
    }

    public CapacityBoundary.Availability batchAvailability(int maximum) {
        requirePositiveBatchLimit(maximum);
        return new CapacityAvailability(maximum);
    }

    private static void requirePositiveBatchLimit(int maximum) {
        if (maximum <= 0) {
            throw new IllegalArgumentException("maximumInflightBatches must be positive");
        }
    }

    /** Exact wake capability permanently paired with this worker runtime. */
    private final class CapacityAvailability
            implements CapacityBoundary.Availability {
        private final int maximum;
        private Runnable subscribed;

        private CapacityAvailability(int maximum) {
            this.maximum = maximum;
        }

        @Override
        public boolean isAvailable() {
            return batchCapacityAvailable(maximum);
        }

        @Override
        public synchronized void addListener(Runnable listener) {
            if (listener != capacityAvailable) {
                throw new IllegalArgumentException(
                        "Prefill availability requires its exact worker wake callback");
            }
            if (subscribed != null && subscribed != listener) {
                throw new IllegalStateException(
                        "Prefill availability already has a listener");
            }
            subscribed = listener;
        }

        @Override
        public synchronized void removeListener(Runnable listener) {
            if (subscribed == listener) {
                subscribed = null;
            }
        }
    }

    CommittedHandoff commitRouteGroup(
            List<ScheduledRequest> items,
            List<RouteReservation> exactReservations,
            EndpointGenerationLifecycle.HandoffPermit generationHandoff) {
        Objects.requireNonNull(generationHandoff, "generationHandoff");
        if (exactReservations.isEmpty()) {
            throw new IllegalArgumentException(
                    "route commit requires at least one reservation");
        }
        List<RouteReservation> leases = new ArrayList<>(
                exactReservations.size());
        for (RouteReservation reservation : exactReservations) {
            if (reservation == null || reservation.owner != this) {
                throw new IllegalArgumentException(
                        "route reservation belongs to another Prefill ledger");
            }
            leases.add(reservation);
        }
        lock.lock();
        try {
            return commitRoutesUnderLock(items, leases, generationHandoff);
        } finally {
            lock.unlock();
        }
    }

    private CommittedHandoff commitRoutesUnderLock(
            List<ScheduledRequest> items,
            List<RouteReservation> leases,
            EndpointGenerationLifecycle.HandoffPermit generationHandoff) {
        requireLock();
        List<ScheduledRequest> members = validateRouteGroup(items);
        if (members.size() != leases.size()) {
            throw new IllegalArgumentException(
                    "route commit requires one exact lease per member");
        }
        Set<RouteReservation> unique = java.util.Collections.newSetFromMap(
                new IdentityHashMap<>());
        for (int index = 0; index < members.size(); index++) {
            RequestEntry entry = requests.get(members.get(index).requestId());
            RouteReservation lease = leases.get(index);
            if (entry == null || !unique.add(lease)
                    || entry.reservation != lease
                    || lease.requestId != entry.requestId
                    || lease.state != LeaseState.OPEN) {
                throw new IllegalStateException(
                        "route commit does not own exact OPEN lease request_id="
                                + members.get(index).requestId());
            }
        }
        long nowMs = clock.getAsLong();
        CommittedHandoff committedHandoff = new CommittedHandoff(generationHandoff,
                capturePrecedingWorkUnderLock(members, nowMs));
        for (ScheduledRequest item : members) {
            detachAdmissionIndexUnderLock(requests.get(item.requestId()), item);
        }
        for (int index = 0; index < members.size(); index++) {
            RouteReservation lease = leases.get(index);
            lease.state = LeaseState.OWNED;
            RequestEntry entry = requests.get(members.get(index).requestId());
            entry.commitIndividual(lease, nowMs);
        }
        committedWorkCapture = null;
        recordMutationUnderLock();
        return committedHandoff;
    }

    private CommittedHandoff commitBatchUnderLock(
            BatchReservation lease,
            List<ScheduledRequest> items,
            long predictedMs) {
        requireLock();
        List<ScheduledRequest> members = validateActiveGroup(items);
        RequestEntry head = requests.get(lease.headRequestId);
        ScheduledRequest headItem = head == null ? null : head.activeItem;
        if (lease.state != LeaseState.OPEN || head == null
                || head.reservation != lease
                || !members.contains(headItem)
                || openGenerationHandoff(lease) == null) {
            throw new IllegalStateException(
                    "batch commit does not own exact OPEN lease batch_id="
                            + lease.batchId);
        }
        if (findBatchWorkUnderLock(lease.batchId) != null) {
            throw new IllegalStateException(
                    "batch id already committed batch_id=" + lease.batchId);
        }
        for (ScheduledRequest item : members) {
            RequestEntry member = requests.get(item.requestId());
            Reservation expected = item == headItem ? lease : null;
            if (member.reservation != expected) {
                throw new IllegalStateException(
                        "batch member owns another exact reservation request_id="
                                + item.requestId());
            }
        }
        long nowMs = clock.getAsLong();
        BatchWork work = new BatchWork(
                lease,
                predictedMs,
                PrefillBatchFeatures.from(
                        members,
                        ScheduledRequest::seqLen,
                        ScheduledRequest::hitCache),
                nowMs);
        CommittedHandoff committedHandoff = new CommittedHandoff(openGenerationHandoff(lease),
                capturePrecedingWorkUnderLock(members, nowMs));
        for (ScheduledRequest item : members) {
            removeValidatedActiveIndex(item);
        }
        moveGenerationHandoffToOwnedUnderLock(lease, committedHandoff);
        for (ScheduledRequest item : members) {
            requests.get(item.requestId()).commitBatch(work);
        }
        committedWorkCapture = null;
        recordMutationUnderLock();
        return committedHandoff;
    }

    private List<ScheduledRequest> validateActiveGroup(List<ScheduledRequest> items) {
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

    private void removeValidatedActiveIndex(ScheduledRequest item) {
        boolean removed = activeIndex.remove(item);
        requireState(removed,
                "validated ACTIVE queue index disappeared request_id="
                        + item.requestId());
    }

    /**
     * Reserve immediate route work against current ownership in one transaction.
     * Selection revisions are advisory; only exact identity and current capacity
     * decide admission. A zero request limit leaves count admission disabled.
     */
    public ReservationResult<RouteReservation> reserveUnqueuedRoute(
            ScheduledRequest item, long predictedMs, long maxOutstandingRequests) {
        Objects.requireNonNull(item, "item");
        if (maxOutstandingRequests < 0L) {
            throw new IllegalArgumentException("request limit must be non-negative");
        }
        lock.lock();
        try {
            if (requests.containsKey(item.requestId())) {
                return new ReservationResult<>(CapacityStatus.REQUEST_ALREADY_RESERVED, null);
            }
            if (maxOutstandingRequests > 0L && !canAcceptRequestUnderLock(maxOutstandingRequests)) {
                return new ReservationResult<>(CapacityStatus.CAPACITY_FULL, null);
            }
            RequestEntry entry = new RequestEntry(item, QueueMembership.UNINDEXED);
            RouteReservation reservation = new RouteReservation(entry, item.requestId(), predictedMs);
            ReservationResult<RouteReservation> result = new ReservationResult<>(CapacityStatus.ACQUIRED, reservation);
            entry.reservation = reservation;
            putRequestUnderLock(item.requestId(), entry);
            committedWorkCapture = null;
            recordMutationUnderLock();
            return result;
        } finally {
            lock.unlock();
        }
    }

    private List<ScheduledRequest> validateRouteGroup(List<ScheduledRequest> items) {
        requireState(!items.isEmpty(), "committed group requires members");
        for (ScheduledRequest item : items) {
            RequestEntry entry = requests.get(item.requestId());
            requireState(entry != null && entry.activeIdentity(item),
                    "route member is not canonical admission request_id=" + item.requestId());
            validateAdmissionIndexUnderLock(entry, item);
        }
        return items;
    }

    private void validateAdmissionIndexUnderLock(RequestEntry entry, ScheduledRequest item) {
        requireState(entry.queueMembership != QueueMembership.STOP_DETACHED,
                "stopped admission cannot commit request_id=" + item.requestId());
        requireState(activeIndex.contains(item) == (entry.queueMembership == QueueMembership.WAITING),
                "canonical admission and waiting index disagree request_id=" + item.requestId());
    }

    private void detachAdmissionIndexUnderLock(RequestEntry entry, ScheduledRequest item) {
        validateAdmissionIndexUnderLock(entry, item);
        if (entry.queueMembership == QueueMembership.WAITING) {
            removeValidatedActiveIndex(item);
        }
        // The caller removes this entry or commits it before releasing the lock.
        // Keep its admission origin until then: removing queued-only work must
        // not invalidate the immutable committed-work projection.
    }

    /**
     * Total counterpart cleanup bound to one exact committed ScheduledRequest. A
     * reused request id or an ACTIVE item is a no-op.
     */
    public boolean terminalizeCommittedItem(ScheduledRequest exactItem) {
        boolean capacityReleased = false;
        lock.lock();
        try {
            RequestEntry entry = requests.get(exactItem.requestId());
            if (entry == null || entry.isActive()
                    || entry.committedItem != exactItem) {
                return false;
            }
            BatchReduction reduction = entry.batchWork == null
                    ? null : batchReductionUnderLock(entry.batchWork);
            // Local cleanup does not prove that Engine omitted this member's work.
            capacityReleased = settleUnderLock(entry, TerminalObservation.external(entry.requestId),
                    reduction, null);
            return true;
        } finally {
            lock.unlock();
            notifyCapacityAvailable(capacityReleased);
        }
    }

    /**
     * Reconcile ownership and publish its matching WorkerStatus before the
     * canonical queue lock is released. Projection readers therefore observe
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

    public HeartbeatReconciliation reconcileHeartbeat(
            WorkerStatus.StatusObservation observation) {
        List<WorkerStatusFact> facts = new ArrayList<>(
                observation.runningTasks().size());
        boolean capacityReleased = false;
        boolean schedulingInputsChanged = false;
        lock.lock();
        try {
            long nowMs = clock.getAsLong();
            WorkerStatus.EngineObservation engine = observation.engine();
            IdentityHashMap<RequestEntry, Phase> individualPhases =
                    new IdentityHashMap<>();
            IdentityHashMap<BatchWork, Phase> batchPhases =
                    new IdentityHashMap<>();
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
                schedulingInputsChanged |= observed.getKey().individualPhase != observed.getValue();
                observed.getKey().observeIndividualPhase(observed.getValue(), nowMs);
            }
            for (var observed : batchPhases.entrySet()) {
                schedulingInputsChanged |= observed.getKey().servicePhase != observed.getValue();
                observed.getKey().observePhase(observed.getValue(), nowMs);
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
            Map<Long, TerminalObservation> terminals,
            Map<String, WorkerStatus.TaskObservation> activeTasks,
            IdentityHashMap<BatchWork, BatchReduction> reductions) {
        requireLock();
        List<WorkerStatusFact> schedulerFacts = new ArrayList<>(
                terminals.size() + activeTasks.size());
        for (TerminalObservation terminal : terminals.values()) {
            WorkerStatusFact fact = terminalFact(
                    requests.get(terminal.requestId), terminal);
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

        List<BatchCompletion> completions = new ArrayList<>(
                Math.min(terminals.size(), reductions.size()));
        for (BatchReduction reduction : reductions.values()) {
            BatchCompletion completion = reduction.projectedCompletion(terminals);
            if (completion != null) {
                completions.add(completion);
            }
        }
        return new StatusReconciliation(
                schedulerFacts, completions, null);
    }

    /** Resolve activity only through the exact committed ledger identity. */
    private WorkerStatusFact activeStatusFactUnderLock(
            WorkerStatus.TaskObservation task) {
        requireLock();
        RequestEntry entry = requests.get(task.requestId());
        if (entry == null
                || entry.isActive()
                || !matchesObservedBatch(entry, task.batchId())
                || entry.committedItem == null
                || isPriorityCancelOverlayOnly(task)) {
            return null;
        }
        return WorkerStatusFact.active(entry.committedItem);
    }

    private void prepareBatchPredictionsUnderLock(
            Set<BatchReduction> changedBatches,
            Map<Long, TerminalObservation> terminals,
            IdentityHashMap<BatchWork, Phase> batchPhases,
            ToLongFunction<List<ScheduledRequest>> repredictor,
            IdentityHashMap<BatchWork, Long> predictions) {
        requireLock();
        for (BatchReduction reduction : changedBatches) {
            BatchWork batch = reduction.batch;
            if (batch.executionStarted
                    || batchPhases.get(batch) == Phase.ENGINE_RUNNING) {
                continue;
            }
            List<ScheduledRequest> survivors = new ArrayList<>(
                    reduction.members.size());
            boolean executionObserved = false;
            for (RequestEntry member : reduction.members) {
                TerminalObservation terminal = terminals.get(member.requestId);
                if (terminal == null) {
                    survivors.add(member.committedItem);
                } else {
                    // Completion can arrive before the first RUNNING observation.
                    executionObserved |= terminal.errorCode == 0L || terminal.executionTimeMs > 0L;
                }
            }
            if (executionObserved || survivors.isEmpty()) {
                continue;
            }
            survivors.sort(Comparator.comparingLong(ScheduledRequest::enqueueSeq)
                    .thenComparingLong(ScheduledRequest::requestId));
            long prediction = repredictor.applyAsLong(survivors);
            predictions.put(batch, prediction);
        }
    }

    private long prepareActiveObservationsUnderLock(
            Map<String, WorkerStatus.TaskObservation> activeTasks,
            Map<Long, TerminalObservation> terminals,
            long reportedActive,
            IdentityHashMap<RequestEntry, Phase> individualPhases,
            IdentityHashMap<BatchWork, Phase> batchPhases,
            List<WorkerStatusFact> activeFacts) {
        requireLock();
        Set<Long> unknownDetailed = new HashSet<>();
        Set<Long> knownObserved = new HashSet<>();
        for (WorkerStatus.TaskObservation task : activeTasks.values()) {
            if (terminals.containsKey(task.requestId())) {
                continue;
            }
            RequestEntry entry = requests.get(task.requestId());
            if (entry == null || !matchesObservedBatch(entry, task.batchId())) {
                if (!isPriorityCancelOverlayOnly(task)) {
                    unknownDetailed.add(task.requestId());
                }
                continue;
            }
            if (!isPriorityCancelOverlayOnly(task)) {
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
            if (entry.batchWork == null) {
                individualPhases.put(entry, observed);
            } else {
                batchPhases.merge(
                        entry.batchWork,
                        observed,
                        PrefillState::strongerEnginePhase);
            }
        }
        long scalarUnknown = Math.max(
                0L, reportedActive - knownObserved.size());
        return Math.max(unknownDetailed.size(), scalarUnknown);
    }

    private StatusReconciliation reconcileEngineStatus(
            WorkerStatus.EngineObservation engine,
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
            IdentityHashMap<BatchWork, BatchReduction> reductions =
                    batchReductionsUnderLock();
            Map<Long, TerminalObservation> terminals = terminalObservationsUnderLock(
                    finishedTasks);
            Map<String, WorkerStatus.TaskObservation> activeTasks =
                    engine.runningTaskList();
            IdentityHashMap<RequestEntry, TerminalObservation> settlements =
                    new IdentityHashMap<>(terminals.size());
            Set<BatchReduction> changedBatches = java.util.Collections.newSetFromMap(
                    new IdentityHashMap<>());
            IdentityHashMap<BatchWork, Phase> batchPhases = new IdentityHashMap<>();
            for (TerminalObservation terminal : terminals.values()) {
                RequestEntry entry = requests.get(terminal.requestId);
                if (entry == null
                        || settlements.put(entry, terminal) != null) {
                    throw new IllegalStateException(
                            "status terminal lost its exact owner request_id="
                                    + terminal.requestId);
                }
                BatchReduction reduction = entry.batchWork == null
                        ? null : reductions.get(entry.batchWork);
                if (reduction != null) {
                    changedBatches.add(reduction);
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
            IdentityHashMap<BatchWork, Long> predictions = new IdentityHashMap<>();
            prepareBatchPredictionsUnderLock(
                    changedBatches, terminals, batchPhases, repredictor, predictions);
            outcome = prepareStatusReconciliationUnderLock(
                    terminals, activeTasks, reductions);

            // Everything below this boundary is assignment/removal against
            // exact prevalidated identities. Any invariant failure is captured
            // into the already materialized outcome and forces retirement.
            canonicalMutationStarted = true;
            committedWorkCapture = null;
            recordMutationUnderLock();
            for (Map.Entry<RequestEntry, TerminalObservation> settlement
                    : settlements.entrySet()) {
                RequestEntry entry = settlement.getKey();
                capacityReleased |= settleUnderLock(
                        entry,
                        settlement.getValue(),
                        entry.batchWork == null
                                ? null : reductions.get(entry.batchWork),
                        null,
                        nowMs);
            }
            individualPhases.forEach(
                    (entry, phase) -> entry.observeIndividualPhase(
                            phase, nowMs));
            batchPhases.forEach(
                    (batch, phase) -> batch.observePhase(phase, nowMs));
            predictions.forEach((batch, prediction) -> {
                batch.remainingWorkMs = prediction;
                batch.phaseBaseMs = nowMs;
                batch.touch(nowMs);
            });
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

    private static StatusReconciliation withPublicationFailure(
            StatusReconciliation outcome,
            Throwable failure) {
        return outcome.publicationFailure() != null
                ? outcome
                : new StatusReconciliation(
                        outcome.schedulerFacts(),
                        outcome.batchCompletions(),
                        failure);
    }

    /**
     * End every owner for this retired Prefill generation in one queue-lock
     * transaction. All callback facts and completion DTOs are constructed
     * before the canonical table, queue index, and capacity counters are
     * cleared. Once clearing starts, the remaining operations are allocation-
     * free field updates; no failure can leave a partially retired registry.
     *
     * <p>Ordinarily {@link WorkerBatcher#stopAndAwait()} has already reduced every
     * ACTIVE item. Including a defensively remaining ACTIVE identity here makes
     * endpoint close total if that earlier invariant check failed.</p>
     */
    public Retirement retireGenerationOwnership() {
        List<ScheduledRequest> ownedItems = new ArrayList<>();
        List<BatchCompletion> completions = new ArrayList<>();
        List<EndpointGenerationLifecycle.HandoffPermit> orphanedHandoffs =
                new ArrayList<>();
        Set<Reservation> leases = java.util.Collections.newSetFromMap(
                new IdentityHashMap<>());
        Set<BatchWork> batches = java.util.Collections.newSetFromMap(
                new IdentityHashMap<>());
        Throwable invariantFailure = null;
        Retirement plannedRetirement;
        lock.lock();
        try {
            Set<ScheduledRequest> canonicalActive = java.util.Collections.newSetFromMap(
                    new IdentityHashMap<>());
            for (Map.Entry<Long, RequestEntry> canonical : requests.entrySet()) {
                RequestEntry entry = canonical.getValue();
                if (canonical.getKey() != entry.requestId) {
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
                if (entry.batchWork != null) {
                    batches.add(entry.batchWork);
                    leases.add(entry.batchWork.lease);
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
                if (lease instanceof RouteReservation) {
                    if (lease.state != LeaseState.OPEN
                            && lease.state != LeaseState.OWNED) {
                        invariantFailure = appendRetirementInvariant(
                                invariantFailure,
                                "canonical retirement owner has an invalid route-reservation state");
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
            if (batchLeasesInUse != observedBatchLeases) {
                invariantFailure = appendRetirementInvariant(
                        invariantFailure,
                        "batch capacity does not match canonical retirement leases");
            }

            for (BatchWork batch : batches) {
                completions.add(batch.retirementCompletion());
            }
            ownedItems.sort(Comparator.comparingLong(ScheduledRequest::enqueueSeq)
                    .thenComparingLong(ScheduledRequest::requestId));
            completions.sort(Comparator.comparingLong(BatchCompletion::batchId));
            plannedRetirement = new Retirement(
                    ownedItems,
                    completions,
                    invariantFailure);

            // Canonical retirement commit. Everything which may allocate or
            // validate has completed above this line.
            for (RequestEntry entry : requests.values()) {
                entry.activeItem = null;
                entry.committedItem = null;
                entry.reservation = null;
                entry.batchWork = null;
                entry.individualPhase = null;
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
            batchLeasesInUse = 0;
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

    public int evictExpiredIndividuals(
            long ttlMs, java.util.function.LongPredicate schedulerOwnsRequest) {
        int evicted = 0;
        boolean capacityReleased = false;
        lock.lock();
        try {
            long nowMs = clock.getAsLong();
            List<RequestEntry> candidates = new ArrayList<>();
            for (RequestEntry entry : requests.values()) {
                if (entry.batchWork == null
                        && !entry.isActive()
                        && nowMs - entry.lastObservedAtMs >= Math.max(0L, ttlMs)
                        && !schedulerOwnsRequest.test(entry.requestId)) {
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

    public int evictExpiredBatches(
            long ttlMs, java.util.function.LongPredicate schedulerOwnsRequest) {
        int evicted = 0;
        boolean capacityReleased = false;
        lock.lock();
        try {
            long nowMs = clock.getAsLong();
            for (BatchReduction reduction
                    : batchReductionsUnderLock().values()) {
                boolean retained = nowMs - reduction.batch.lastObservedAtMs
                        < Math.max(0L, ttlMs);
                for (RequestEntry entry : reduction.members) {
                    retained |= schedulerOwnsRequest.test(entry.requestId);
                }
                if (retained) {
                    continue;
                }
                for (RequestEntry entry : List.copyOf(reduction.members)) {
                    capacityReleased |= settleUnderLock(
                            entry,
                            TerminalObservation.external(entry.requestId),
                            reduction,
                            new ArrayList<>(0));
                }
                evicted++;
            }
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
            int individual = 0;
            long maxAgeMs = 0L;
            Set<BatchWork> batches = java.util.Collections.newSetFromMap(
                    new IdentityHashMap<>());
            long nowMs = clock.getAsLong();
            for (RequestEntry entry : requests.values()) {
                if (entry.isActive()) {
                    continue;
                }
                locallyOwned++;
                if (entry.batchWork == null) {
                    individual++;
                    maxAgeMs = Math.max(
                            maxAgeMs,
                            Math.max(0L, nowMs - entry.lastObservedAtMs));
                } else {
                    batches.add(entry.batchWork);
                }
            }
            for (BatchWork batch : batches) {
                maxAgeMs = Math.max(
                        maxAgeMs,
                        Math.max(0L, nowMs - batch.lastObservedAtMs));
            }
            return new Stats(
                    locallyOwned,
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
        List<ScheduledRequest> active = new ArrayList<>(activeIndex.size());
        activeIndex.forEach(active::add);
        return new Snapshot(
                nowMs,
                active,
                captureWorkUnderLock(nowMs));
    }

    public WorkSnapshot committedSnapshot() {
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

    private WorkSnapshot capturePrecedingWorkUnderLock(List<ScheduledRequest> members, long nowMs) {
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
        List<WorkSnapshot.RequestWork> individual = new ArrayList<>();
        IdentityHashMap<BatchWork, List<Long>> batchMembers =
                new IdentityHashMap<>();
        for (RequestEntry entry : requests.values()) {
            if (excluded.contains(entry)) {
                continue;
            }
            if (entry.isActive()) {
                if (entry.queueMembership == QueueMembership.UNINDEXED
                        && entry.reservation instanceof RouteReservation route) {
                    // Capacity has been reserved for immediate handoff. Later admissions
                    // must include this work even before its route is published.
                    individual.add(new WorkSnapshot.RequestWork(
                            entry.requestId, Phase.COMMITTED, route.predictedWorkMs));
                }
                continue;
            }
            if (entry.batchWork == null) {
                individual.add(new WorkSnapshot.RequestWork(
                        entry.requestId,
                        entry.individualPhase,
                        individualRemaining(entry, nowMs)));
            } else {
                batchMembers.computeIfAbsent(
                                entry.batchWork, ignored -> new ArrayList<>())
                        .add(entry.requestId);
            }
        }
        List<WorkSnapshot.BatchWork> batches =
                new ArrayList<>(batchMembers.size());
        for (Map.Entry<BatchWork, List<Long>> observed : batchMembers.entrySet()) {
            BatchWork batch = observed.getKey();
            batches.add(new WorkSnapshot.BatchWork(
                    batch.lease.batchId,
                    observed.getValue(),
                    batch.servicePhase,
                    OptionalLong.of(batch.remainingAt(nowMs))));
        }
        return new WorkCapture(
                nowMs,
                individual,
                batches,
                unknownEngineRequestCount);
    }

    private boolean settleUnderLock(
            RequestEntry entry,
            TerminalObservation terminal,
            BatchReduction reduction,
            List<BatchCompletion> completions) {
        return settleUnderLock(
                entry,
                terminal,
                reduction,
                completions,
                clock.getAsLong());
    }

    private boolean settleUnderLock(
            RequestEntry entry,
            TerminalObservation terminal,
            BatchReduction reduction,
            List<BatchCompletion> completions,
            long nowMs) {
        requireLock();
        BatchWork batch = entry.batchWork;
        boolean lastBatchMember = false;
        Reservation lease;
        if (batch != null) {
            if (reduction == null || reduction.batch != batch
                    || !reduction.members.contains(entry)) {
                throw new IllegalStateException(
                        "missing batch reduction for live request_id="
                                + entry.requestId);
            }
            lastBatchMember = reduction.members.size() == 1;
            lease = lastBatchMember ? batch.lease : null;
        } else {
            lease = entry.reservation;
        }
        if (lease != null && lease.state != LeaseState.OWNED) {
            throw new IllegalStateException(
                    "terminal request owns a non-committed Prefill lease"
                            + " request_id=" + entry.requestId);
        }
        if (batch != null) {
            batch.observeTerminal(terminal, nowMs);
            reduction.remove(entry);
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
        if (lastBatchMember && completions != null) {
            completions.add(batch.completion());
        }
        return true;
    }

    private Map<Long, TerminalObservation> terminalObservationsUnderLock(
            Map<String, WorkerStatus.TaskObservation> finishedTasks) {
        requireLock();
        Map<Long, TerminalObservation> terminals = new HashMap<>();
        for (WorkerStatus.TaskObservation task : finishedTasks.values()) {
            RequestEntry entry = requests.get(task.requestId());
            if (entry == null || !matchesObservedBatch(entry, task.batchId())) {
                continue;
            }
            TerminalObservation terminal = TerminalObservation.from(
                    task, entry.batchWork != null);
            terminals.merge(
                    task.requestId(), terminal, TerminalObservation::merge);
        }
        return terminals;
    }

    private static WorkerStatusFact terminalFact(
            RequestEntry entry,
            TerminalObservation terminal) {
        if (entry == null || entry.committedItem == null
                || !terminal.workerObserved) {
            return null;
        }
        WorkerStatusFact.Kind kind = terminal.errorCode == 0L
                ? WorkerStatusFact.Kind.COMPLETED
                : terminal.preemptionProgress
                        == PriorityPreemptionProgress.CANCELED
                    && terminal.errorCode
                            == StrategyErrorType.PRIORITY_PREEMPTED.getErrorCode()
                ? WorkerStatusFact.Kind.PRIORITY_CANCELED
                : WorkerStatusFact.Kind.FAILED;
        return WorkerStatusFact.terminal(
                entry.committedItem, kind, terminal.errorCode);
    }

    private static PriorityPreemptionProgress strongerPreemptionProgress(
            PriorityPreemptionProgress left,
            PriorityPreemptionProgress right) {
        if (left == PriorityPreemptionProgress.CANCELED
                || right == PriorityPreemptionProgress.CANCELED) {
            return PriorityPreemptionProgress.CANCELED;
        }
        if (left == PriorityPreemptionProgress.CANCELING
                || right == PriorityPreemptionProgress.CANCELING) {
            return PriorityPreemptionProgress.CANCELING;
        }
        return PriorityPreemptionProgress.NONE;
    }

    private static boolean matchesObservedBatch(
            RequestEntry entry, long observedBatchId) {
        if (entry.batchWork == null) {
            return true;
        }
        return observedBatchId > 0L
                && entry.batchWork.lease.batchId == observedBatchId;
    }

    private static Phase strongerEnginePhase(Phase left, Phase right) {
        return left == Phase.ENGINE_RUNNING || right == Phase.ENGINE_RUNNING
                ? Phase.ENGINE_RUNNING : Phase.ENGINE_QUEUED;
    }

    private static boolean isPriorityCancelOverlayOnly(
            WorkerStatus.TaskObservation task) {
        PriorityPreemptionProgress progress =
                task.priorityPreemptionProgress();
        return (progress == PriorityPreemptionProgress.CANCELING
                || progress == PriorityPreemptionProgress.CANCELED)
                && task.phase() == TaskPhase.PENDING;
    }

    private IdentityHashMap<BatchWork, BatchReduction>
            batchReductionsUnderLock() {
        requireLock();
        IdentityHashMap<BatchWork, BatchReduction> reductions =
                new IdentityHashMap<>();
        for (RequestEntry entry : requests.values()) {
            if (entry.batchWork != null) {
                reductions.computeIfAbsent(
                                entry.batchWork, BatchReduction::new)
                        .add(entry);
            }
        }
        return reductions;
    }

    /** Build only the exact batch needed by a single-item terminal path. */
    private BatchReduction batchReductionUnderLock(BatchWork exactBatch) {
        requireLock();
        BatchReduction reduction = new BatchReduction(exactBatch);
        for (RequestEntry entry : requests.values()) {
            if (entry.batchWork == exactBatch) {
                reduction.add(entry);
            }
        }
        return reduction;
    }

    private BatchReservation findBatchReservationUnderLock(long batchId) {
        for (RequestEntry entry : requests.values()) {
            if (entry.reservation instanceof BatchReservation reservation
                    && reservation.batchId == batchId) {
                return reservation;
            }
            if (entry.batchWork != null
                    && entry.batchWork.lease.batchId == batchId) {
                return entry.batchWork.lease;
            }
        }
        return null;
    }

    /**
     * Roll back only a provisional lease. Once committed, the request table is
     * the sole owner and only its terminal reducer may release the capacity.
     */
    private void releaseOpenLease(Reservation lease) {
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

    private void notifyCapacityAvailable(boolean capacityReleased) {
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

    private EndpointGenerationLifecycle.HandoffPermit openGenerationHandoff(
            BatchReservation lease) {
        requireLock();
        return lease.generationHandoff;
    }

    /** Move the exact handoff out of the OPEN admission before commit publishes. */
    private void moveGenerationHandoffToOwnedUnderLock(
            BatchReservation lease,
            CommittedHandoff committedHandoff) {
        requireLock();
        if (lease.state != LeaseState.OPEN
                || lease.generationHandoff == null
                || committedHandoff.generationHandoff
                    != lease.generationHandoff) {
            throw new IllegalStateException(
                    "Prefill commit requires an exact OPEN generation handoff");
        }
        lease.generationHandoff = null;
        lease.state = LeaseState.OWNED;
    }

    /** OPEN rollback closes quota and any batch-owned generation handoff. */
    private EndpointGenerationLifecycle.HandoffPermit
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
            if (owner.queueMembership == QueueMembership.UNINDEXED) {
                removeRequestUnderLock(owner.requestId, owner);
                committedWorkCapture = null;
            }
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
            if (batchLeasesInUse <= 0) {
                throw new IllegalStateException(
                        "Prefill batch capacity accounting underflow");
            }
            batchLeasesInUse--;
        } else if (!(lease instanceof RouteReservation)) {
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

    private BatchWork findBatchWorkUnderLock(long batchId) {
        for (RequestEntry entry : requests.values()) {
            if (entry.batchWork != null
                    && entry.batchWork.lease.batchId == batchId) {
                return entry.batchWork;
            }
        }
        return null;
    }

    private void requireLock() {
        requireState(lock.isHeldByCurrentThread(),
                "Prefill ownership requires queueLock");
    }

    private static void requireState(boolean condition, String message) {
        if (!condition) {
            throw new IllegalStateException(message);
        }
    }

    private static long individualRemaining(RequestEntry entry, long nowMs) {
        if (entry.individualPhase != Phase.ENGINE_RUNNING) {
            return entry.remainingWorkMs;
        }
        return Math.max(0L, entry.remainingWorkMs
                - Math.max(0L, nowMs - entry.phaseBaseMs));
    }

    private static long boundedPrediction(long predictedMs) {
        return Math.min(Integer.MAX_VALUE, Math.max(0L, predictedMs));
    }

    private static long saturatedAdd(long left, long right) {
        return left > Long.MAX_VALUE - right ? Long.MAX_VALUE : left + right;
    }
}
