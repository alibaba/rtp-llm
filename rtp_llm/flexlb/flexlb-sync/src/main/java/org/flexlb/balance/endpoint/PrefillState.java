package org.flexlb.balance.endpoint;

import org.flexlb.balance.prediction.PrefillBatchFeatures;
import org.flexlb.balance.projection.WorkSnapshot;
import org.flexlb.balance.projection.WorkSnapshot.Phase;
import org.flexlb.balance.scheduler.RequestRoute;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.enums.PriorityPreemptionProgress;
import org.flexlb.enums.TaskPhase;
import org.flexlb.util.Failures;
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

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;
import static com.google.common.math.LongMath.saturatedAdd;

/**
 * Canonical Prefill request ownership for one worker generation.
 *
 * <p>Every known request id has exactly one {@link RequestEntry}. The worker
 * queue is only an ordered index over entries waiting for worker delivery;
 * an immediate admission owns the same resource record without a queue entry.
 * Callback and Engine progress mutate the same entry instead of moving ownership
 * between containers. Methods ending in {@code Locked} require the caller to
 * hold {@link #ownershipLock()}, shared with the worker queue. Other mutation
 * methods acquire that lock internally. Methods return resource facts; Endpoint
 * executes notifications and request continuations after unlocking.
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

    public record ReservationResult<R extends Reservation>(
            CapacityStatus status,
            R reservation) {
        public ReservationResult {
            Objects.requireNonNull(status, "status");
            checkArgument((status == CapacityStatus.ACQUIRED) == (reservation != null),
                    "only ACQUIRED may carry a reservation");
        }
    }

    /** Request status matched to its exact Prefill route, published after ledger reconciliation. */
    public record PrefillRequestStatus(
            RequestRoute route,
            Kind kind,
            long errorCode) {
        public PrefillRequestStatus {
            Objects.requireNonNull(route, "route");
            Objects.requireNonNull(kind, "kind");
            checkArgument(kind != Kind.ACTIVE || errorCode == 0L,
                    "an active Prefill request status cannot carry an error code");
        }

        public static PrefillRequestStatus active(RequestRoute route) {
            return new PrefillRequestStatus(route, Kind.ACTIVE, 0L);
        }

        public static PrefillRequestStatus terminal(
                RequestRoute route, Kind kind, long errorCode) {
            checkArgument(kind != Kind.ACTIVE, "terminal Prefill request status requires a terminal kind");
            return new PrefillRequestStatus(route, kind, errorCode);
        }

        public enum Kind {
            ACTIVE,
            COMPLETED,
            FAILED,
            PRIORITY_CANCELED
        }
    }

    public record StatusReconciliation(
            List<PrefillRequestStatus> requestStatuses,
            List<BatchCompletion> batchCompletions,
            boolean schedulingInputsChanged,
            boolean capacityReleased) {
        public StatusReconciliation {
            requestStatuses = List.copyOf(requestStatuses);
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
            List<RequestRoute> ownedItems,
            List<BatchCompletion> batchCompletions,
            Throwable invariantFailure,
            List<EndpointGenerationLifecycle.HandoffPermit> orphanedHandoffs) {
        public Retirement {
            ownedItems = List.copyOf(ownedItems);
            batchCompletions = List.copyOf(batchCompletions);
            orphanedHandoffs = List.copyOf(orphanedHandoffs);
        }
    }

    public record AdmissionSummary(long occupiedRequests, long requestsToRelease,
            long lowerPriorityRequests, long samePriorityRequests, long higherPriorityRequests,
            long unknownPriorityRequests) { }

    public record Stats(
            int locallyOwnedRequests,
            int individuallyOwnedRequests,
            int batchCount,
            long maxObservedAgeMs) {
        public Stats {
            checkArgument(locallyOwnedRequests >= 0
                    && individuallyOwnedRequests >= 0
                    && batchCount >= 0
                    && maxObservedAgeMs >= 0L,
                    "Prefill state stats must be non-negative");
        }
    }

    /** One-shot capability returned when an OPEN admission commits. */
    public static final class CommittedHandoff implements AutoCloseable {
        private final EndpointGenerationLifecycle.HandoffPermit generationHandoff;
        private final WorkCapture precedingWork;
        private final int remainingQueueDepth;
        private final boolean removedFailedMember;

        private CommittedHandoff(
                EndpointGenerationLifecycle.HandoffPermit generationHandoff,
                WorkCapture precedingWork, int remainingQueueDepth, boolean removedFailedMember) {
            this.remainingQueueDepth = remainingQueueDepth;
            this.removedFailedMember = removedFailedMember;
            this.generationHandoff = generationHandoff;
            this.precedingWork = Objects.requireNonNull(precedingWork, "precedingWork");
        }

        /** Other reserved work at the same ownership boundary that committed this admission. */
        public WorkCapture precedingWork() {
            return precedingWork;
        }

        public int remainingQueueDepth() { return remainingQueueDepth; }
        public boolean removedFailedMember() { return removedFailedMember; }

        @Override
        public synchronized void close() {
            generationHandoff.close();
        }
    }

    /** Exact preparation capability, consumed by commit or rollback. */
    public abstract static class Reservation {
        final PrefillState owner;
        /* Guarded by owner.lock; null means committed, rolled back or retired. */
        RequestEntry originalOwner;

        private Reservation(PrefillState owner, RequestEntry originalOwner) {
            this.owner = owner;
            this.originalOwner = Objects.requireNonNull(
                    originalOwner, "originalOwner");
        }
    }

    public static final class RouteReservation extends Reservation {
        /* guarded by PrefillState.lock until the reservation commits */
        private final long predictedWorkMs;

        private RouteReservation(PrefillState owner, RequestEntry originalOwner, long predictedWorkMs) {
            super(owner, originalOwner);
            this.predictedWorkMs = Math.clamp(predictedWorkMs, 0L, (long) Integer.MAX_VALUE);
        }
    }

    public static final class BatchReservation extends Reservation {
        private final long batchId;

        private BatchReservation(PrefillState owner, RequestEntry originalOwner,
                                 long batchId,
                                 EndpointGenerationLifecycle.HandoffPermit generationHandoff) {
            super(owner, originalOwner);
            this.batchId = batchId;
            this.generationHandoff = Objects.requireNonNull(
                    generationHandoff, "generationHandoff");
        }

        /* guarded by PrefillState.lock; non-null only while OPEN */
        private EndpointGenerationLifecycle.HandoffPermit generationHandoff;

        /** Commit this exact batch lease while the caller holds ownershipLock(). */
        public CommittedHandoff commitLocked(
                List<RequestRoute> items,
                long predictedMs, RequestRoute failedMember, long nowMs) {
            return owner.commitBatchLocked(this, items, predictedMs, failedMember, nowMs);
        }
    }

    /** One committed Prefill batch: exact members, execution progress and accumulated results. */
    private static final class PrefillBatch {
        private final long batchId;
        private final Set<RequestEntry> members;
        private final long originalPredictionMs;
        private final PrefillBatchFeatures originalFeatures;
        private long remainingWorkMs;
        private Phase servicePhase = Phase.COMMITTED;
        private BatchOutcome outcome = new BatchOutcome();
        private long phaseBaseMs;
        private long lastObservedAtMs;

        private PrefillBatch(long batchId, Set<RequestEntry> members,
                          long predictedWorkMs,
                          PrefillBatchFeatures originalFeatures,
                          long nowMs) {
            checkArgument(batchId >= 0L, "batchId must be non-negative");
            checkArgument(predictedWorkMs >= 0L, "predicted batch work must be non-negative");
            this.batchId = batchId;
            this.members = members;
            this.originalPredictionMs = predictedWorkMs;
            this.originalFeatures = originalFeatures;
            this.remainingWorkMs = predictedWorkMs;
            this.phaseBaseMs = nowMs;
            this.lastObservedAtMs = nowMs;
        }

        private long remainingAt(long nowMs) {
            return remainingWorkAt(servicePhase, remainingWorkMs, phaseBaseMs, nowMs);
        }

        private void touch(long nowMs) {
            lastObservedAtMs = Math.max(lastObservedAtMs, nowMs);
        }

        private void updateExecutionProgress(Phase phase, long nowMs) {
            remainingWorkMs = remainingAt(nowMs);
            phaseBaseMs = Math.max(phaseBaseMs, nowMs);
            touch(nowMs);
            outcome.executionStarted |= phase == Phase.ENGINE_RUNNING;
            servicePhase = phase;
        }

        /** Retirement is external termination and must never train prediction. */
        private BatchCompletion retirementCompletion() {
            return new BatchCompletion(
                    batchId,
                    originalFeatures,
                    originalPredictionMs,
                    outcome.maxExecutionTimeMs,
                    outcome.successfulCompletion,
                    false);
        }
    }

    /** Resource ownership within this Prefill generation, independent of request protocol. */
    private enum OwnershipStage { QUEUED, DIRECT_RESERVED, COMMITTED, STOP_PENDING }

    /** Canonical resource facts for one exact route. Guarded by {@link #lock}. */
    private static final class RequestEntry {
        private OwnershipStage ownership;
        private final RequestRoute route;
        private Phase individualPhase;
        private long remainingWorkMs;
        private long phaseBaseMs;
        private PrefillBatch batch;
        private Reservation reservation;
        private RequestEntry(RequestRoute route, OwnershipStage ownership) {
            this.ownership = ownership;
            this.route = Objects.requireNonNull(route, "route");
        }

        private boolean isCommitted() {
            return ownership == OwnershipStage.COMMITTED;
        }

        private boolean isCommitCandidate(RequestRoute route) {
            return this.route == route && (ownership == OwnershipStage.QUEUED
                    || ownership == OwnershipStage.DIRECT_RESERVED);
        }

        private void commitIndividual(long predictedMs, long nowMs) {
            remainingWorkMs = Math.clamp(predictedMs, 0L, (long) Integer.MAX_VALUE);
            phaseBaseMs = nowMs;
            ownership = OwnershipStage.COMMITTED;
            individualPhase = Phase.COMMITTED;
            reservation = null;
        }

        private void updateExecutionProgress(Phase next, long nowMs) {
            checkArgument(next == Phase.ENGINE_QUEUED || next == Phase.ENGINE_RUNNING, "invalid Engine phase %s", next);
            if (!isCommitted() || batch != null) {
                throw new IllegalStateException(
                        "request is not individual request_id=" + route.requestId());
            }
            remainingWorkMs = remainingWorkAt(nowMs);
            phaseBaseMs = Math.max(phaseBaseMs, nowMs);
            individualPhase = next;
        }

        private long remainingWorkAt(long nowMs) {
            return PrefillState.remainingWorkAt(individualPhase, remainingWorkMs, phaseBaseMs, nowMs);
        }
    }

    public record ProjectionVersion(
            long queue,
            long schedulingInputs,
            long ownership) {
    }

    /** Queue and committed work captured at one ownership linearization point. */
    public record Snapshot(ProjectionVersion version, long capturedAtMs,
                           PrefillActiveIndex.Capture active,
                           WorkCapture work) {
        public Snapshot {
            Objects.requireNonNull(active, "missing active queue snapshot");
            Objects.requireNonNull(
                    work, "missing committed work snapshot");
        }
    }

    /** Immutable leaves copied under the ownership lock; projection runs outside it. */
    public static final class WorkCapture {
        private final long capturedAtMs;
        private List<WorkSnapshot.RequestWork> requests;
        private List<WorkSnapshot.BatchWork> batches;
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
                    snapshot = new WorkSnapshot(capturedAtMs, requests, batches, unknownRequestCount);
                    requests = null;
                    batches = null;
                    materialized = snapshot;
                }
                return materialized;
            }
        }
    }

    /** Terminal proof is bound to the canonical owner before any ledger mutation. */
    private record TerminalObservation(RequestEntry owner,
                                       long executionTimeMs,
                                       long errorCode,
                                       PriorityPreemptionProgress preemptionProgress) {
        private static TerminalObservation from(RequestEntry owner, WorkerStatus.TaskObservation task) {
            return new TerminalObservation(owner, task.executionTimeMs(), task.errorCode(),
                    task.priorityPreemptionProgress());
        }

        private TerminalObservation merge(TerminalObservation other) {
            checkArgument(owner == other.owner, "cannot merge different terminal owners");
            return new TerminalObservation(owner,
                    Math.max(executionTimeMs, other.executionTimeMs),
                    errorCode != 0L ? errorCode : other.errorCode,
                    PriorityPreemptionProgress.merge(preemptionProgress, other.preemptionProgress));
        }
    }

    private final ReentrantLock lock;
    /** Non-owning index containing only ACTIVE RequestRoute identities. */
    private PrefillActiveIndex activeIndex;
    /** Canonical request ownership, changed only under the endpoint lock. */
    private final Map<Long, RequestEntry> requests = new HashMap<>();
    private final LongSupplier clock;
    /** Monotonic ownership/work revision used by projection snapshots. */
    private volatile long mutationVersion;
    private volatile long schedulingInputVersion;
    /** Published at ownership mutation boundaries; admission still checks under the lock. */
    private volatile long outstandingRequestCount;
    private long requestCountsVersion = -1;
    private long[] requestCountsByPriority;
    /** Derived immutable work; ACTIVE queue mutations leave committed work unchanged. */
    private WorkCapture committedWorkCapture;
    private long unknownEngineRequestCount;
    private final Map<Long, BatchReservation> preparedBatches = new HashMap<>();
    private final Map<Long, PrefillBatch> batches = new HashMap<>();

    /** Publish the capacity summary before readers observe a new ownership revision. */
    private void recordMutationLocked() {
        publishRequestCountLocked();
        mutationVersion++;
    }

    private void publishRequestCountLocked() {
        requireLock();
        long count = saturatedAdd(requests.size(), unknownEngineRequestCount);
        if (outstandingRequestCount != count) {
            outstandingRequestCount = count;
        }
    }

    public PrefillState(ReentrantLock lock, PrefillActiveIndex activeIndex) {
        this(lock, activeIndex, System::currentTimeMillis);
    }

    public PrefillState(ReentrantLock lock, PrefillActiveIndex activeIndex,
                        LongSupplier clock) {
        this.lock = Objects.requireNonNull(lock, "lock");
        this.activeIndex = Objects.requireNonNull(activeIndex, "activeIndex");
        this.clock = Objects.requireNonNull(clock, "clock");
    }

    /** Shared lock for atomic queue validation, delivery commit and condition waits. */
    public ReentrantLock ownershipLock() { return lock; }

    /** Enable waiting work without replacing the ledger of existing DIRECT reservations. */
    void enableQueueLocked(Comparator<RequestRoute> ordering) {
        requireLock();
        if (activeIndex == PrefillActiveIndex.disabled()) {
            activeIndex = PrefillActiveIndex.ordered(16, ordering);
            schedulingInputsChangedLocked();
            recordMutationLocked();
        }
    }

    /** One bounded, ordered queue view; all metadata belongs to the same ownership revision. */
    public record QueueSnapshot(long queueVersion, long schedulingInputVersion, List<RequestRoute> items) {
        public RequestRoute head() {
            return items.isEmpty() ? null : items.get(0);
        }
    }

    /** Diagnostic counters captured from one queue ownership revision. */
    public record QueueCounters(long version, int[] byPriority, long outstandingRequests, int batchSlots) { }

    public QueueCounters captureQueueCounters() {
        lock.lock();
        try {
            int[] byPriority = new int[PriorityNormalizer.MAX_PRIORITY + 1];
            for (int priority = 0; priority < byPriority.length; priority++) {
                byPriority[priority] = activeIndex.size(priority);
            }
            return new QueueCounters(activeIndex.version(), byPriority,
                    saturatedAdd(requests.size(), unknownEngineRequestCount), (preparedBatches.size() + batches.size()));
        } finally {
            lock.unlock();
        }
    }

    /** Current queue depth for the endpoint's status view. */
    public int queueDepth() {
        lock.lock();
        try {
            return activeIndex.size();
        } finally {
            lock.unlock();
        }
    }

    public QueueSnapshot captureQueue(int limit) {
        checkArgument(limit > 0, "queue capture limit must be positive");
        lock.lock();
        try {
            List<RequestRoute> items = new ArrayList<>(Math.min(limit, activeIndex.size()));
            var ordered = activeIndex.iterator();
            while (items.size() < limit && ordered.hasNext()) {
                items.add(ordered.next());
            }
            return new QueueSnapshot(activeIndex.version(), schedulingInputVersion(), List.copyOf(items));
        } finally {
            lock.unlock();
        }
    }

    long schedulingInputVersion() { return schedulingInputVersion; }

    public void schedulingInputsChangedLocked() {
        requireLock();
        schedulingInputVersion++;
    }

    private boolean removeRequestLocked(long requestId, RequestEntry entry) {
        requireLock();
        if (!requests.remove(requestId, entry)) {
            return false;
        }
        if (entry.ownership == OwnershipStage.DIRECT_RESERVED || entry.isCommitted()) {
            committedWorkCapture = null;
        }
        return true;
    }

    /** Advisory check; callers recheck under ownershipLock before replacing a projection. */
    public boolean isCurrentProjection(ProjectionVersion version) {
        return version.queue() == activeIndex.version()
                && version.schedulingInputs() == schedulingInputVersion
                && version.ownership() == mutationVersion;
    }

    /** Advisory revision read for the lock-free projection-cache fast path. */
    public long mutationVersion() {
        return mutationVersion;
    }

    public boolean enqueueActiveLocked(RequestRoute item, long maxOutstandingRequests) {
        requireLock();
        if (requests.containsKey(item.requestId())
                || (maxOutstandingRequests > 0L && !canAcceptRequestLocked(maxOutstandingRequests))) {
            return false;
        }
        insertQueuedRouteLocked(item);
        return true;
    }

    /** Caller holds the ownership lock and has checked that the request ID is available. */
    private void insertQueuedRouteLocked(RequestRoute item) {
        RequestEntry entry = new RequestEntry(item, OwnershipStage.QUEUED);
        requests.put(item.requestId(), entry);
        try {
            activeIndex.add(item);
        } catch (RuntimeException | Error failure) {
            removeRequestLocked(item.requestId(), entry);
            throw failure;
        }
        recordMutationLocked();
    }

    public boolean ownsSelectionLocked(List<RequestRoute> items, long nowMs) {
        requireLock();
        for (RequestRoute item : items) {
            if (!activeIndex.contains(item) || item.requestExpired(nowMs)) {
                return false;
            }
        }
        return true;
    }

    /** Remove an exact, still waiting and unexpired queue owner in the current transaction. */
    public boolean removeQueuedIfUnexpiredLocked(RequestRoute exact, long nowMs) {
        requireLock();
        return exact != null && !exact.requestExpired(nowMs) && removeQueuedLocked(exact);
    }

    /** The queue part of a worker wait; the worker owns stop and control wakeups. */
    public boolean queueWaitCurrentLocked(
            RequestRoute head, long queueVersion, long inputVersion,
            boolean capacityWait, long nowMs) {
        requireLock();
        if (head == null) {
            return activeIndex.isEmpty();
        }
        if (activeIndex.peek() != head) {
            return false;
        }
        return capacityWait
                ? !head.requestExpired(nowMs)
                : activeIndex.version() == queueVersion && schedulingInputVersion == inputVersion;
    }

    /** Remove exact ACTIVE ownership; batch preparation retains its OPEN lease. */
    public boolean removeQueuedLocked(RequestRoute item) {
        requireLock();
        RequestEntry entry = requests.get(item.requestId());
        if (entry == null || entry.route != item || entry.ownership != OwnershipStage.QUEUED) {
            return false;
        }
        Reservation lease = entry.reservation;
        checkState(lease == null || lease.originalOwner != null,
                "ACTIVE request owns a non-OPEN Prefill lease request_id=%s", item.requestId());
        removeValidatedActiveIndexLocked(item);
        // Batch preparation still owns its OPEN lease and generation handoff.
        removeRequestLocked(item.requestId(), entry);
        recordMutationLocked();
        return true;
    }

    /** Only failed admission needs priority provenance; reuse one summary per revision. */
    public AdmissionSummary admissionSummary(int priority, long requestLimit) {
        long[] counts;
        long occupiedRequests;
        lock.lock();
        try {
            if (requestCountsByPriority == null || requestCountsVersion != mutationVersion) {
                long[] updatedCounts = new long[PriorityNormalizer.MAX_PRIORITY + 1];
                updatedCounts[0] = unknownEngineRequestCount;
                for (RequestEntry entry : requests.values()) {
                    RequestRoute item = entry.route;
                    int occupant = item.priority();
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
        long lower = 0L, same = 0L, higher = 0L;
        for (int occupant = 1; occupant <= PriorityNormalizer.MAX_PRIORITY; occupant++) {
            if (occupant < priority) { lower += counts[occupant]; }
            else if (occupant == priority) { same += counts[occupant]; } else { higher += counts[occupant]; }
        }
        return new AdmissionSummary(occupiedRequests,
                requestLimit > 0 ? Math.max(0L, occupiedRequests + 1L - requestLimit) : 0L,
                lower, same, higher, counts[0]);
    }

    boolean hasUncommittedQueuedRequestsBelowPriorityLocked(int priority, long required) {
        requireLock();
        if (required <= 0L || required > activeIndex.size()) { return false; }
        for (RequestRoute item : activeIndex) {
            int occupantPriority = item.priority();
            if (PriorityNormalizer.hasPriority(occupantPriority) && occupantPriority < priority
                    && isUncommittedQueuedRequest(item) && --required == 0L) { return true; }
        }
        return false;
    }

    List<RequestRoute> uncommittedQueuedRoutesLocked() {
        requireLock();
        List<RequestRoute> candidates = new ArrayList<>();
        for (RequestRoute item : activeIndex) {
            if (isUncommittedQueuedRequest(item)) { candidates.add(item); }
        }
        return candidates;
    }

    private boolean isUncommittedQueuedRequest(RequestRoute item) {
        requireLock();
        RequestEntry entry = requests.get(item.requestId());
        return entry != null && entry.route == item
                && entry.ownership == OwnershipStage.QUEUED && entry.reservation == null;
    }

    long requestSlotsToReleaseLocked(long requestLimit) {
        requireLock();
        return requestLimit <= 0L ? 0L
                : Math.max(0L, saturatedAdd(requests.size(), unknownEngineRequestCount) - requestLimit + 1L);
    }

    /** Revalidate and replace exact uncommitted owners under the shared queue lock. */
    boolean replaceQueuedRoutesLocked(RequestRoute incoming, long requestLimit, List<RequestRoute> victims) {
        requireLock();
        long required = requestSlotsToReleaseLocked(requestLimit);
        if (requests.containsKey(incoming.requestId()) || required == 0L || victims.size() != required) { return false; }
        Set<RequestRoute> distinct = java.util.Collections.newSetFromMap(new IdentityHashMap<>());
        for (RequestRoute victim : victims) {
            if (!distinct.add(victim) || !isUncommittedQueuedRequest(victim)) { return false; }
        }
        // The selected victims fund this seat; the shared lock hides the temporary excess.
        insertQueuedRouteLocked(incoming);
        for (RequestRoute victim : victims) {
            checkState(removeQueuedLocked(victim), "queued preemption lost its exact victim");
        }
        return true;
    }

    /**
     * Detach one exact queue head for the stop callback without discarding its
     * canonical request owner. A failed callback therefore remains visible to
     * generation retirement, while a successful callback must explicitly
     * acknowledge the exact pending identity below.
     */
    public RequestRoute detachNextActiveForStop() {
        lock.lock();
        try {
            RequestRoute item = activeIndex.peek();
            if (item == null) { return null; }
            RequestEntry entry = requests.get(item.requestId());
            checkState(entry != null && entry.isCommitCandidate(item),
                    "stopped queue head has no canonical ACTIVE owner request_id=%s", item.requestId());
            Reservation lease = entry.reservation;
            checkState(lease == null || lease.originalOwner != null,
                    "stopped ACTIVE request owns a non-OPEN Prefill lease request_id=%s", item.requestId());
            removeValidatedActiveIndexLocked(item);
            entry.ownership = OwnershipStage.STOP_PENDING;
            recordMutationLocked();
            return item;
        } finally {
            lock.unlock();
        }
    }

    /** Remove only the exact stop-pending owner whose callback completed. */
    public boolean acknowledgeStopTerminalLocked(RequestRoute item) {
        requireLock();
        RequestEntry entry = requests.get(item.requestId());
        if (entry == null) { return true; }
        if (entry.route != item || entry.ownership != OwnershipStage.STOP_PENDING
                || activeIndex.contains(item)) {
            return false;
        }
        boolean removed = removeRequestLocked(item.requestId(), entry);
        if (removed) {
            recordMutationLocked();
        }
        return removed;
    }

    ReservationResult<BatchReservation> reserveBatch(
            RequestRoute exactHead,
            long batchId,
            int maximum,
            EndpointGenerationLifecycle.HandoffPermit generationHandoff) {
        requirePositiveBatchLimit(maximum);
        RequestRoute head = exactHead;
        lock.lock();
        try {
            RequestEntry entry = requests.get(head.requestId());
            if (entry == null || !entry.isCommitCandidate(head)) {
                return new ReservationResult<>(
                        CapacityStatus.REQUEST_NOT_ACTIVE, null);
            }
            if (entry.reservation != null) {
                return new ReservationResult<>(
                        CapacityStatus.REQUEST_ALREADY_RESERVED, null);
            }
            if (preparedBatches.containsKey(batchId) || batches.containsKey(batchId)) {
                return new ReservationResult<>(
                        CapacityStatus.BATCH_ID_ALREADY_RESERVED, null);
            }
            if ((preparedBatches.size() + batches.size()) >= maximum) {
                return new ReservationResult<>(
                        CapacityStatus.CAPACITY_FULL, null);
            }
            BatchReservation lease = new BatchReservation(
                    this, entry, batchId, generationHandoff);
            preparedBatches.put(batchId, lease);
            entry.reservation = lease;
            recordMutationLocked();
            return new ReservationResult<>(CapacityStatus.ACQUIRED, lease);
        } finally {
            lock.unlock();
        }
    }

    public boolean batchCapacityAvailable(int maximum) {
        requirePositiveBatchLimit(maximum);
        lock.lock();
        try {
            return (preparedBatches.size() + batches.size()) < maximum;
        } finally {
            lock.unlock();
        }
    }

    private static void requirePositiveBatchLimit(int maximum) {
        checkArgument(maximum > 0, "maximumInflightBatches must be positive");
    }

    CommittedHandoff commitRouteGroupLocked(
            List<RequestRoute> items,
            List<RouteReservation> exactReservations,
            EndpointGenerationLifecycle.HandoffPermit generationHandoff) {
        requireLock();
        Objects.requireNonNull(generationHandoff, "generationHandoff");
        checkArgument(!exactReservations.isEmpty(), "route commit requires at least one reservation");
        List<RequestEntry> members = validateGroupLocked(items, OwnershipStage.DIRECT_RESERVED);
        checkArgument(items.size() == exactReservations.size(), "route commit requires one exact lease per member");
        for (int index = 0; index < items.size(); index++) {
            RequestEntry entry = members.get(index);
            RouteReservation lease = exactReservations.get(index);
            checkArgument(lease != null && lease.owner == this, "route reservation belongs to another Prefill ledger");
            if (entry.reservation != lease
                    || lease.originalOwner != entry) {
                throw new IllegalStateException(
                        "route commit does not own exact OPEN lease request_id="
                                + items.get(index).requestId());
            }
        }
        long nowMs = clock.getAsLong();
        CommittedHandoff committedHandoff = new CommittedHandoff(generationHandoff,
                captureCurrentWorkLocked(nowMs, Set.copyOf(members)), activeIndex.size(), false);
        // Every exact DIRECT token is validated before any ownership changes.
        for (int index = 0; index < items.size(); index++) {
            RequestEntry entry = members.get(index);
            RouteReservation lease = exactReservations.get(index);
            entry.commitIndividual(lease.predictedWorkMs, nowMs);
            lease.originalOwner = null;
        }
        committedWorkCapture = null;
        recordMutationLocked();
        return committedHandoff;
    }

    /** Queued predictions belong to the transaction until exact queue ownership commits. */
    CommittedHandoff commitQueuedRoutesLocked(
            List<RequestRoute> items, long[] predictions,
            EndpointGenerationLifecycle.HandoffPermit generationHandoff, RequestRoute failedMember, long selectionTimeMs) {
        requireLock();
        Objects.requireNonNull(generationHandoff, "generationHandoff");
        checkArgument(items.size() == predictions.length, "route commit requires one prediction per member");
        List<RequestEntry> members = validateGroupLocked(items, OwnershipStage.QUEUED);
        for (RequestEntry member : members) {
            checkState(member.reservation == null,
                    "queued route commit cannot consume another preparation request_id=%s", member.route.requestId());
        }
        RequestEntry failed = failedSelectionMemberLocked(failedMember, members, selectionTimeMs);
        long nowMs = clock.getAsLong();
        CommittedHandoff committedHandoff = new CommittedHandoff(generationHandoff, captureWorkLocked(nowMs),
                activeIndex.size() - members.size() - (failed == null ? 0 : 1), failed != null);
        for (int index = 0; index < items.size(); index++) {
            RequestEntry member = members.get(index);
            removeValidatedActiveIndexLocked(member.route);
            member.commitIndividual(predictions[index], nowMs);
        }
        removeFailedSelectionMemberLocked(failed);
        committedWorkCapture = null;
        recordMutationLocked();
        return committedHandoff;
    }

    private CommittedHandoff commitBatchLocked(
            BatchReservation lease,
            List<RequestRoute> items,
            long predictedMs, RequestRoute failedMember, long selectionTimeMs) {
        requireLock();
        if (!ownsSelectionLocked(items, selectionTimeMs)) { return null; }
        List<RequestEntry> members = validateGroupLocked(items, OwnershipStage.QUEUED);
        RequestEntry head = lease.originalOwner;
        checkState(head != null && members.contains(head),
                "batch commit lost its exact head batch_id=%s", lease.batchId);
        checkState(head.reservation == lease && preparedBatches.get(lease.batchId) == lease,
                "batch commit does not own exact OPEN lease batch_id=%s", lease.batchId);
        checkState(lease.generationHandoff != null, "batch preparation lost its handoff");
        for (RequestEntry member : members) {
            Reservation expected = member == head ? lease : null;
            checkState(member.reservation == expected,
                    "batch member owns another exact reservation request_id=%s", member.route.requestId());
        }
        RequestEntry failed = failedSelectionMemberLocked(failedMember, members, selectionTimeMs);
        long nowMs = clock.getAsLong();
        PrefillBatch batch = new PrefillBatch(
                lease.batchId, new HashSet<>(members),
                predictedMs,
                PrefillBatchFeatures.from(
                        items,
                        RequestRoute::seqLen,
                        RequestRoute::hitCache),
                nowMs);
        CommittedHandoff committedHandoff = new CommittedHandoff(lease.generationHandoff,
                captureWorkLocked(nowMs), activeIndex.size() - members.size() - (failed == null ? 0 : 1), failed != null);
        batches.put(lease.batchId, batch);
        preparedBatches.remove(lease.batchId);
        lease.generationHandoff = null;
        lease.originalOwner = null;
        for (RequestEntry member : members) {
            removeValidatedActiveIndexLocked(member.route);
            member.batch = batch;
            member.reservation = null;
            member.ownership = OwnershipStage.COMMITTED;
        }
        removeFailedSelectionMemberLocked(failed);
        committedWorkCapture = null;
        recordMutationLocked();
        return committedHandoff;
    }

    /** Validate the failed boundary before committing any member or consuming a preparation. */
    private RequestEntry failedSelectionMemberLocked(RequestRoute route, List<RequestEntry> members, long nowMs) {
        if (route == null || route.requestExpired(nowMs)) { return null; }
        RequestEntry entry = requests.get(route.requestId());
        if (entry == null || entry.route != route || entry.ownership != OwnershipStage.QUEUED) { return null; }
        checkState(!members.contains(entry), "failed member belongs to committed selection");
        checkState(activeIndex.contains(route), "failed member lost queue index");
        checkState(entry.reservation == null || entry.reservation.originalOwner != null,
                "failed member owns a consumed reservation");
        return entry;
    }

    private void removeFailedSelectionMemberLocked(RequestEntry failed) {
        if (failed == null) { return; }
        removeValidatedActiveIndexLocked(failed.route);
        removeRequestLocked(failed.route.requestId(), failed);
    }

    /** Resolve exact members once; every caller validates the whole group before mutating it. */
    private List<RequestEntry> validateGroupLocked(List<RequestRoute> items, OwnershipStage ownership) {
        requireLock();
        checkState(!items.isEmpty(), "committed group requires members");
        Set<RequestEntry> unique = java.util.Collections.newSetFromMap(new IdentityHashMap<>());
        List<RequestEntry> members = new ArrayList<>(items.size());
        for (RequestRoute item : items) {
            RequestEntry entry = requests.get(item.requestId());
            checkState(entry != null && entry.route == item && entry.ownership == ownership,
                    "group member is not canonical %s request_id=%s", ownership, item.requestId());
            checkState(unique.add(entry), "duplicate group member request_id=%s", item.requestId());
            checkState(activeIndex.contains(item) == (ownership == OwnershipStage.QUEUED),
                    "group member has inconsistent queue ownership request_id=%s", item.requestId());
            members.add(entry);
        }
        return members;
    }

    private void removeValidatedActiveIndexLocked(RequestRoute item) {
        requireLock();
        boolean removed = activeIndex.remove(item);
        checkState(removed,
                "validated ACTIVE queue index disappeared request_id=%s", item.requestId());
    }

    /**
     * Reserve immediate route work against current ownership in one transaction.
     * Selection revisions are advisory; only exact identity and current capacity
     * decide admission. A zero request limit leaves count admission disabled.
     */
    public ReservationResult<RouteReservation> reserveUnqueuedRoute(
            RequestRoute item, long predictedMs, long maxOutstandingRequests) {
        Objects.requireNonNull(item, "item");
        checkArgument(maxOutstandingRequests >= 0L, "request limit must be non-negative");
        lock.lock();
        try {
            if (requests.containsKey(item.requestId())) {
                return new ReservationResult<>(CapacityStatus.REQUEST_ALREADY_RESERVED, null);
            }
            if (maxOutstandingRequests > 0L && !canAcceptRequestLocked(maxOutstandingRequests)) {
                return new ReservationResult<>(CapacityStatus.CAPACITY_FULL, null);
            }
            RequestEntry entry = new RequestEntry(item, OwnershipStage.DIRECT_RESERVED);
            RouteReservation reservation = new RouteReservation(this, entry, predictedMs);
            ReservationResult<RouteReservation> result = new ReservationResult<>(CapacityStatus.ACQUIRED, reservation);
            entry.reservation = reservation;
            requests.put(item.requestId(), entry);
            committedWorkCapture = null;
            recordMutationLocked();
            return result;
        } finally {
            lock.unlock();
        }
    }

    enum RequestRelease { NONE, QUEUED, RESERVED, COMMITTED }

    /** Release only this exact request; preparations owned by a batch transaction survive. */
    RequestRelease releaseRequest(RequestRoute exactItem) {
        lock.lock();
        try {
            RequestEntry entry = requests.get(exactItem.requestId());
            if (entry == null || entry.route != exactItem) { return RequestRelease.NONE; }
            return switch (entry.ownership) {
                case COMMITTED -> {
                    releaseCommittedLocked(entry, clock.getAsLong());
                    yield RequestRelease.COMMITTED;
                }
                case DIRECT_RESERVED -> {
                    checkState(entry.reservation instanceof RouteReservation && entry.reservation.originalOwner != null,
                            "unqueued admission has no preparation request_id=%s", exactItem.requestId());
                    closeOpenLeaseLocked(entry.reservation);
                    yield RequestRelease.RESERVED;
                }
                case QUEUED, STOP_PENDING -> {
                    if (entry.ownership == OwnershipStage.QUEUED) {
                        removeValidatedActiveIndexLocked(exactItem);
                    }
                    // A stopped route still owns its seat until cleanup or callback acknowledgement.
                    removeRequestLocked(exactItem.requestId(), entry);
                    recordMutationLocked();
                    yield RequestRelease.QUEUED;
                }
            };
        } finally { lock.unlock(); }
    }

    /** Heartbeats renew activity but never consume terminal reports. */
    public StatusReconciliation reconcileHeartbeat(WorkerStatus.StatusObservation observation) {
        lock.lock();
        try {
            StatusReduction reduction = prepareReductionLocked(observation, false);
            return commitStatusLocked(reduction, Map.of());
        } finally {
            lock.unlock();
        }
    }

    /** Facts prepared under the lock, then read during prediction and exact-version commit. */
    public static final class StatusReduction {
        private final PrefillState owner;
        private long version;
        private final Map<Long, TerminalObservation> terminals;
        private final Map<RequestEntry, Phase> individualPhases = new IdentityHashMap<>();
        private final Map<PrefillBatch, Phase> batchPhases = new IdentityHashMap<>();
        private final Map<PrefillBatch, BatchOutcome> batchOutcomes;
        // Assigned before publication; only the owning State applies the facts under its lock.
        private long nowMs;
        private long unknownRequests;
        private Map<Long, List<RequestRoute>> predictionInputs;
        private StatusReconciliation result;

        private StatusReduction(PrefillState owner, Map<Long, TerminalObservation> terminals) {
            this.owner = owner;
            this.terminals = terminals;
            this.batchOutcomes = terminals.isEmpty() ? Map.of() : new IdentityHashMap<>();
        }

        public Map<Long, List<RequestRoute>> predictionInputs() { return predictionInputs; }
    }

    /** Owned by one batch or by an uncommitted status reduction, never both while preparing. */
    private static final class BatchOutcome {
        private long maxExecutionTimeMs;
        private boolean successfulCompletion;
        private boolean learningEligible = true;
        /** Historical execution evidence survives a later queued observation. */
        private boolean executionStarted;

        private BatchOutcome copy() {
            BatchOutcome copy = new BatchOutcome();
            copy.maxExecutionTimeMs = maxExecutionTimeMs;
            copy.successfulCompletion = successfulCompletion;
            copy.learningEligible = learningEligible;
            copy.executionStarted = executionStarted;
            return copy;
        }

        private void include(TerminalObservation terminal) {
            boolean succeeded = terminal.errorCode == 0L;
            executionStarted |= succeeded || terminal.executionTimeMs > 0L;
            maxExecutionTimeMs = Math.max(maxExecutionTimeMs, terminal.executionTimeMs);
            successfulCompletion |= succeeded;
            learningEligible &= succeeded;
        }
    }

    private static long remainingWorkAt(Phase phase, long remainingWorkMs, long phaseBaseMs, long nowMs) {
        if (phase != Phase.ENGINE_RUNNING) {
            return remainingWorkMs;
        }
        long elapsedMs = Math.max(0L, nowMs - phaseBaseMs);
        return Math.max(0L, remainingWorkMs - elapsedMs);
    }

    /** Capture exact activity into the same transaction that will commit it. */
    private void prepareActiveObservationsLocked(WorkerStatus.EngineObservation engine,
                                                StatusReduction reduction,
                                                List<PrefillRequestStatus> activeRequestStatuses) {
        requireLock();
        Set<Long> unknownDetailed = new HashSet<>();
        Set<Long> knownObserved = new HashSet<>();
        for (WorkerStatus.TaskObservation task : engine.runningTaskList().values()) {
            if (reduction.terminals.containsKey(task.requestId())) {
                continue;
            }
            RequestEntry entry = requests.get(task.requestId());
            if (entry == null || !matchesObservedBatch(entry, task.batchId())) {
                if (!task.isPriorityCancelOverlayOnly()) {
                    unknownDetailed.add(task.requestId());
                }
                continue;
            }
            if (!task.isPriorityCancelOverlayOnly()) {
                knownObserved.add(task.requestId());
                if (entry.isCommitted()) {
                    activeRequestStatuses.add(PrefillRequestStatus.active(entry.route));
                }
            }
            if (!entry.isCommitted()) {
                continue;
            }
            Phase observed = task.phase() == TaskPhase.RUNNING
                    ? Phase.ENGINE_RUNNING : Phase.ENGINE_QUEUED;
            if (entry.batch == null) {
                reduction.individualPhases.put(entry, observed);
            } else {
                reduction.batchPhases.merge(
                        entry.batch,
                        observed,
                        PrefillState::strongerEnginePhase);
            }
        }
        long reportedActive = saturatedAdd(Math.max(0L, engine.waitingQueryLen()),
                Math.max(0L, engine.runningQueryLen()));
        long scalarUnknown = Math.max(0L, reportedActive - knownObserved.size());
        reduction.unknownRequests = Math.max(unknownDetailed.size(), scalarUnknown);
    }

    /** Full status captures its execution clock before scanning Worker terminal and activity facts. */
    public StatusReduction prepareStatusLocked(WorkerStatus.StatusObservation observation) {
        requireLock();
        return prepareReductionLocked(observation, true);
    }

    private StatusReduction prepareReductionLocked(WorkerStatus.StatusObservation observation, boolean fullStatus) {
        requireLock();
        long nowMs = fullStatus ? clock.getAsLong() : 0L;
        StatusReduction reduction = new StatusReduction(this,
                fullStatus ? terminalObservationsLocked(observation.finishedTasks()) : Map.of());
        List<PrefillRequestStatus> requestStatuses = new ArrayList<>(reduction.terminals.size()
                + observation.engine().runningTaskList().size());
        for (TerminalObservation terminal : reduction.terminals.values()) {
            requestStatuses.add(terminalRequestStatus(terminal));
            PrefillBatch batch = terminal.owner.batch;
            if (batch != null) {
                reduction.batchOutcomes.computeIfAbsent(batch, ignored -> batch.outcome.copy()).include(terminal);
            }
        }
        prepareActiveObservationsLocked(observation.engine(), reduction, requestStatuses);
        boolean capacityReleased = !reduction.terminals.isEmpty()
                || reduction.unknownRequests < unknownEngineRequestCount;
        // Heartbeats historically measure time after the activity scan; keep that boundary.
        reduction.nowMs = fullStatus ? nowMs : clock.getAsLong();
        Map<Long, List<RequestRoute>> predictionInputs = reduction.batchOutcomes.isEmpty() ? Map.of() : new HashMap<>();
        List<BatchCompletion> completions = reduction.batchOutcomes.isEmpty() ? List.of() : new ArrayList<>(reduction.batchOutcomes.size());
        for (var change : reduction.batchOutcomes.entrySet()) {
            PrefillBatch batch = change.getKey();
            BatchOutcome outcome = change.getValue();
            List<RequestRoute> survivors = new ArrayList<>(batch.members.size());
            for (RequestEntry member : batch.members) {
                if (!reduction.terminals.containsKey(member.route.requestId())) {
                    survivors.add(member.route);
                }
            }
            if (survivors.isEmpty()) {
                completions.add(new BatchCompletion(batch.batchId, batch.originalFeatures,
                        batch.originalPredictionMs, outcome.maxExecutionTimeMs,
                        outcome.successfulCompletion, outcome.learningEligible));
            } else if (!outcome.executionStarted && reduction.batchPhases.get(batch) != Phase.ENGINE_RUNNING) {
                survivors.sort(Comparator.comparingLong(RequestRoute::enqueueSeq).thenComparingLong(RequestRoute::requestId));
                predictionInputs.put(batch.batchId, List.copyOf(survivors));
            }
        }
        boolean activityChanged = reduction.unknownRequests != unknownEngineRequestCount;
        for (var phase : reduction.individualPhases.entrySet()) {
            activityChanged |= phase.getKey().individualPhase != phase.getValue();
        }
        for (var phase : reduction.batchPhases.entrySet()) {
            activityChanged |= phase.getKey().servicePhase != phase.getValue();
        }
        reduction.version = mutationVersion;
        reduction.predictionInputs = Map.copyOf(predictionInputs);
        reduction.result = new StatusReconciliation(requestStatuses, completions,
                activityChanged || !reduction.terminals.isEmpty() || !predictionInputs.isEmpty(), capacityReleased);
        return reduction;
    }

    /** Null means the out-of-lock prediction was invalidated; no fact has changed. */
    public StatusReconciliation commitStatusLocked(StatusReduction reduction, Map<Long, Long> predictions) {
        requireLock();
        checkArgument(reduction.owner == this, "Status reduction belongs to another State");
        if (reduction.version != mutationVersion) { return null; }
        checkArgument(predictions.keySet().equals(reduction.predictionInputs.keySet()),
                "Predictions do not match the prepared batches");
        for (long prediction : predictions.values()) {
            checkArgument(prediction >= 0L, "Negative batch prediction");
        }
        // The version check also validates the capacity facts captured in the prepared result.
        for (var change : reduction.batchOutcomes.entrySet()) {
            PrefillBatch batch = change.getKey();
            batch.outcome = change.getValue();
            batch.touch(reduction.nowMs);
        }
        for (TerminalObservation terminal : reduction.terminals.values()) { removeCommittedLocked(terminal.owner); }
        for (var phase : reduction.individualPhases.entrySet()) {
            phase.getKey().updateExecutionProgress(phase.getValue(), reduction.nowMs);
        }
        for (var phase : reduction.batchPhases.entrySet()) {
            phase.getKey().updateExecutionProgress(phase.getValue(), reduction.nowMs);
        }
        unknownEngineRequestCount = reduction.unknownRequests;
        for (long batchId : reduction.predictionInputs.keySet()) {
            PrefillBatch batch = batches.get(batchId);
            batch.remainingWorkMs = predictions.get(batchId);
            batch.phaseBaseMs = reduction.nowMs;
            batch.touch(reduction.nowMs);
        }
        if (reduction.result.schedulingInputsChanged()) {
            committedWorkCapture = null;
            recordMutationLocked();
        }
        return reduction.result;
    }

    /**
     * End every owner for this retired Prefill generation in one queue-lock
     * transaction. All callback facts and completion DTOs are constructed
     * before the canonical table, queue index, and capacity counters are
     * cleared. Once clearing starts, the remaining operations are allocation-
     * free field updates; no failure can leave a partially retired registry.
     *
     * <p>Ordinarily the worker batcher has already reduced every ACTIVE item.
     * Including a defensively remaining ACTIVE identity here makes
     * endpoint close total if that earlier invariant check failed.</p>
     */
    public Retirement retireGenerationOwnership() {
        List<EndpointGenerationLifecycle.HandoffPermit> orphanedHandoffs = new ArrayList<>();
        Retirement retirement;
        lock.lock();
        try {
            List<RequestRoute> ownedItems = requests.values().stream().map(entry -> entry.route)
                    .sorted(Comparator.comparingLong(RequestRoute::enqueueSeq).thenComparingLong(RequestRoute::requestId)).toList();
            List<BatchCompletion> completions = batches.values().stream()
                    .map(PrefillBatch::retirementCompletion).sorted(Comparator.comparingLong(BatchCompletion::batchId)).toList();
            Throwable invariantFailure = null;
            for (BatchReservation reservation : preparedBatches.values()) {
                if (reservation.generationHandoff != null) {
                    orphanedHandoffs.add(reservation.generationHandoff);
                }
                invariantFailure = Failures.append(invariantFailure,
                        new IllegalStateException("retirement reached an OPEN Prefill batch lease"));
            }
            retirement = new Retirement(ownedItems, completions, invariantFailure, orphanedHandoffs);
            for (RequestEntry entry : requests.values()) {
                if (entry.reservation != null) { entry.reservation.originalOwner = null; }
                entry.reservation = null;
            }
            for (BatchReservation reservation : preparedBatches.values()) {
                reservation.generationHandoff = null;
                reservation.originalOwner = null;
            }
            for (PrefillBatch batch : batches.values()) { batch.members.clear(); }
            requests.clear();
            preparedBatches.clear();
            batches.clear();
            activeIndex.clear();
            unknownEngineRequestCount = 0L;
            committedWorkCapture = null;
            recordMutationLocked();
        } finally {
            lock.unlock();
        }
        return retirement;
    }

    /** Capture exact committed identities before the workflow checks request ownership outside this lock. */
    public List<RequestRoute> cleanupCandidates() {
        lock.lock();
        try { return requests.values().stream().filter(entry -> entry.isCommitted()).map(entry -> entry.route).toList(); }
        finally { lock.unlock(); }
    }

    /** One orphan pass: a retained member protects its whole batch. Counts batches and individuals. */
    public int evictExpiredInflight(long ttlMs, Set<RequestRoute> orphanCandidates) {
        int evicted = 0;
        lock.lock();
        try {
            long nowMs = clock.getAsLong();
            long ttl = Math.max(0L, ttlMs);
            for (PrefillBatch batch : List.copyOf(batches.values())) {
                Set<RequestEntry> members = batch.members;
                boolean retained = nowMs - batch.lastObservedAtMs < ttl;
                for (RequestEntry entry : members) {
                    retained |= !orphanCandidates.contains(entry.route);
                }
                if (retained) { continue; }
                for (RequestEntry entry : List.copyOf(members)) {
                    releaseCommittedLocked(entry, nowMs);
                }
                evicted++;
            }
            List<RequestEntry> individuals = new ArrayList<>();
            for (RequestEntry entry : requests.values()) {
                if (entry.batch == null && entry.isCommitted()
                        && nowMs - entry.phaseBaseMs >= ttl
                        && orphanCandidates.contains(entry.route)) {
                    individuals.add(entry);
                }
            }
            for (RequestEntry entry : individuals) {
                releaseCommittedLocked(entry, nowMs);
            }
            evicted += individuals.size();
        } finally {
            lock.unlock();
        }
        return evicted;
    }

    public Stats stats() {
        lock.lock();
        try {
            int locallyOwned = 0;
            int individual = 0;
            long maxAgeMs = 0L;
            long nowMs = clock.getAsLong();
            for (RequestEntry entry : requests.values()) {
                if (!entry.isCommitted()) {
                    continue;
                }
                locallyOwned++;
                if (entry.batch == null) {
                    individual++;
                    maxAgeMs = Math.max(
                            maxAgeMs,
                            Math.max(0L, nowMs - entry.phaseBaseMs));
                }
            }
            for (PrefillBatch batch : batches.values()) {
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

    private boolean canAcceptRequestLocked(long maxOutstandingRequests) {
        requireLock();
        return requests.size() < maxOutstandingRequests
                && unknownEngineRequestCount < maxOutstandingRequests - requests.size();
    }

    Snapshot snapshotLocked() {
        requireLock();
        ProjectionVersion version = new ProjectionVersion(
                activeIndex.version(), schedulingInputVersion, mutationVersion);
        long nowMs = clock.getAsLong();
        return new Snapshot(
                version, nowMs,
                activeIndex.capture(),
                captureWorkLocked(nowMs));
    }

    public WorkSnapshot committedSnapshot() {
        WorkCapture capture;
        lock.lock();
        try {
            capture = captureCurrentWorkLocked(clock.getAsLong(), Set.of());
        } finally {
            lock.unlock();
        }
        return capture.materialize();
    }

    private WorkCapture captureWorkLocked(long nowMs) {
        requireLock();
        if (committedWorkCapture == null || committedWorkCapture.capturedAtMs > nowMs) {
            committedWorkCapture = captureCurrentWorkLocked(nowMs, Set.of());
        }
        return committedWorkCapture;
    }

    private WorkCapture captureCurrentWorkLocked(long nowMs, Set<RequestEntry> excluded) {
        requireLock();
        List<WorkSnapshot.RequestWork> individual = new ArrayList<>();
        for (RequestEntry entry : requests.values()) {
            if (excluded.contains(entry) || entry.batch != null) { continue; }
            if (entry.isCommitted()) {
                individual.add(new WorkSnapshot.RequestWork(entry.route.requestId(), entry.individualPhase,
                        entry.remainingWorkAt(nowMs)));
            } else if (entry.ownership == OwnershipStage.DIRECT_RESERVED
                    && entry.reservation instanceof RouteReservation route) {
                individual.add(new WorkSnapshot.RequestWork(entry.route.requestId(), Phase.COMMITTED, route.predictedWorkMs));
            }
        }
        List<WorkSnapshot.BatchWork> capturedBatches = new ArrayList<>(batches.size());
        for (PrefillBatch batch : batches.values()) {
            List<Long> members = new ArrayList<>(batch.members.size());
            for (RequestEntry member : batch.members) {
                if (!excluded.contains(member)) { members.add(member.route.requestId()); }
            }
            if (!members.isEmpty()) {
                capturedBatches.add(new WorkSnapshot.BatchWork(members, batch.servicePhase,
                        OptionalLong.of(batch.remainingAt(nowMs))));
            }
        }
        return new WorkCapture(nowMs, individual, capturedBatches, unknownEngineRequestCount);
    }

    /** Local cleanup releases ownership without supplying a Worker execution result. */
    private void releaseCommittedLocked(RequestEntry entry, long nowMs) {
        requireLock();
        if (entry.batch != null) {
            entry.batch.touch(nowMs);
            entry.batch.outcome.learningEligible = false;
        }
        removeCommittedLocked(entry);
        recordMutationLocked();
    }

    private void removeCommittedLocked(RequestEntry entry) {
        PrefillBatch batch = entry.batch;
        if (batch != null) {
            checkState(batches.get(batch.batchId) == batch && batch.members.contains(entry),
                    "terminal member is not owned by its batch");
            batch.members.remove(entry);
            if (batch.members.isEmpty()) {
                batches.remove(batch.batchId, batch);
            }
        }
        checkState(entry.reservation == null, "committed request retained a preparation");
        checkState(removeRequestLocked(entry.route.requestId(), entry),
                "terminal request is not canonical request_id=%s", entry.route.requestId());
    }

    private Map<Long, TerminalObservation> terminalObservationsLocked(
            Map<String, WorkerStatus.TaskObservation> finishedTasks) {
        requireLock();
        Map<Long, TerminalObservation> terminals = new HashMap<>();
        for (WorkerStatus.TaskObservation task : finishedTasks.values()) {
            RequestEntry entry = requests.get(task.requestId());
            if (entry == null || !entry.isCommitted() || !matchesObservedBatch(entry, task.batchId())) {
                continue;
            }
            TerminalObservation terminal = TerminalObservation.from(entry, task);
            terminals.merge(
                    task.requestId(), terminal, TerminalObservation::merge);
        }
        return terminals;
    }

    private static PrefillRequestStatus terminalRequestStatus(TerminalObservation terminal) {
        PrefillRequestStatus.Kind kind = terminal.errorCode == 0L
                ? PrefillRequestStatus.Kind.COMPLETED
                : terminal.preemptionProgress
                        == PriorityPreemptionProgress.CANCELED
                    && terminal.errorCode
                            == StrategyErrorType.PRIORITY_PREEMPTED.getErrorCode()
                ? PrefillRequestStatus.Kind.PRIORITY_CANCELED
                : PrefillRequestStatus.Kind.FAILED;
        return PrefillRequestStatus.terminal(
                terminal.owner.route, kind, terminal.errorCode);
    }

    private static boolean matchesObservedBatch(
            RequestEntry entry, long observedBatchId) {
        if (entry.batch == null) {
            return true;
        }
        return observedBatchId > 0L
                && entry.batch.batchId == observedBatchId;
    }

    private static Phase strongerEnginePhase(Phase left, Phase right) {
        return left == Phase.ENGINE_RUNNING || right == Phase.ENGINE_RUNNING
                ? Phase.ENGINE_RUNNING : Phase.ENGINE_QUEUED;
    }

    /**
     * Roll back only a provisional lease. Once committed, the request table is
     * the sole owner and only its terminal reducer may release the capacity.
     */
    public record PreparationRollback(boolean released, EndpointGenerationLifecycle.HandoffPermit generationHandoff) {
        private static final PreparationRollback UNCHANGED = new PreparationRollback(false, null);
        private static final PreparationRollback ROUTE = new PreparationRollback(true, null);
    }

    /** Consume only open preparation and return any generation capability to its execution owner. */
    public PreparationRollback rollbackPreparation(Reservation reservation) {
        checkArgument(reservation.owner == this, "Preparation belongs to another State");
        checkState(!lock.isHeldByCurrentThread(), "Preparation rollback cannot run under ownershipLock");
        lock.lock();
        try {
            if (reservation.originalOwner == null) { return PreparationRollback.UNCHANGED; }
            var result = reservation instanceof BatchReservation batch
                    ? new PreparationRollback(true, batch.generationHandoff) : PreparationRollback.ROUTE;
            closeOpenLeaseLocked(reservation);
            return result;
        } finally { lock.unlock(); }
    }

    /** Rollback returns only resources owned by this uncommitted preparation. */
    private void closeOpenLeaseLocked(Reservation lease) {
        requireLock();
        checkState(lease.originalOwner != null, "Prefill preparation was already consumed");
        RequestEntry owner = openLeaseOwnerLocked(lease);
        checkState(owner == null || !owner.isCommitted(), "preparation has a committed owner");
        if (lease instanceof BatchReservation batch) {
            checkState(preparedBatches.remove(batch.batchId, batch), "batch preparation is not canonical");
            Objects.requireNonNull(batch.generationHandoff, "batch preparation lost its handoff");
            batch.generationHandoff = null;
        }
        lease.originalOwner = null;
        if (owner != null) {
            owner.reservation = null;
            if (owner.ownership == OwnershipStage.DIRECT_RESERVED) {
                removeRequestLocked(owner.route.requestId(), owner);
            }
        }
        recordMutationLocked();
    }

    private RequestEntry openLeaseOwnerLocked(Reservation lease) {
        requireLock();
        RequestEntry originalOwner = lease.originalOwner;
        RequestEntry current = requests.get(originalOwner.route.requestId());
        if (current == originalOwner) {
            if (current.reservation != lease) {
                throw new IllegalStateException(
                        "canonical Prefill lease owner lost its exact reservation"
                                + " request_id=" + originalOwner.route.requestId());
            }
            return current;
        }
        if (current != null && current.reservation == lease) {
            throw new IllegalStateException(
                    "replacement request cannot own an earlier Prefill lease"
                            + " request_id=" + originalOwner.route.requestId());
        }
        return null;
    }

    private void requireLock() {
        checkState(lock.isHeldByCurrentThread(),
                "Prefill ownership requires queueLock");
    }

}
