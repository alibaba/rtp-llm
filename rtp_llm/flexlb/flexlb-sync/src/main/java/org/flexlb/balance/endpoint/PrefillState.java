package org.flexlb.balance.endpoint;

import org.flexlb.balance.prediction.PrefillBatchFeatures;
import org.flexlb.balance.projection.WorkSnapshot;
import org.flexlb.balance.projection.WorkSnapshot.Phase;
import org.flexlb.balance.scheduler.RequestRoute;
import org.flexlb.enums.PriorityPreemptionProgress;
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
import java.util.concurrent.locks.Condition;
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
 * hold {@link #queueGuard()}, shared with the worker queue. Other mutation
 * methods acquire that lock internally. Methods return resource facts; Endpoint
 * executes notifications and request continuations after unlocking. Locked
 * methods must not perform RPC, prediction, or request notification work.
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

    public record ReservationResult<R extends Reservation>(CapacityStatus status, R reservation) {
        public ReservationResult {
            Objects.requireNonNull(status, "status");
            checkArgument(
                    (status == CapacityStatus.ACQUIRED) == (reservation != null),
                    "only ACQUIRED may carry a reservation");
        }
    }

    /**
     * Request status matched to its exact Prefill route, published after ledger reconciliation.
     */
    public record PrefillRequestStatus(RequestRoute route, Kind kind, long errorCode) {
        public PrefillRequestStatus {
            Objects.requireNonNull(route, "route");
            Objects.requireNonNull(kind, "kind");
            checkArgument(
                    kind != Kind.ACTIVE || errorCode == 0L,
                    "an active Prefill request status cannot carry an error code");
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

    public record AdmissionSummary(
            long occupiedRequests,
            long requestsToRelease,
            long lowerPriorityRequests,
            long samePriorityRequests,
            long higherPriorityRequests,
            long unknownPriorityRequests) {}

    public record Stats(
            int locallyOwnedRequests,
            int inflightRequests,
            int individuallyOwnedRequests,
            int batchCount,
            long maxObservedAgeMs) {
        public Stats {
            checkArgument(
                    locallyOwnedRequests >= 0
                            && inflightRequests >= 0
                            && individuallyOwnedRequests >= 0
                            && batchCount >= 0
                            && maxObservedAgeMs >= 0L,
                    "Prefill state stats must be non-negative");
        }
    }

    /**
     * An accepted plan owns its generation until the sender workflow exits.
     */
    public record CommittedHandoff(
            EndpointGenerationLifecycle.HandoffPermit generationHandoff,
            WorkCapture precedingWork,
            int remainingQueueDepth,
            boolean removedFailedMember)
            implements AutoCloseable {
        @Override
        public void close() {
            generationHandoff.close();
        }
    }

    /**
     * One exact ownership transaction; predictions are computed before entering the ledger lock.
     */
    public enum CommitKind {
        DIRECT,
        QUEUED,
        BATCH
    }

    public record CommitPlan(
            CommitKind kind,
            List<RequestRoute> items,
            List<RouteReservation> routeReservations,
            BatchReservation batchReservation,
            long[] predictions,
            RequestRoute failedMember,
            long selectionTimeMs,
            RequestRoute bypassedHead) {
        public CommitPlan(
                CommitKind kind,
                List<RequestRoute> items,
                List<RouteReservation> routeReservations,
                BatchReservation batchReservation,
                long[] predictions,
                RequestRoute failedMember,
                long selectionTimeMs) {
            this(
                    kind,
                    items,
                    routeReservations,
                    batchReservation,
                    predictions,
                    failedMember,
                    selectionTimeMs,
                    null);
        }

        public CommitPlan {
            Objects.requireNonNull(kind, "kind");
            items = List.copyOf(items);
            routeReservations = List.copyOf(routeReservations);
            predictions = predictions.clone();
        }

        @Override
        public long[] predictions() {
            return predictions.clone();
        }

        private List<RequestEntry> membersLocked(PrefillState state) {
            boolean direct = kind == CommitKind.DIRECT;
            checkState(!items.isEmpty(), "committed group requires members");
            var unique =
                    java.util.Collections.newSetFromMap(
                            new IdentityHashMap<RequestEntry, Boolean>());
            var members = new ArrayList<RequestEntry>(items.size());
            for (var route : items) {
                var entry = state.requests.get(route.requestId());
                if (!direct
                        && (entry == null
                                || entry.route != route
                                || entry.ownership != OwnershipStage.QUEUED)) {
                    return null;
                }
                checkState(
                        entry != null
                                && entry.route == route
                                && entry.ownership
                                        == (direct
                                                ? OwnershipStage.DIRECT_RESERVED
                                                : OwnershipStage.QUEUED),
                        "plan member is not canonical");
                checkState(unique.add(entry), "duplicate plan member");
                checkState(
                        state.activeIndex.contains(route) == !direct,
                        "plan member has inconsistent queue ownership");
                if (bypassedHead != null && !state.canBypassLocked(bypassedHead, route)) {
                    return null;
                }
                members.add(entry);
            }
            return members;
        }

        private void validatePreparationsLocked(PrefillState state, List<RequestEntry> members) {
            for (int i = 0; i < members.size(); i++) {
                var entry = members.get(i);
                Reservation expected =
                        kind == CommitKind.DIRECT
                                ? routeReservations.get(i)
                                : batchReservation != null
                                                && entry == batchReservation.originalOwner
                                        ? batchReservation
                                        : null;
                checkArgument(
                        expected == null || expected.owner == state,
                        "preparation belongs to another ledger");
                checkState(
                        entry.reservation == expected
                                && (expected == null || expected.originalOwner == entry),
                        "plan does not own exact member preparation");
            }
        }

        private RequestEntry failedMemberLocked(PrefillState state, List<RequestEntry> members) {
            if (kind == CommitKind.DIRECT
                    || failedMember == null
                    || failedMember.requestExpired(selectionTimeMs)) {
                return null;
            }
            var failed = state.requests.get(failedMember.requestId());
            if (failed == null
                    || failed.route != failedMember
                    || failed.ownership != OwnershipStage.QUEUED) {
                return null;
            }
            checkState(!members.contains(failed), "failed member belongs to committed selection");
            checkState(state.activeIndex.contains(failedMember), "failed member lost queue index");
            checkState(
                    failed.reservation == null || failed.reservation.originalOwner != null,
                    "failed member owns a consumed preparation");
            return failed;
        }
    }

    /**
     * Exact preparation capability, consumed by commit or rollback.
     */
    public abstract static class Reservation {
        final PrefillState owner;
        /* Guarded by owner.lock; null means committed, rolled back or retired. */
        RequestEntry originalOwner;

        private Reservation(PrefillState owner, RequestEntry originalOwner) {
            this.owner = owner;
            this.originalOwner = Objects.requireNonNull(originalOwner, "originalOwner");
        }
    }

    public static final class RouteReservation extends Reservation {
        /* guarded by PrefillState.lock until the reservation commits */
        private final long predictedWorkMs;

        private RouteReservation(
                PrefillState owner, RequestEntry originalOwner, long predictedWorkMs) {
            super(owner, originalOwner);
            this.predictedWorkMs = Math.clamp(predictedWorkMs, 0L, (long) Integer.MAX_VALUE);
        }
    }

    public static final class BatchReservation extends Reservation {
        private final long batchId;

        private BatchReservation(
                PrefillState owner,
                RequestEntry originalOwner,
                long batchId,
                EndpointGenerationLifecycle.HandoffPermit generationHandoff) {
            super(owner, originalOwner);
            this.batchId = batchId;
            this.generationHandoff = Objects.requireNonNull(generationHandoff, "generationHandoff");
        }

        /* guarded by PrefillState.lock; non-null only while OPEN */
        private EndpointGenerationLifecycle.HandoffPermit generationHandoff;
    }

    /**
     * One committed Prefill batch: exact members, execution progress and accumulated results.
     */
    static final class PrefillBatch {
        final long batchId;
        Set<RequestEntry> members = Set.of();
        BatchReservation preparation;
        long originalPredictionMs;
        PrefillBatchFeatures originalFeatures;
        long remainingWorkMs;
        Phase servicePhase = Phase.COMMITTED;
        BatchOutcome outcome = new BatchOutcome();
        long phaseBaseMs;
        long lastObservedAtMs;

        private PrefillBatch(long batchId, BatchReservation preparation) {
            checkArgument(batchId >= 0L, "batchId must be non-negative");
            this.batchId = batchId;
            this.preparation = preparation;
        }
    }

    /**
     * Resource ownership within this Prefill generation, independent of request protocol.
     */
    enum OwnershipStage {
        QUEUED,
        DIRECT_RESERVED,
        COMMITTED,
        STOP_PENDING
    }

    /**
     * Canonical resource facts for one exact route. Guarded by {@link #lock}.
     */
    static final class RequestEntry {
        OwnershipStage ownership;
        final RequestRoute route;
        Phase individualPhase;
        long remainingWorkMs;
        long phaseBaseMs;
        PrefillBatch batch;
        Reservation reservation;
        boolean bypassUsed;

        private RequestEntry(RequestRoute route, OwnershipStage ownership) {
            this.ownership = ownership;
            this.route = Objects.requireNonNull(route, "route");
        }

        private void commit(
                PrefillBatch committedBatch,
                RouteReservation directLease,
                long predictedMs,
                long nowMs) {
            if (committedBatch == null) {
                remainingWorkMs = Math.clamp(predictedMs, 0L, (long) Integer.MAX_VALUE);
                phaseBaseMs = nowMs;
                individualPhase = Phase.COMMITTED;
                if (directLease != null) {
                    directLease.originalOwner = null;
                }
            } else {
                batch = committedBatch;
            }
            ownership = OwnershipStage.COMMITTED;
            reservation = null;
        }
    }

    /**
     * Queue and committed work captured at one ownership linearization point.
     */
    public record Snapshot(
            Progress version,
            long capturedAtMs,
            PrefillActiveIndex.Capture active,
            WorkCapture work) {
        public Snapshot {
            Objects.requireNonNull(active, "missing active queue snapshot");
            Objects.requireNonNull(work, "missing committed work snapshot");
        }
    }

    /**
     * Immutable leaves copied under the ownership lock; projection runs outside it.
     */
    public static final class WorkCapture {
        private final long capturedAtMs;
        private List<WorkSnapshot.RequestWork> requests;
        private List<WorkSnapshot.BatchWork> batches;
        private final long unknownRequestCount;
        private volatile WorkSnapshot materialized;

        private WorkCapture(
                long capturedAtMs,
                List<WorkSnapshot.RequestWork> requests,
                List<WorkSnapshot.BatchWork> batches,
                long unknownRequestCount) {
            this.capturedAtMs = capturedAtMs;
            this.requests = List.copyOf(requests);
            this.batches = List.copyOf(batches);
            this.unknownRequestCount = unknownRequestCount;
        }

        /**
         * Shared lazy projection; callers never hold the endpoint ownership lock.
         */
        public WorkSnapshot materialize() {
            WorkSnapshot snapshot = materialized;
            if (snapshot != null) {
                return snapshot;
            }
            synchronized (this) {
                if (materialized == null) {
                    snapshot =
                            new WorkSnapshot(capturedAtMs, requests, batches, unknownRequestCount);
                    requests = null;
                    batches = null;
                    materialized = snapshot;
                }
                return materialized;
            }
        }
    }

    record TerminalObservation(
            RequestEntry owner,
            long executionTimeMs,
            long errorCode,
            PriorityPreemptionProgress preemptionProgress) {}

    /**
     * Package-local only for observation reduction; queue consumers use the guard.
     */
    final ReentrantLock lock;

    private final QueueGuard queueGuard;

    /**
     * Coordinates queue predicates and their wakeups on the ownership lock.
     * Waiting atomically releases this same lock, so a queue/input change cannot
     * be lost between checking a predicate and sleeping. No raw lock or Condition
     * escapes to the worker runtime.
     */
    public final class QueueGuard {
        private final Condition stateChanged;

        private QueueGuard() {
            stateChanged = lock.newCondition();
        }

        public void lock() {
            lock.lock();
        }

        public void lockInterruptibly() throws InterruptedException {
            lock.lockInterruptibly();
        }

        public void unlock() {
            lock.unlock();
        }

        public boolean isHeldByCurrentThread() {
            return lock.isHeldByCurrentThread();
        }

        public void requireHeld() {
            checkState(lock.isHeldByCurrentThread(), "Prefill ownership requires queue lock");
        }

        public void signal() {
            requireHeld();
            stateChanged.signal();
        }

        public void signalAll() {
            requireHeld();
            stateChanged.signalAll();
        }

        public void await() throws InterruptedException {
            requireHeld();
            stateChanged.await();
        }

        public void awaitNanos(long timeoutNanos) throws InterruptedException {
            requireHeld();
            stateChanged.awaitNanos(timeoutNanos);
        }
    }

    public QueueGuard queueGuard() {
        return queueGuard;
    }

    /**
     * Non-owning index containing only ACTIVE RequestRoute identities.
     */
    private PrefillActiveIndex activeIndex;

    /**
     * Canonical request ownership, changed only under the endpoint lock.
     */
    final Map<String, RequestEntry> requests = new HashMap<>();

    final LongSupplier clock;

    /**
     * Monotonic ownership/work revision used by projection snapshots.
     */
    private volatile long mutationVersion;

    private volatile long schedulingInputVersion;

    /**
     * Published at ownership mutation boundaries; admission still checks under the lock.
     */
    private volatile long outstandingRequestCount;

    private int committedBatchCount;
    private static final Progress EMPTY_PROGRESS = new Progress(0L, 0L, 0L, 0, 0, 0L);
    private volatile Progress progress = EMPTY_PROGRESS;
    private long requestCountsVersion = -1;
    private long[] requestCountsByPriority;

    /**
     * Derived immutable work; ACTIVE queue mutations leave committed work unchanged.
     */
    private WorkCapture committedWorkCapture;

    long unknownEngineRequestCount;
    private final Map<Long, PrefillBatch> batches = new HashMap<>();

    /**
     * Publish the capacity summary before readers observe a new ownership revision.
     */
    private void recordMutationLocked() {
        queueGuard.requireHeld();
        outstandingRequestCount = saturatedAdd(requests.size(), unknownEngineRequestCount);
        mutationVersion++;
        progress =
                new Progress(
                        activeIndex.version(),
                        schedulingInputVersion,
                        mutationVersion,
                        activeIndex.size(),
                        batches.size(),
                        outstandingRequestCount);
    }

    public PrefillState(ReentrantLock lock, PrefillActiveIndex activeIndex, LongSupplier clock) {
        this.lock = Objects.requireNonNull(lock, "lock");
        this.queueGuard = new QueueGuard();
        this.activeIndex = Objects.requireNonNull(activeIndex, "activeIndex");
        this.clock = Objects.requireNonNull(clock, "clock");
    }

    public record Progress(
            long queueVersion,
            long schedulingInputVersion,
            long ownershipVersion,
            int queueDepth,
            int batchSlots,
            long outstandingRequests) {}

    public Progress progress() {
        return progress;
    }

    /**
     * Enable waiting work without replacing the ledger of existing DIRECT reservations.
     */
    void enableQueueLocked(Comparator<RequestRoute> ordering) {
        queueGuard.requireHeld();
        if (activeIndex == PrefillActiveIndex.disabled()) {
            activeIndex = PrefillActiveIndex.ordered(16, ordering);
            schedulingInputsChangedLocked();
            recordMutationLocked();
        }
    }

    /**
     * One bounded, ordered queue view; all metadata belongs to the same ownership revision.
     */
    public record QueueSnapshot(
            long queueVersion,
            long schedulingInputVersion,
            List<RequestRoute> items,
            RequestRoute head) {}

    /**
     * Diagnostic counters captured from one queue ownership revision.
     */
    public record QueueCounters(
            long version, int[] byPriority, long outstandingRequests, int batchSlots) {}

    public QueueCounters captureQueueCounters() {
        lock.lock();
        try {
            int[] byPriority = new int[PriorityNormalizer.MAX_PRIORITY + 1];
            for (int priority = 0; priority < byPriority.length; priority++) {
                byPriority[priority] = activeIndex.size(priority);
            }
            return new QueueCounters(
                    activeIndex.version(),
                    byPriority,
                    saturatedAdd(requests.size(), unknownEngineRequestCount),
                    batches.size());
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
            return new QueueSnapshot(
                    activeIndex.version(),
                    schedulingInputVersion,
                    List.copyOf(items),
                    items.isEmpty() ? null : items.getFirst());
        } finally {
            lock.unlock();
        }
    }

    public void schedulingInputsChangedLocked() {
        queueGuard.requireHeld();
        schedulingInputVersion++;
        progress =
                new Progress(
                        activeIndex.version(),
                        schedulingInputVersion,
                        mutationVersion,
                        activeIndex.size(),
                        batches.size(),
                        outstandingRequestCount);
    }

    /**
     * Remove a known exact owner and its non-owning indexes; batch leases keep their own preparation.
     */
    private boolean removeEntry(RequestEntry entry) {
        if (requests.get(entry.route.requestId()) != entry) {
            return false;
        }
        if (entry.ownership == OwnershipStage.QUEUED) {
            checkState(activeIndex.remove(entry.route), "canonical queued owner lost its index");
        }
        if (entry.batch != null) {
            var batch = entry.batch;
            checkState(
                    batches.get(batch.batchId) == batch && batch.members.remove(entry),
                    "terminal member lost its batch");
            if (batch.members.isEmpty()) {
                batches.remove(batch.batchId, batch);
                committedBatchCount--;
            }
        }
        if (entry.ownership == OwnershipStage.COMMITTED
                || entry.ownership == OwnershipStage.DIRECT_RESERVED) {
            committedWorkCapture = null;
        }
        return requests.remove(entry.route.requestId(), entry);
    }

    public enum QueueChange {
        ENQUEUE,
        REMOVE,
        REMOVE_UNEXPIRED,
        STOP,
        ACK_STOP
    }

    /**
     * One exact queue transition. Worker runtime decisions may hold the same reentrant lock around it.
     */
    public RequestRoute updateQueue(QueueChange change, RequestRoute exact, long nowMs) {
        lock.lock();
        try {
            var route =
                    change == QueueChange.STOP
                            ? activeIndex.peek()
                            : Objects.requireNonNull(exact, "exact");
            if (route == null) {
                return null;
            }
            var entry = requests.get(route.requestId());
            if (change == QueueChange.ENQUEUE) {
                if (entry != null) {
                    return null;
                }
                entry = new RequestEntry(route, OwnershipStage.QUEUED);
                requests.put(route.requestId(), entry);
                try {
                    activeIndex.add(route);
                } catch (RuntimeException | Error failed) {
                    requests.remove(route.requestId(), entry);
                    throw failed;
                }
            } else if (change == QueueChange.ACK_STOP) {
                if (entry == null) {
                    return route;
                }
                if (entry.route != route
                        || entry.ownership != OwnershipStage.STOP_PENDING
                        || activeIndex.contains(route)) {
                    return null;
                }
                removeEntry(entry);
            } else {
                if (entry == null
                        || entry.route != route
                        || entry.ownership != OwnershipStage.QUEUED
                        || change == QueueChange.REMOVE_UNEXPIRED && route.requestExpired(nowMs)) {
                    return null;
                }
                checkState(
                        entry.reservation == null || entry.reservation.originalOwner != null,
                        "queue owner retained a consumed preparation");
                if (change == QueueChange.STOP) {
                    checkState(activeIndex.remove(route), "stopped owner lost its queue index");
                    entry.ownership = OwnershipStage.STOP_PENDING;
                } else {
                    removeEntry(entry);
                }
            }
            recordMutationLocked();
            return route;
        } finally {
            lock.unlock();
        }
    }

    public boolean ownsSelectionLocked(List<RequestRoute> items, long nowMs) {
        queueGuard.requireHeld();
        for (RequestRoute item : items) {
            if (!activeIndex.contains(item) || item.requestExpired(nowMs)) {
                return false;
            }
        }
        return true;
    }

    /**
     * The worker additionally requires PRIORITY ordering and a Decode-capacity wait.
     */
    public boolean canBypassLocked(RequestRoute head, RequestRoute candidate) {
        queueGuard.requireHeld();
        var entry = requests.get(head.requestId());
        return activeIndex.peek() == head
                && entry != null
                && entry.route == head
                && !entry.bypassUsed
                && activeIndex.contains(candidate)
                && !head.requestExpired(clock.getAsLong())
                && !candidate.requestExpired(clock.getAsLong())
                && head.priority() == candidate.priority()
                && head.decodeEp() != null
                && candidate.decodeEp() != null
                && head.decodeEp() != candidate.decodeEp();
    }

    /**
     * The queue part of a worker wait; the worker owns stop and control wakeups.
     */
    public boolean queueWaitCurrentLocked(
            RequestRoute head,
            long queueVersion,
            long inputVersion,
            boolean capacityWait,
            long nowMs) {
        queueGuard.requireHeld();
        if (head == null) {
            return activeIndex.isEmpty();
        }
        if (activeIndex.peek() != head) {
            return false;
        }
        return capacityWait
                ? !head.requestExpired(nowMs)
                        && (queueVersion == 0L
                                || requests.get(head.requestId()).bypassUsed
                                || activeIndex.version() == queueVersion)
                : activeIndex.version() == queueVersion && schedulingInputVersion == inputVersion;
    }

    // Protected by the ownership lock already acquired by every summary query.
    private long admissionSummaryVersion;
    private int admissionSummaryPriority;
    private long admissionSummaryLimit;
    private AdmissionSummary admissionSummaryCache;

    /**
     * Reuse stable inputs without allocating cache envelopes on mixed-priority misses.
     */
    public AdmissionSummary admissionSummary(int priority, long requestLimit) {
        lock.lock();
        try {
            if (admissionSummaryCache != null
                    && admissionSummaryVersion == mutationVersion
                    && admissionSummaryPriority == priority
                    && admissionSummaryLimit == requestLimit) {
                return admissionSummaryCache;
            }
            if (requestCountsByPriority == null || requestCountsVersion != mutationVersion) {
                long[] updatedCounts = new long[PriorityNormalizer.MAX_PRIORITY + 1];
                updatedCounts[0] = unknownEngineRequestCount;
                for (RequestEntry entry : requests.values()) {
                    int occupant = entry.route.priority();
                    updatedCounts[
                            occupant > 0 && occupant <= PriorityNormalizer.MAX_PRIORITY
                                    ? occupant
                                    : 0]++;
                }
                requestCountsByPriority = updatedCounts;
                requestCountsVersion = mutationVersion;
            }
            long lower = 0L, same = 0L, higher = 0L;
            for (int occupant = 1; occupant <= PriorityNormalizer.MAX_PRIORITY; occupant++) {
                if (occupant < priority) {
                    lower += requestCountsByPriority[occupant];
                } else if (occupant == priority) {
                    same += requestCountsByPriority[occupant];
                } else {
                    higher += requestCountsByPriority[occupant];
                }
            }
            admissionSummaryCache =
                    new AdmissionSummary(
                            outstandingRequestCount,
                            requestLimit > 0
                                    ? Math.max(0L, outstandingRequestCount - requestLimit + 1L)
                                    : 0L,
                            lower,
                            same,
                            higher,
                            requestCountsByPriority[0]);
            admissionSummaryVersion = mutationVersion;
            admissionSummaryPriority = priority;
            admissionSummaryLimit = requestLimit;
            return admissionSummaryCache;
        } finally {
            lock.unlock();
        }
    }

    ReservationResult<BatchReservation> reserveBatch(
            RequestRoute exactHead,
            long batchId,
            int maximum,
            EndpointGenerationLifecycle.HandoffPermit generationHandoff) {
        checkArgument(maximum > 0, "maximumInflightBatches must be positive");
        RequestRoute head = exactHead;
        lock.lock();
        try {
            RequestEntry entry = requests.get(head.requestId());
            if (entry == null
                    || entry.route != head
                    || entry.ownership != OwnershipStage.QUEUED
                            && entry.ownership != OwnershipStage.DIRECT_RESERVED) {
                return new ReservationResult<>(CapacityStatus.REQUEST_NOT_ACTIVE, null);
            }
            if (entry.reservation != null) {
                return new ReservationResult<>(CapacityStatus.REQUEST_ALREADY_RESERVED, null);
            }
            if (batches.containsKey(batchId)) {
                return new ReservationResult<>(CapacityStatus.BATCH_ID_ALREADY_RESERVED, null);
            }
            if (batches.size() >= maximum) {
                return new ReservationResult<>(CapacityStatus.CAPACITY_FULL, null);
            }
            BatchReservation lease = new BatchReservation(this, entry, batchId, generationHandoff);
            batches.put(batchId, new PrefillBatch(batchId, lease));
            entry.reservation = lease;
            recordMutationLocked();
            return new ReservationResult<>(CapacityStatus.ACQUIRED, lease);
        } finally {
            lock.unlock();
        }
    }

    /**
     * Queued routes consume no delivery slots; reserved and committed work do.
     */
    public boolean requestCapacityAvailable(long maximum, int requested) {
        checkArgument(maximum >= 0L && requested > 0, "invalid request capacity");
        lock.lock();
        try {
            long inFlight =
                    saturatedAdd(requests.size() - activeIndex.size(), unknownEngineRequestCount);
            return maximum == 0L || requested <= maximum && inFlight <= maximum - requested;
        } finally {
            lock.unlock();
        }
    }

    public boolean batchCapacityAvailable(int maximum) {
        checkArgument(maximum > 0, "maximumInflightBatches must be positive");
        lock.lock();
        try {
            return batches.size() < maximum;
        } finally {
            lock.unlock();
        }
    }

    /**
     * Validate the complete plan before changing a seat, preparation, or generation capability.
     */
    public CommittedHandoff commitLocked(
            CommitPlan plan, EndpointGenerationLifecycle.HandoffPermit handoff) {
        queueGuard.requireHeld();
        var items = plan.items();
        var leases = plan.routeReservations();
        var batchLease = plan.batchReservation();
        var predictions = plan.predictions;
        boolean direct = plan.kind() == CommitKind.DIRECT;
        if (plan.kind() == CommitKind.BATCH
                && !ownsSelectionLocked(items, plan.selectionTimeMs())) {
            return null;
        }
        checkArgument(
                (plan.kind() == CommitKind.BATCH) == (batchLease != null),
                "plan kind does not match batch preparation");
        checkArgument(
                direct ? !leases.isEmpty() : leases.isEmpty(),
                "DIRECT plan requires reservations; queued plan cannot own them");
        checkArgument(
                direct
                        ? predictions.length == 0
                        : predictions.length == (batchLease == null ? items.size() : 1),
                "prediction count does not match plan");
        List<RequestEntry> members = plan.membersLocked(this);
        if (members == null) {
            return null;
        }
        checkArgument(
                !direct || leases.size() == items.size(),
                "DIRECT plan requires one exact lease per member");
        PrefillBatch batch = batchLease == null ? null : batches.get(batchLease.batchId);
        if (batchLease != null) {
            checkState(
                    batchLease.owner == this
                            && batch != null
                            && batch.preparation == batchLease
                            && batchLease.originalOwner != null
                            && members.contains(batchLease.originalOwner),
                    "batch plan lost exact OPEN head");
            checkState(batchLease.generationHandoff != null, "batch preparation lost its handoff");
            handoff = batchLease.generationHandoff;
        }
        Objects.requireNonNull(handoff, "generationHandoff");
        plan.validatePreparationsLocked(this, members);
        RequestEntry failed = plan.failedMemberLocked(this, members);
        long nowMs = clock.getAsLong();
        Set<RequestEntry> batchMembers = batch == null ? null : new HashSet<>(members);
        PrefillBatchFeatures features =
                batch == null
                        ? null
                        : PrefillBatchFeatures.from(
                                items, RequestRoute::seqLen, RequestRoute::hitCache);
        checkArgument(
                batch == null || predictions[0] >= 0L, "predicted batch work must be non-negative");
        var committed =
                new CommittedHandoff(
                        handoff,
                        direct
                                ? captureWorkLocked(nowMs, Set.copyOf(members))
                                : captureWorkLocked(nowMs, null),
                        activeIndex.size()
                                - (direct ? 0 : members.size())
                                - (failed == null ? 0 : 1),
                        failed != null);
        if (batch != null) {
            batch.members = batchMembers;
            batch.preparation = null;
            committedBatchCount++;
            batch.originalPredictionMs = predictions[0];
            batch.remainingWorkMs = predictions[0];
            batch.originalFeatures = features;
            batch.phaseBaseMs = nowMs;
            batch.lastObservedAtMs = nowMs;
            batchLease.generationHandoff = null;
            batchLease.originalOwner = null;
        }
        for (int i = 0; i < members.size(); i++) {
            var member = members.get(i);
            if (!direct) {
                checkState(activeIndex.remove(member.route), "committed owner lost queue index");
            }
            member.commit(
                    batch,
                    direct ? leases.get(i) : null,
                    direct ? leases.get(i).predictedWorkMs : predictions[batch == null ? i : 0],
                    nowMs);
        }
        if (failed != null) {
            removeEntry(failed);
        }
        if (plan.bypassedHead() != null) {
            requests.get(plan.bypassedHead().requestId()).bypassUsed = true;
        }
        committedWorkCapture = null;
        recordMutationLocked();
        return committed;
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
            if (!requestCapacityAvailable(maxOutstandingRequests, 1)) {
                return new ReservationResult<>(CapacityStatus.CAPACITY_FULL, null);
            }
            RequestEntry entry = new RequestEntry(item, OwnershipStage.DIRECT_RESERVED);
            RouteReservation reservation = new RouteReservation(this, entry, predictedMs);
            ReservationResult<RouteReservation> result =
                    new ReservationResult<>(CapacityStatus.ACQUIRED, reservation);
            entry.reservation = reservation;
            requests.put(item.requestId(), entry);
            committedWorkCapture = null;
            recordMutationLocked();
            return result;
        } finally {
            lock.unlock();
        }
    }

    public enum RequestRelease {
        NONE,
        QUEUED,
        RESERVED,
        COMMITTED
    }

    public record Release(
            RequestRelease kind, EndpointGenerationLifecycle.HandoffPermit generationHandoff) {
        private static final Release NONE = new Release(RequestRelease.NONE, null);
        private static final Release RESERVED = new Release(RequestRelease.RESERVED, null);
        private static final Release COMMITTED = new Release(RequestRelease.COMMITTED, null);
        private static final Release QUEUED = new Release(RequestRelease.QUEUED, null);

        public boolean released() {
            return kind != RequestRelease.NONE;
        }

        public boolean queued() {
            return kind == RequestRelease.QUEUED;
        }
    }

    /**
     * Preparation rollback and exact resource cleanup share a result contract, never execution callbacks.
     */
    public Release release(Reservation preparation, RequestRoute route) {
        checkArgument(
                (preparation == null) != (route == null),
                "release needs one exact resource identity");
        if (preparation != null) {
            checkArgument(preparation.owner == this, "preparation belongs to another ledger");
            checkState(
                    !lock.isHeldByCurrentThread(),
                    "preparation rollback cannot run under ownership lock");
        }
        lock.lock();
        try {
            RequestEntry entry;
            if (preparation != null) {
                if (preparation.originalOwner == null) {
                    return Release.NONE;
                }
                entry = requests.get(preparation.originalOwner.route.requestId());
                if (entry == preparation.originalOwner) {
                    checkState(
                            entry.reservation == preparation,
                            "canonical preparation lost its exact owner");
                } else {
                    checkState(
                            entry == null || entry.reservation != preparation,
                            "replacement owns an earlier preparation");
                    entry = null;
                }
            } else {
                entry = requests.get(route.requestId());
                if (entry == null || entry.route != route) {
                    return Release.NONE;
                }
                if (entry.ownership != OwnershipStage.COMMITTED) {
                    preparation = entry.reservation;
                }
            }
            EndpointGenerationLifecycle.HandoffPermit handoff = null;
            var kind =
                    preparation != null
                                    || entry != null
                                            && entry.ownership == OwnershipStage.DIRECT_RESERVED
                            ? RequestRelease.RESERVED
                            : entry != null && entry.ownership == OwnershipStage.COMMITTED
                                    ? RequestRelease.COMMITTED
                                    : RequestRelease.QUEUED;
            if (preparation != null) {
                checkState(
                        preparation.originalOwner != null
                                && (entry == null || entry.ownership != OwnershipStage.COMMITTED),
                        "preparation retained a committed owner");
                if (preparation instanceof BatchReservation batch) {
                    var slot = batches.get(batch.batchId);
                    checkState(
                            slot != null
                                    && slot.preparation == batch
                                    && batches.remove(batch.batchId, slot),
                            "batch preparation is not canonical");
                    handoff =
                            Objects.requireNonNull(
                                    batch.generationHandoff, "missing generation handoff");
                    batch.generationHandoff = null;
                }
                preparation.originalOwner = null;
                if (entry != null) {
                    entry.reservation = null;
                    if (route != null || entry.ownership == OwnershipStage.DIRECT_RESERVED) {
                        removeEntry(entry);
                    }
                }
                kind =
                        route != null && entry.ownership == OwnershipStage.QUEUED
                                ? RequestRelease.QUEUED
                                : RequestRelease.RESERVED;
            } else if (entry != null) {
                if (entry.batch != null) {
                    entry.batch.lastObservedAtMs =
                            Math.max(entry.batch.lastObservedAtMs, clock.getAsLong());
                    entry.batch.outcome.learningEligible = false;
                }
                checkState(removeEntry(entry), "exact release lost canonical ownership");
            }
            recordMutationLocked();
            return handoff != null
                    ? new Release(kind, handoff)
                    : kind == RequestRelease.RESERVED
                            ? Release.RESERVED
                            : kind == RequestRelease.COMMITTED ? Release.COMMITTED : Release.QUEUED;
        } finally {
            lock.unlock();
        }
    }

    /**
     * Facts prepared under the lock, then read during prediction and exact-version commit.
     */
    public static final class StatusReduction {
        final PrefillState owner;
        long version;
        final Map<String, TerminalObservation> terminals;
        final Map<RequestEntry, Phase> individualPhases = new IdentityHashMap<>();
        final Map<PrefillBatch, Phase> batchPhases = new IdentityHashMap<>();
        final Map<PrefillBatch, BatchOutcome> batchOutcomes;
        // Assigned before publication; only the owning State applies the facts under its lock.
        long nowMs;
        long unknownRequests;
        Map<Long, List<RequestRoute>> predictionInputs;
        StatusReconciliation result;

        StatusReduction(PrefillState owner, Map<String, TerminalObservation> terminals) {
            this.owner = owner;
            this.terminals = terminals;
            this.batchOutcomes = terminals.isEmpty() ? Map.of() : new IdentityHashMap<>();
        }

        public Map<Long, List<RequestRoute>> predictionInputs() {
            return predictionInputs;
        }
    }

    /**
     * Owned by one batch or by an uncommitted status reduction, never both while preparing.
     */
    static final class BatchOutcome {
        long maxExecutionTimeMs;
        boolean successfulCompletion;
        boolean learningEligible = true;

        /**
         * Historical execution evidence survives a later queued observation.
         */
        boolean executionStarted;
    }

    private static long remainingWorkAt(
            Phase phase, long remainingWorkMs, long phaseBaseMs, long nowMs) {
        if (phase != Phase.ENGINE_RUNNING) {
            return remainingWorkMs;
        }
        long elapsedMs = Math.max(0L, nowMs - phaseBaseMs);
        return Math.max(0L, remainingWorkMs - elapsedMs);
    }

    /**
     * Capture exact activity into the same transaction that will commit it.
     */
    /**
     * Null means the out-of-lock prediction was invalidated; no fact has changed.
     */
    public StatusReconciliation commitStatusLocked(
            StatusReduction reduction, Map<Long, Long> predictions) {
        queueGuard.requireHeld();
        checkArgument(reduction.owner == this, "Status reduction belongs to another State");
        if (reduction.version != mutationVersion) {
            return null;
        }
        checkArgument(
                predictions.keySet().equals(reduction.predictionInputs.keySet()),
                "Predictions do not match the prepared batches");
        for (long prediction : predictions.values()) {
            checkArgument(prediction >= 0L, "Negative batch prediction");
        }
        // The version check also validates the capacity facts captured in the prepared result.
        for (var change : reduction.batchOutcomes.entrySet()) {
            PrefillBatch batch = change.getKey();
            batch.outcome = change.getValue();
            batch.lastObservedAtMs = Math.max(batch.lastObservedAtMs, reduction.nowMs);
        }
        for (TerminalObservation terminal : reduction.terminals.values()) {
            removeEntry(terminal.owner);
        }
        for (var phase : reduction.individualPhases.entrySet()) {
            var entry = phase.getKey();
            entry.remainingWorkMs =
                    remainingWorkAt(
                            entry.individualPhase,
                            entry.remainingWorkMs,
                            entry.phaseBaseMs,
                            reduction.nowMs);
            entry.phaseBaseMs = Math.max(entry.phaseBaseMs, reduction.nowMs);
            entry.individualPhase = phase.getValue();
        }
        for (var phase : reduction.batchPhases.entrySet()) {
            var batch = phase.getKey();
            batch.remainingWorkMs =
                    remainingWorkAt(
                            batch.servicePhase,
                            batch.remainingWorkMs,
                            batch.phaseBaseMs,
                            reduction.nowMs);
            batch.phaseBaseMs = Math.max(batch.phaseBaseMs, reduction.nowMs);
            batch.lastObservedAtMs = Math.max(batch.lastObservedAtMs, reduction.nowMs);
            batch.outcome.executionStarted |= phase.getValue() == Phase.ENGINE_RUNNING;
            batch.servicePhase = phase.getValue();
        }
        unknownEngineRequestCount = reduction.unknownRequests;
        for (long batchId : reduction.predictionInputs.keySet()) {
            PrefillBatch batch = batches.get(batchId);
            batch.remainingWorkMs = predictions.get(batchId);
            batch.phaseBaseMs = reduction.nowMs;
            batch.lastObservedAtMs = Math.max(batch.lastObservedAtMs, reduction.nowMs);
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
            List<RequestRoute> ownedItems =
                    requests.values().stream()
                            .map(entry -> entry.route)
                            .sorted(
                                    Comparator.comparingLong(RequestRoute::enqueueSeq)
                                            .thenComparing(RequestRoute::requestId))
                            .toList();
            List<BatchCompletion> completions =
                    batches.values().stream()
                            .filter(batch -> batch.preparation == null)
                            .map(
                                    batch ->
                                            new BatchCompletion(
                                                    batch.batchId,
                                                    batch.originalFeatures,
                                                    batch.originalPredictionMs,
                                                    batch.outcome.maxExecutionTimeMs,
                                                    batch.outcome.successfulCompletion,
                                                    false))
                            .sorted(Comparator.comparingLong(BatchCompletion::batchId))
                            .toList();
            Throwable invariantFailure = null;
            for (PrefillBatch prepared : batches.values()) {
                BatchReservation reservation = prepared.preparation;
                if (reservation == null) {
                    continue;
                }
                if (reservation.generationHandoff != null) {
                    orphanedHandoffs.add(reservation.generationHandoff);
                }
                invariantFailure =
                        Failures.append(
                                invariantFailure,
                                new IllegalStateException(
                                        "retirement reached an OPEN Prefill batch lease"));
            }
            retirement =
                    new Retirement(ownedItems, completions, invariantFailure, orphanedHandoffs);
            var retiredProgress =
                    new Progress(
                            activeIndex.version() + (activeIndex.isEmpty() ? 0L : 1L),
                            schedulingInputVersion,
                            mutationVersion + 1L,
                            0,
                            0,
                            0L);
            for (RequestEntry entry : requests.values()) {
                if (entry.reservation != null) {
                    entry.reservation.originalOwner = null;
                }
                entry.reservation = null;
            }
            for (PrefillBatch prepared : batches.values()) {
                BatchReservation reservation = prepared.preparation;
                if (reservation == null) {
                    continue;
                }
                reservation.generationHandoff = null;
                reservation.originalOwner = null;
            }
            for (PrefillBatch batch : batches.values()) {
                if (batch.preparation == null) {
                    batch.members.clear();
                }
            }
            requests.clear();
            batches.clear();
            committedBatchCount = 0;
            activeIndex.clear();
            unknownEngineRequestCount = 0L;
            committedWorkCapture = null;
            outstandingRequestCount = 0L;
            mutationVersion++;
            progress = retiredProgress;
        } finally {
            lock.unlock();
        }
        return retirement;
    }

    /**
     * Capture exact committed identities before the workflow checks request ownership outside this lock.
     */
    public List<RequestRoute> cleanupCandidates() {
        lock.lock();
        try {
            return requests.values().stream()
                    .filter(entry -> entry.ownership == OwnershipStage.COMMITTED)
                    .map(entry -> entry.route)
                    .toList();
        } finally {
            lock.unlock();
        }
    }

    /**
     * One orphan pass: a retained member protects its whole batch. Counts batches and individuals.
     */
    public int evictExpiredInflight(long ttlMs, Set<RequestRoute> orphanCandidates) {
        int evicted = 0;
        lock.lock();
        try {
            long nowMs = clock.getAsLong();
            long ttl = Math.max(0L, ttlMs);
            for (PrefillBatch batch : List.copyOf(batches.values())) {
                if (batch.preparation != null) {
                    continue;
                }
                Set<RequestEntry> members = batch.members;
                boolean retained = nowMs - batch.lastObservedAtMs < ttl;
                for (RequestEntry entry : members) {
                    retained |= !orphanCandidates.contains(entry.route);
                }
                if (retained) {
                    continue;
                }
                for (RequestEntry entry : List.copyOf(members)) {
                    if (entry.batch != null) {
                        entry.batch.lastObservedAtMs =
                                Math.max(entry.batch.lastObservedAtMs, nowMs);
                        entry.batch.outcome.learningEligible = false;
                    }
                    removeEntry(entry);
                    recordMutationLocked();
                }
                evicted++;
            }
            List<RequestEntry> individuals = new ArrayList<>();
            for (RequestEntry entry : requests.values()) {
                if (entry.batch == null
                        && entry.ownership == OwnershipStage.COMMITTED
                        && nowMs - entry.phaseBaseMs >= ttl
                        && orphanCandidates.contains(entry.route)) {
                    individuals.add(entry);
                }
            }
            for (RequestEntry entry : individuals) {
                if (entry.batch != null) {
                    entry.batch.lastObservedAtMs = Math.max(entry.batch.lastObservedAtMs, nowMs);
                    entry.batch.outcome.learningEligible = false;
                }
                removeEntry(entry);
                recordMutationLocked();
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
            int inflight = 0;
            int individual = 0;
            long maxAgeMs = 0L;
            long nowMs = clock.getAsLong();
            for (RequestEntry entry : requests.values()) {
                if (entry.ownership != OwnershipStage.COMMITTED) {
                    continue;
                }
                locallyOwned++;
                Phase reportedPhase =
                        entry.batch == null ? entry.individualPhase : entry.batch.servicePhase;
                if (reportedPhase == Phase.COMMITTED) {
                    inflight++;
                }
                if (entry.batch == null) {
                    individual++;
                    maxAgeMs = Math.max(maxAgeMs, Math.max(0L, nowMs - entry.phaseBaseMs));
                }
            }
            for (PrefillBatch batch : batches.values()) {
                if (batch.preparation != null) {
                    continue;
                }
                maxAgeMs = Math.max(maxAgeMs, Math.max(0L, nowMs - batch.lastObservedAtMs));
            }
            return new Stats(locallyOwned, inflight, individual, committedBatchCount, maxAgeMs);
        } finally {
            lock.unlock();
        }
    }

    Snapshot snapshotLocked() {
        queueGuard.requireHeld();
        long nowMs = clock.getAsLong();
        return new Snapshot(progress, nowMs, activeIndex.capture(), captureWorkLocked(nowMs, null));
    }

    public java.util.OptionalLong tryLoadMetric() {
        if (!lock.tryLock()) {
            return java.util.OptionalLong.empty();
        }
        WorkCapture capture;
        try {
            capture = captureWorkLocked(clock.getAsLong(), Set.of());
        } finally {
            lock.unlock();
        }
        return capture.materialize().totalRemainingWorkMs();
    }

    public WorkSnapshot committedSnapshot() {
        WorkCapture capture;
        lock.lock();
        try {
            capture = captureWorkLocked(clock.getAsLong(), Set.of());
        } finally {
            lock.unlock();
        }
        return capture.materialize();
    }

    private WorkCapture captureWorkLocked(long nowMs, Set<RequestEntry> excluded) {
        queueGuard.requireHeld();
        boolean cached = excluded == null;
        if (cached && committedWorkCapture != null && committedWorkCapture.capturedAtMs <= nowMs) {
            return committedWorkCapture;
        }
        if (cached) {
            excluded = Set.of();
        }
        List<WorkSnapshot.RequestWork> individual = new ArrayList<>();
        for (RequestEntry entry : requests.values()) {
            if (excluded.contains(entry) || entry.batch != null) {
                continue;
            }
            if (entry.ownership == OwnershipStage.COMMITTED) {
                individual.add(
                        new WorkSnapshot.RequestWork(
                                entry.route.requestId(),
                                entry.individualPhase,
                                remainingWorkAt(
                                        entry.individualPhase,
                                        entry.remainingWorkMs,
                                        entry.phaseBaseMs,
                                        nowMs)));
            } else if (entry.ownership == OwnershipStage.DIRECT_RESERVED
                    && entry.reservation instanceof RouteReservation route) {
                individual.add(
                        new WorkSnapshot.RequestWork(
                                entry.route.requestId(), Phase.COMMITTED, route.predictedWorkMs));
            }
        }
        List<WorkSnapshot.BatchWork> capturedBatches = new ArrayList<>(committedBatchCount);
        for (PrefillBatch batch : batches.values()) {
            if (batch.preparation != null) {
                continue;
            }
            List<String> members = new ArrayList<>(batch.members.size());
            for (RequestEntry member : batch.members) {
                if (!excluded.contains(member)) {
                    members.add(member.route.requestId());
                }
            }
            if (!members.isEmpty()) {
                capturedBatches.add(
                        new WorkSnapshot.BatchWork(
                                members,
                                batch.servicePhase,
                                OptionalLong.of(
                                        remainingWorkAt(
                                                batch.servicePhase,
                                                batch.remainingWorkMs,
                                                batch.phaseBaseMs,
                                                nowMs))));
            }
        }
        var capture =
                new WorkCapture(nowMs, individual, capturedBatches, unknownEngineRequestCount);
        if (cached) {
            committedWorkCapture = capture;
        }
        return capture;
    }
}
