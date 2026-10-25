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
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.util.PriorityNormalizer;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
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
 * between containers.
 *
 * <p>Prefill 的请求表、等待队列索引和容量计数只用 worker 的 queue lock 这一把锁保护。
 * 自己加锁的公开方法会在方法里拿锁；其余方法要求调用方已经持有这把锁，方法开头用
 * {@link #requireLock()} 检查。BatchPrefill、SinglePrefill、QueuedPreemption、
 * WorkerStatusUpdater 用的也是同一把锁。</p>
 */
public final class PrefillState {

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
    public final WorkerStatusUpdater workerStatusUpdater = new WorkerStatusUpdater(this);

    /** Publish the capacity summary before readers observe a new ownership revision. */
    public void recordMutation() {
        publishRequestCount();
        mutationVersion++;
    }

    public void publishRequestCount() {
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

    public RequestEntry putRequest(String requestId, RequestEntry entry) {
        requireLock();
        return requests.put(requestId, entry);
    }

    public boolean removeRequest(String requestId, RequestEntry entry) {
        requireLock();
        if (!requests.remove(requestId, entry)) {
            return false;
        }
        if (entry.queueMembership == QueueMembership.UNINDEXED) {
            committedWorkCapture = null;
        }
        return true;
    }

    /** Advisory revision read for the lock-free load-view cache fast path. */
    public long mutationVersion() {
        return mutationVersion;
    }

    public boolean enqueueActive(ScheduledRequest item, long maxOutstandingRequests) {
        requireLock();
        if (requests.containsKey(item.requestId())
                || (maxOutstandingRequests > 0L && !hasRequestCapacity(maxOutstandingRequests))) {
            return false;
        }
        RequestEntry entry = new RequestEntry(item);
        putRequest(item.requestId(), entry);
        try {
            activeIndex.add(item);
        } catch (RuntimeException | Error failure) {
            removeRequest(item.requestId(), entry);
            throw failure;
        }
        recordMutation();
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
        removeFromWaitingQueue(entry, item);
        // BATCH preparation owns any OPEN batch lease together with a
        // generation handoff. NON_BATCH item-owned single reservations use the
        // atomic counterpart below so retirement can never observe an orphan.
        removeRequest(item.requestId(), entry);
        recordMutation();
        return true;
    }

    /** Only failed admission needs priority provenance; reuse one summary per revision. */
    public AdmissionRejectReason rejectReason(int priority, long requestLimit) {
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

    public List<ScheduledRequest> queuedPreemptionVictims(int priority, long requestLimit) {
        return queuedPreemption.queuedPreemptionVictims(priority, requestLimit);
    }

    public ScheduledRequest oneQueuedPreemptionVictim(int priority) {
        return queuedPreemption.oneQueuedPreemptionVictim(priority);
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
    public ScheduledRequest detachNextActiveForStop() {
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
        recordMutation();
        return item;
    }

    /** Remove only the exact stop-pending owner whose callback completed. */
    public boolean removeStoppedRequest(ScheduledRequest item) {
        requireLock();
        RequestEntry entry = requests.get(item.requestId());
        if (entry == null
                || entry.activeItem != item
                || entry.queueMembership != QueueMembership.STOP_DETACHED
                || activeIndex.contains(item)) {
            return false;
        }
        boolean removed = removeRequest(item.requestId(), entry);
        if (removed) {
            recordMutation();
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

    public int batchLeasesInUse() {
        return batchPrefill.batchLeasesInUse();
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

    public void checkQueueMembership(RequestEntry entry, ScheduledRequest item) {
        requireState(entry.queueMembership != QueueMembership.STOP_DETACHED,
                "stopped admission cannot commit request_id=" + item.requestId());
        requireState(activeIndex.contains(item) == (entry.queueMembership == QueueMembership.WAITING),
                "canonical admission and waiting index disagree request_id=" + item.requestId());
    }

    public void removeFromWaitingQueue(RequestEntry entry, ScheduledRequest item) {
        checkQueueMembership(entry, item);
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
                    entry.batchProgress == null ? null : batchPrefill.liveBatchMembers(entry.batchProgress);
            // Local cleanup does not prove that Engine omitted this member's work.
            capacityReleased = removeCommittedEntry(entry, FinishReport.local(entry.requestId), liveMembers, null);
            return true;
        } finally {
            lock.unlock();
            notifyCapacityAvailable(capacityReleased);
        }
    }

    public WorkerStatusResult updateFromWorkerStatus(WorkerStatus.StatusObservation observation,
                                                     ToLongFunction<List<ScheduledRequest>> predictBatchTime,
                                                     Runnable publishStatus,
                                                     Runnable onPublishFailure) {
        return workerStatusUpdater.updateFromWorkerStatus(observation, predictBatchTime, publishStatus,
                onPublishFailure);
    }

    public HeartbeatResult updateFromHeartbeat(WorkerStatus.StatusObservation observation) {
        return workerStatusUpdater.updateFromHeartbeat(observation);
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
                    if (lease.state != LeaseState.OPEN && lease.state != LeaseState.OWNED) {
                        invariantFailure = appendRetirementInvariant(invariantFailure,
                                "canonical retirement owner has an invalid single-reservation state");
                    }
                } else if (lease instanceof BatchReservation batch) {
                    observedBatchLeases++;
                    invariantFailure = BatchPrefill.checkRetiringBatchLease(batch, orphanedHandoffs, invariantFailure);
                } else {
                    invariantFailure = appendRetirementInvariant(
                            invariantFailure,
                            "retirement found an unknown Prefill lease type");
                }
            }
            if (batchPrefill.batchLeasesInUse() != observedBatchLeases) {
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
            batchPrefill.resetBatchLeasesInUse();
            unknownEngineRequestCount = 0L;
            committedWorkCapture = null;
            recordMutation();
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

    public static Throwable appendRetirementInvariant(Throwable first, String message) {
        IllegalStateException next = new IllegalStateException(message);
        if (first == null) {
            return next;
        }
        first.addSuppressed(next);
        return first;
    }

    public int evictExpiredSingles(long ttlMs, java.util.function.Predicate<String> retainForSchedulerCleanup) {
        return singlePrefill.evictExpiredSingles(ttlMs, retainForSchedulerCleanup);
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
            maxAgeMs = Math.max(maxAgeMs, BatchPrefill.maxBatchAgeMs(batches, nowMs));
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

    private boolean hasRequestCapacity(long maxOutstandingRequests) {
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

    public Snapshot snapshot() {
        requireLock();
        long nowMs = clock.getAsLong();
        return new Snapshot(nowMs, activeIndex.capture(), captureWork(nowMs));
    }

    public WorkView committedSnapshot() {
        WorkCapture capture;
        lock.lock();
        try {
            capture = captureCurrentWork(clock.getAsLong());
        } finally {
            lock.unlock();
        }
        return capture.materialize();
    }

    private WorkCapture captureWork(long nowMs) {
        requireLock();
        if (committedWorkCapture == null || committedWorkCapture.capturedAtMs > nowMs) {
            committedWorkCapture = captureCurrentWork(nowMs);
        }
        return committedWorkCapture;
    }

    public WorkView capturePrecedingWork(List<ScheduledRequest> members, long nowMs) {
        requireLock();
        Set<RequestEntry> excluded = java.util.Collections.newSetFromMap(new IdentityHashMap<>());
        for (ScheduledRequest member : members) {
            excluded.add(requests.get(member.requestId()));
        }
        return captureCurrentWork(nowMs, excluded).materialize();
    }

    private WorkCapture captureCurrentWork(long nowMs) {
        return captureCurrentWork(nowMs, Set.of());
    }

    private WorkCapture captureCurrentWork(long nowMs, Set<RequestEntry> excluded) {
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
        return new WorkCapture(nowMs, individual, BatchPrefill.buildBatchWork(batchMembers, nowMs),
                unknownEngineRequestCount);
    }

    public boolean removeCommittedEntry(RequestEntry entry,
                                        FinishReport finishReport,
                                        LiveBatchMembers liveMembers,
                                        List<FinishedBatch> finishedBatches) {
        return removeCommittedEntry(entry, finishReport, liveMembers, finishedBatches, clock.getAsLong());
    }

    public boolean removeCommittedEntry(RequestEntry entry,
                                        FinishReport finishReport,
                                        LiveBatchMembers liveMembers,
                                        List<FinishedBatch> finishedBatches,
                                        long nowMs) {
        requireLock();
        BatchProgress batch = entry.batchProgress;
        boolean lastBatchMember = false;
        Reservation lease;
        if (batch != null) {
            lastBatchMember = BatchPrefill.isLastLiveMember(entry, liveMembers);
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
            BatchPrefill.removeFinishedMember(entry, liveMembers, finishReport, nowMs);
        }
        if (!removeRequest(entry.requestId, entry)) {
            throw new IllegalStateException(
                    "terminal request is not canonical request_id="
                            + entry.requestId);
        }
        committedWorkCapture = null;
        recordMutation();
        if (lease != null) {
            closeOwnedLease(lease);
        }
        if (lastBatchMember && finishedBatches != null) {
            finishedBatches.add(batch.finishedBatch());
        }
        return true;
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
                generationHandoff = closeOpenLease(lease);
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
    public EndpointGenerationLifecycle.HandoffPermit closeOpenLease(Reservation lease) {
        requireLock();
        if (lease.state != LeaseState.OPEN) {
            throw new IllegalStateException(
                    "Prefill OPEN rollback lost its exact lease");
        }
        RequestEntry owner = openLeaseOwner(lease);
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
        releaseBatchSlot(lease);
        if (owner != null) {
            owner.reservation = null;
        }
        recordMutation();
        return generationHandoff;
    }

    /** Close exact ownership and return a batch slot when this is the last member. */
    private void closeOwnedLease(Reservation lease) {
        requireLock();
        if (lease.state != LeaseState.OWNED
                || lease instanceof BatchReservation batch
                && batch.generationHandoff != null) {
            throw new IllegalStateException(
                    "Prefill OWNED terminal still owns an admission handoff");
        }
        lease.state = LeaseState.CLOSED;
        releaseBatchSlot(lease);
    }

    private void releaseBatchSlot(Reservation lease) {
        requireLock();
        if (lease instanceof BatchReservation) {
            batchPrefill.releaseBatchSlot();
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

    private RequestEntry openLeaseOwner(Reservation lease) {
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
