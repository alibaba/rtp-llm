package org.flexlb.balance.endpoint;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.scheduler.ScheduledRequest;
import org.flexlb.balance.prediction.PrefillBatchFeatures;
import org.flexlb.balance.projection.WorkSnapshot;
import org.flexlb.balance.projection.WorkSnapshot.Phase;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.PriorityPreemptionProgress;
import org.flexlb.enums.TaskPhase;

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
import java.util.concurrent.PriorityBlockingQueue;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.Function;
import java.util.function.LongPredicate;
import java.util.function.LongSupplier;

/**
 * Canonical Prefill request ownership for one worker generation.
 *
 * <p>Every known request id has exactly one {@link RequestEntry}. The worker
 * queue is only an ordered index over entries which still hold an active item;
 * callback and Engine progress mutate the same entry instead of moving ownership
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

    public record ReservationResult<R extends Reservation>(
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

    /** Production RTP-LLM raw {@code ErrorCode::PRIORITY_PREEMPTED}. */
    private static final long PRIORITY_PREEMPTED_ERROR_CODE = 8429L;

    enum LeaseState {
        OPEN,
        OWNED,
        CLOSED
    }

    /** One-shot capability returned when an OPEN admission commits. */
    public static final class CommittedHandoff implements AutoCloseable {
        private EndpointGenerationLifecycle.HandoffPermit generationHandoff;

        private CommittedHandoff(
                EndpointGenerationLifecycle.HandoffPermit generationHandoff) {
            this.generationHandoff = generationHandoff;
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

    /** Exact one-shot Engine-fence guard for one canonical request generation. */
    public final class Protection {
        private final PrefillState owner = PrefillState.this;
        private final RequestEntry entry;

        private Protection(RequestEntry entry) {
            this.entry = entry;
        }
    }

    /**
     * One-shot rollback right for one provisional DIRECT registration. The
     * capability is bound to the canonical RequestEntry identity, so its close
     * can never remove a later request which reused the same id.
     */
    public final class DirectRegistration implements AutoCloseable {
        private RequestEntry rollbackEntry;

        private DirectRegistration(RequestEntry rollbackEntry) {
            this.rollbackEntry = rollbackEntry;
        }

        /** Leave the registered entry under canonical registry ownership. */
        public void commit() {
            detachRollbackEntry();
        }

        /** Roll back only this exact entry when commit has not consumed it. */
        @Override
        public void close() {
            RequestEntry entry = detachRollbackEntry();
            if (entry == null) {
                return;
            }
            rollbackDirectRegistration(entry);
        }

        private synchronized RequestEntry detachRollbackEntry() {
            RequestEntry exact = rollbackEntry;
            rollbackEntry = null;
            return exact;
        }
    }

    /** The only capacity ownership state passed through admission and callback. */
    public abstract class Reservation implements AutoCloseable {
        /* guarded by PrefillState.lock */ LeaseState state =
                LeaseState.OPEN;
        private final RequestEntry originalOwner;
        /* guarded by PrefillState.lock; non-null only while OPEN */
        private EndpointGenerationLifecycle.HandoffPermit generationHandoff;

        private Reservation(RequestEntry originalOwner,
                            EndpointGenerationLifecycle.HandoffPermit generationHandoff) {
            this.originalOwner = Objects.requireNonNull(
                    originalOwner, "originalOwner");
            this.generationHandoff = Objects.requireNonNull(
                    generationHandoff, "generationHandoff");
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
        private final long predictedWorkMs;

        private RouteReservation(RequestEntry originalOwner,
                                 long requestId,
                                 long predictedWorkMs,
                                 EndpointGenerationLifecycle.HandoffPermit generationHandoff) {
            super(originalOwner, generationHandoff);
            this.requestId = requestId;
            this.predictedWorkMs = boundedPrediction(predictedWorkMs);
        }

        /** Atomically commit one exact route group. */
        public List<CommittedHandoff> commitGroup(
                List<ScheduledRequest> items,
                List<RouteReservation> exactReservations) {
            if (exactReservations.isEmpty() || exactReservations.get(0) != this) {
                throw new IllegalArgumentException(
                        "route commit must be invoked on its first exact reservation");
            }
            List<RouteReservation> exactLeases = new ArrayList<>(
                    exactReservations.size());
            for (RouteReservation reservation
                    : exactReservations) {
                if (reservation.owner != PrefillState.this) {
                    throw new IllegalArgumentException(
                            "route reservation belongs to another Prefill ledger");
                }
                exactLeases.add(reservation);
            }
            lock.lock();
            try {
                return commitRoutesUnderLock(items, exactLeases);
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
            super(originalOwner, generationHandoff);
            this.headRequestId = headRequestId;
            this.batchId = batchId;
        }

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
        private OptionalLong predictedWorkMs;
        private OptionalLong remainingWorkMs;
        private Phase servicePhase = Phase.COMMITTED;
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
            this.predictedWorkMs = OptionalLong.of(predictedWorkMs);
            this.remainingWorkMs = OptionalLong.of(predictedWorkMs);
            this.phaseBaseMs = nowMs;
            this.lastObservedAtMs = nowMs;
        }

        private OptionalLong remainingAt(long nowMs) {
            if (remainingWorkMs.isEmpty()
                    || servicePhase != Phase.ENGINE_RUNNING) {
                return remainingWorkMs;
            }
            return OptionalLong.of(Math.max(
                    0L, remainingWorkMs.getAsLong()
                            - Math.max(0L, nowMs - phaseBaseMs)));
        }

        private void observeTerminal(TerminalObservation terminal, long nowMs) {
            touch(nowMs);
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
            if (servicePhase == Phase.ENGINE_RUNNING
                    && remainingWorkMs.isPresent()) {
                remainingWorkMs = remainingAt(nowMs);
            }
            phaseBaseMs = Math.max(phaseBaseMs, nowMs);
            touch(nowMs);
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

        private List<ScheduledRequest> liveItems() {
            List<ScheduledRequest> live = new ArrayList<>(members.size());
            for (RequestEntry entry : members) {
                live.add(entry.committedItem);
            }
            live.sort(Comparator.comparingLong(ScheduledRequest::enqueueSeq)
                    .thenComparingLong(ScheduledRequest::requestId));
            return live;
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

    /** The sole mutable request lifecycle record. Guarded by {@link #lock}. */
    private static final class RequestEntry {
        private final long requestId;
        private ScheduledRequest activeItem;
        private Phase individualPhase;
        private long remainingWorkMs;
        private long phaseBaseMs;
        private long lastObservedAtMs;
        private BatchWork batchWork;
        /** Canonical callback identity for either queued delivery mode. */
        private ScheduledRequest committedItem;
        private Reservation reservation;
        private Protection protection;
        private TerminalObservation deferredTerminal;
        /**
         * The queue index has been detached for generation stop, while this
         * exact ACTIVE identity remains canonical until its terminal callback
         * acknowledges success or generation retirement projects it again.
         */
        private boolean stopTerminalPending;

        private RequestEntry(ScheduledRequest item) {
            this.requestId = item.requestId();
            this.activeItem = item;
        }

        private RequestEntry(long requestId, long predictedWorkMs, long nowMs) {
            this.requestId = requestId;
            this.individualPhase = Phase.COMMITTED;
            this.remainingWorkMs = boundedPrediction(predictedWorkMs);
            this.phaseBaseMs = nowMs;
            this.lastObservedAtMs = nowMs;
        }

        private boolean isActive() {
            return activeItem != null;
        }

        private boolean activeIdentity(ScheduledRequest item) {
            return activeItem != null
                    && activeItem == item
                    && !stopTerminalPending;
        }

        /** DIRECT has no queued ScheduledRequest owner at any point in its lifetime. */
        private boolean isDirect() {
            return activeItem == null && committedItem == null;
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
                           WorkSnapshot committedWork,
                           long pendingRequestCount) {
        public Snapshot {
            activeItems = List.copyOf(activeItems);
            Objects.requireNonNull(
                    committedWork, "missing committed work snapshot");
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
    private final PriorityBlockingQueue<ScheduledRequest> activeIndex;
    /** Canonical ownership table; replaced only by an exact ACTIVE transaction. */
    private Map<Long, RequestEntry> requests = new HashMap<>();
    private final LongSupplier clock;
    private final Runnable capacityAvailable;
    private long unknownEngineRequestCount;
    private int routeLeasesInUse;
    private int batchLeasesInUse;

    public PrefillState(ReentrantLock lock,
                        PriorityBlockingQueue<ScheduledRequest> activeIndex,
                        Runnable capacityAvailable) {
        this(lock, activeIndex, System::currentTimeMillis, capacityAvailable);
    }

    public PrefillState(ReentrantLock lock,
                        PriorityBlockingQueue<ScheduledRequest> activeIndex,
                        LongSupplier clock,
                        Runnable capacityAvailable) {
        this.lock = Objects.requireNonNull(lock, "lock");
        this.activeIndex = Objects.requireNonNull(activeIndex, "activeIndex");
        this.clock = Objects.requireNonNull(clock, "clock");
        this.capacityAvailable = Objects.requireNonNull(
                capacityAvailable, "capacityAvailable");
    }

    public boolean enqueueActiveUnderLock(ScheduledRequest item) {
        requireLock();
        if (requests.containsKey(item.requestId())) {
            return false;
        }
        RequestEntry entry = new RequestEntry(item);
        requests.put(item.requestId(), entry);
        try {
            activeIndex.add(item);
        } catch (RuntimeException | Error failure) {
            requests.remove(item.requestId(), entry);
            throw failure;
        }
        return true;
    }

    /** Replace exact ACTIVE identities while the caller holds the ownership lock. */
    public boolean replaceActiveExact(
            List<ScheduledRequest> exactVictims,
            ScheduledRequest incoming) {
        requireLock();
        List<ScheduledRequest> victims = exactVictims;
        requireState(!victims.isEmpty(),
                "ACTIVE replacement requires exact victims");

        Set<Long> victimIds = new HashSet<>();
        List<RequestEntry> victimEntries = new ArrayList<>(victims.size());
        for (ScheduledRequest victim : victims) {
            if (victim == null || !victimIds.add(victim.requestId())) {
                return false;
            }
            RequestEntry entry = requests.get(victim.requestId());
            if (entry == null || !entry.activeIdentity(victim)) {
                return false;
            }
            Reservation lease = entry.reservation;
            requireState(lease == null || lease.state == LeaseState.OPEN,
                    "ACTIVE victim owns a non-OPEN Prefill lease request_id="
                            + victim.requestId());
            requireState(activeIndex.contains(victim),
                    "canonical ACTIVE victim has no queue index request_id="
                            + victim.requestId());
            victimEntries.add(entry);
        }
        if (victimIds.contains(incoming.requestId())
                || requests.containsKey(incoming.requestId())) {
            return false;
        }
        for (ScheduledRequest indexed : activeIndex) {
            if (indexed == incoming
                    || indexed.requestId() == incoming.requestId()) {
                return false;
            }
        }

        // Validation above is the transaction boundary. The queue never
        // exposes this intermediate state because the caller owns lock, and a
        // replacement always removes at least one index before adding one.
        for (int index = 0; index < victims.size(); index++) {
            ScheduledRequest victim = victims.get(index);
            RequestEntry entry = victimEntries.get(index);
            boolean removed = activeIndex.remove(victim);
            requireState(removed,
                    "validated ACTIVE victim disappeared before replacement request_id="
                            + victim.requestId());
            requests.remove(victim.requestId(), entry);
        }
        RequestEntry previous = requests.put(
                incoming.requestId(), new RequestEntry(incoming));
        requireState(previous == null, "validated incoming request already exists");
        activeIndex.add(incoming);
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
        boolean removed = activeIndex.remove(item);
        requireState(removed,
                "canonical ACTIVE request has no queue index request_id="
                        + item.requestId());
        // The synchronous admission still owns any OPEN lease and releases it
        // after this queue transaction. ACTIVE removal owns only request/index.
        requests.remove(item.requestId(), entry);
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
        entry.stopTerminalPending = true;
        return item;
    }

    /** Remove only the exact stop-pending owner whose callback completed. */
    public boolean acknowledgeStopTerminalUnderLock(ScheduledRequest item) {
        requireLock();
        RequestEntry entry = requests.get(item.requestId());
        if (entry == null
                || entry.activeItem != item
                || !entry.stopTerminalPending
                || activeIndex.contains(item)) {
            return false;
        }
        return requests.remove(item.requestId(), entry);
    }

    ReservationResult<RouteReservation> reserveRoute(
            ScheduledRequest exactItem,
            long predictedMs,
            int maximum,
            EndpointGenerationLifecycle.HandoffPermit generationHandoff) {
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
            if (maximum > 0
                    && routeCapacityUsedUnderLock() >= maximum) {
                return new ReservationResult<>(
                        CapacityStatus.CAPACITY_FULL, null);
            }
            RouteReservation lease = new RouteReservation(
                    entry, item.requestId(), predictedMs, generationHandoff);
            entry.reservation = lease;
            routeLeasesInUse++;
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
            if (maximum > 0 && batchLeasesInUse >= maximum) {
                return new ReservationResult<>(
                        CapacityStatus.CAPACITY_FULL, null);
            }
            BatchReservation lease = new BatchReservation(
                    entry, head.requestId(), batchId, generationHandoff);
            entry.reservation = lease;
            batchLeasesInUse++;
            return new ReservationResult<>(CapacityStatus.ACQUIRED, lease);
        } finally {
            lock.unlock();
        }
    }

    private boolean routeCapacityAvailable(int maximum) {
        lock.lock();
        try {
            return maximum <= 0
                    || routeCapacityUsedUnderLock() < maximum;
        } finally {
            lock.unlock();
        }
    }

    /**
     * NON_BATCH delivery capacity covers both locally owned route leases and
     * worker-reported requests whose exact Master ownership is unavailable.
     * Ignoring the latter lets an endpoint generation change or status gap
     * feed an already-busy Engine while the local lease count remains small.
     */
    private long routeCapacityUsedUnderLock() {
        requireLock();
        return saturatedAdd(routeLeasesInUse, unknownEngineRequestCount);
    }

    private boolean batchCapacityAvailable(int maximum) {
        lock.lock();
        try {
            return maximum <= 0 || batchLeasesInUse < maximum;
        } finally {
            lock.unlock();
        }
    }

    public CapacityBoundary.Availability routeAvailability(int maximum) {
        return new CapacityAvailability(false, maximum);
    }

    public CapacityBoundary.Availability batchAvailability(int maximum) {
        return new CapacityAvailability(true, maximum);
    }

    /** Exact wake capability permanently paired with this worker runtime. */
    private final class CapacityAvailability
            implements CapacityBoundary.Availability {
        private final boolean batch;
        private final int maximum;
        private Runnable subscribed;

        private CapacityAvailability(boolean batch, int maximum) {
            this.batch = batch;
            this.maximum = maximum;
        }

        @Override
        public boolean isAvailable() {
            return batch
                    ? batchCapacityAvailable(maximum)
                    : routeCapacityAvailable(maximum);
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

    private List<CommittedHandoff> commitRoutesUnderLock(
            List<ScheduledRequest> items,
            List<RouteReservation> leases) {
        requireLock();
        List<ScheduledRequest> members = validateActiveGroup(items);
        if (members.size() != leases.size()) {
            throw new IllegalArgumentException(
                    "route commit requires one exact lease per member");
        }
        Set<RouteReservation> unique = java.util.Collections.newSetFromMap(
                new IdentityHashMap<>());
        List<CommittedHandoff> committedHandoffs =
                new ArrayList<>(members.size());
        for (int index = 0; index < members.size(); index++) {
            RequestEntry entry = requests.get(members.get(index).requestId());
            RouteReservation lease = leases.get(index);
            if (!unique.add(lease) || entry.reservation != lease
                    || lease.requestId != entry.requestId
                    || lease.state != LeaseState.OPEN
                    || openGenerationHandoff(lease) == null) {
                throw new IllegalStateException(
                        "route commit does not own exact OPEN lease request_id="
                                + entry.requestId);
            }
            committedHandoffs.add(new CommittedHandoff(
                    openGenerationHandoff(lease)));
        }
        List<CommittedHandoff> result = List.copyOf(committedHandoffs);
        long nowMs = clock.getAsLong();
        for (ScheduledRequest item : members) {
            removeValidatedActiveIndex(item);
        }
        for (int index = 0; index < members.size(); index++) {
            RouteReservation lease = leases.get(index);
            moveGenerationHandoffToOwnedUnderLock(
                    lease, committedHandoffs.get(index));
            RequestEntry entry = requests.get(members.get(index).requestId());
            entry.commitIndividual(lease, nowMs);
        }
        return result;
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
        CommittedHandoff committedHandoff =
                new CommittedHandoff(openGenerationHandoff(lease));
        for (ScheduledRequest item : members) {
            removeValidatedActiveIndex(item);
        }
        moveGenerationHandoffToOwnedUnderLock(lease, committedHandoff);
        for (ScheduledRequest item : members) {
            requests.get(item.requestId()).commitBatch(work);
        }
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

    public DirectRegistration tryRegisterDirect(
            long requestId,
            long predictedMs) {
        lock.lock();
        try {
            if (requests.containsKey(requestId)) {
                return null;
            }
            RequestEntry entry = new RequestEntry(
                    requestId, predictedMs, clock.getAsLong());
            requests.put(requestId, entry);
            return new DirectRegistration(entry);
        } finally {
            lock.unlock();
        }
    }

    private void rollbackDirectRegistration(RequestEntry entry) {
        lock.lock();
        try {
            if (!entry.isDirect()) {
                throw new IllegalStateException(
                        "DIRECT rollback capability lost its exact owner");
            }
            requests.remove(entry.requestId, entry);
        } finally {
            lock.unlock();
        }
    }

    /**
     * Total counterpart cleanup bound to one exact committed ScheduledRequest. A
     * reused request id or an ACTIVE item is a no-op.
     */
    public boolean terminalizeCommittedItem(ScheduledRequest exactItem) {
        ScheduledRequest item = exactItem;
        long requestId = item.requestId();
        boolean terminalized = false;
        boolean capacityReleased = false;
        lock.lock();
        try {
            RequestEntry entry = requests.get(requestId);
            if (entry == null || entry.isActive()
                    || entry.committedItem != item) {
                terminalized = false;
            } else {
                TerminalObservation terminal =
                        TerminalObservation.external(requestId);
                if (entry.protection != null) {
                    entry.deferredTerminal = entry.deferredTerminal == null
                            ? terminal
                            : entry.deferredTerminal.merge(terminal);
                } else {
                    BatchReduction reduction = entry.batchWork == null
                            ? null
                            : batchReductionUnderLock(entry.batchWork);
                    capacityReleased = settleUnderLock(
                            entry, terminal, reduction, new ArrayList<>(1));
                    invalidateRemainingBatchPredictionUnderLock(reduction);
                }
                terminalized = true;
            }
        } finally {
            lock.unlock();
            notifyCapacityAvailable(capacityReleased);
        }
        return terminalized;
    }

    public Protection tryAcquireProtection(ScheduledRequest exactItem) {
        ScheduledRequest item = exactItem;
        lock.lock();
        try {
            RequestEntry entry = requests.get(item.requestId());
            if (entry == null || entry.isActive()
                    || entry.batchWork != null
                    || entry.committedItem != item
                    || entry.protection != null) {
                return null;
            }
            Protection protection = new Protection(entry);
            entry.protection = protection;
            return protection;
        } finally {
            lock.unlock();
        }
    }

    public Protection tryAcquireBatchProtection(
            long batchId, ScheduledRequest exactItem) {
        ScheduledRequest item = exactItem;
        lock.lock();
        try {
            RequestEntry entry = requests.get(item.requestId());
            if (entry == null || entry.isActive()
                    || entry.batchWork == null
                    || entry.batchWork.lease.batchId != batchId
                    || entry.committedItem != item
                    || entry.protection != null) {
                return null;
            }
            Protection protection = new Protection(entry);
            entry.protection = protection;
            return protection;
        } finally {
            lock.unlock();
        }
    }

    public List<BatchCompletion> releaseProtection(
            Protection exactProtection,
            Function<List<ScheduledRequest>, OptionalLong> repredictor) {
        Protection protection = exactProtection;
        if (protection.owner != this) {
            throw new IllegalArgumentException(
                    "protection belongs to another Prefill ledger");
        }
        List<BatchCompletion> completions = new ArrayList<>(1);
        boolean capacityReleased = false;
        lock.lock();
        try {
            RequestEntry entry = protection.entry;
            if (requests.get(entry.requestId) != entry
                    || entry.protection != protection) {
                return List.of();
            }
            entry.protection = null;
            TerminalObservation deferred = entry.deferredTerminal;
            if (deferred != null) {
                BatchReduction reduction = entry.batchWork == null
                        ? null
                        : batchReductionUnderLock(entry.batchWork);
                capacityReleased = settleUnderLock(
                        entry, deferred, reduction, completions);
                refreshRemainingBatchPredictionUnderLock(
                        reduction, repredictor);
            }
        } finally {
            lock.unlock();
            notifyCapacityAvailable(capacityReleased);
        }
        return List.copyOf(completions);
    }

    /**
     * Reconcile ownership and publish its matching WorkerStatus before the
     * canonical queue lock is released. Projection readers therefore observe
     * either the previous pair or the fully reduced new pair.
     */
    public StatusReconciliation reconcileWorkerStatus(
            WorkerStatus.StatusObservation observation,
            Function<List<ScheduledRequest>, OptionalLong> repredictor,
            Runnable committedPublication,
            Runnable failedReduction) {
        return reconcileEngineStatus(
                observation.engine(),
                observation.finishedTasks(),
                repredictor,
                committedPublication,
                failedReduction);
    }

    public List<WorkerStatusFact> heartbeatFacts(
            WorkerStatus.StatusObservation observation) {
        List<WorkerStatusFact> facts = new ArrayList<>(
                observation.runningTasks().size());
        lock.lock();
        try {
            for (WorkerStatus.TaskObservation task
                    : observation.runningTasks().values()) {
                WorkerStatusFact fact = activeStatusFactUnderLock(task);
                if (fact != null) {
                    facts.add(fact);
                }
            }
        } finally {
            lock.unlock();
        }
        return List.copyOf(facts);
    }

    /**
     * Materialize every externally observable status fact before the first
     * canonical entry is settled. The returned object is the exact outcome
     * later published by the endpoint; reduction only binds a callback failure.
     */
    private StatusReconciliation prepareStatusReconciliationUnderLock(
            RoleType role,
            Map<Long, TerminalObservation> terminals,
            Map<String, WorkerStatus.TaskObservation> activeTasks,
            IdentityHashMap<BatchWork, BatchReduction> reductions) {
        requireLock();
        List<WorkerStatusFact> schedulerFacts = new ArrayList<>(
                terminals.size() + activeTasks.size());
        for (TerminalObservation terminal : terminals.values()) {
            WorkerStatusFact fact = terminalFact(
                    role, requests.get(terminal.requestId), terminal);
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
            Set<RequestEntry> terminalEntries,
            Function<List<ScheduledRequest>, OptionalLong> repredictor,
            long nowMs,
            IdentityHashMap<BatchWork, OptionalLong> predictions,
            IdentityHashMap<BatchWork, OptionalLong> remainingPredictions) {
        requireLock();
        for (BatchReduction reduction : changedBatches) {
            BatchWork batch = reduction.batch;
            List<ScheduledRequest> survivors = new ArrayList<>(
                    reduction.members.size());
            for (RequestEntry member : reduction.members) {
                if (!terminalEntries.contains(member)) {
                    survivors.add(member.committedItem);
                }
            }
            if (survivors.isEmpty()) {
                continue;
            }
            survivors.sort(Comparator.comparingLong(ScheduledRequest::enqueueSeq)
                    .thenComparingLong(ScheduledRequest::requestId));
            OptionalLong oldRemaining = batch.remainingAt(nowMs);
            OptionalLong prediction;
            try {
                prediction = repredictor.apply(survivors);
                if (prediction.isPresent() && prediction.getAsLong() < 0L) {
                    throw new IllegalArgumentException(
                            "batch reprediction must be non-negative");
                }
            } catch (Throwable predictionFailure) {
                try {
                    org.flexlb.util.Logger.error(
                            "Prefill batch reprediction failed; remaining work is unknown",
                            predictionFailure);
                } catch (Throwable ignoredLoggingFailure) {
                    // Optional prediction cannot block canonical settlement.
                }
                prediction = OptionalLong.empty();
            }
            OptionalLong newRemaining = OptionalLong.empty();
            if (prediction.isPresent()) {
                long remaining = prediction.getAsLong();
                if (batch.predictedWorkMs.isPresent()
                        && oldRemaining.isPresent()) {
                    long consumed = Math.max(
                            0L,
                            batch.predictedWorkMs.getAsLong()
                                    - oldRemaining.getAsLong());
                    remaining = Math.max(0L, remaining - consumed);
                }
                newRemaining = OptionalLong.of(remaining);
            }
            predictions.put(batch, prediction);
            remainingPredictions.put(batch, newRemaining);
        }
    }

    private long prepareActiveObservationsUnderLock(
            Map<String, WorkerStatus.TaskObservation> activeTasks,
            Map<Long, TerminalObservation> terminals,
            long reportedActive,
            long locallyOwnedAfterTerminals,
            IdentityHashMap<RequestEntry, Phase> individualPhases,
            IdentityHashMap<BatchWork, Phase> batchPhases) {
        requireLock();
        Set<Long> unknownDetailed = new HashSet<>();
        for (WorkerStatus.TaskObservation task : activeTasks.values()) {
            RequestEntry entry = terminals.containsKey(task.requestId())
                    ? null : requests.get(task.requestId());
            if (entry == null || entry.isActive()
                    || !matchesObservedBatch(entry, task.batchId())) {
                if (!isPriorityCancelOverlayOnly(task)) {
                    unknownDetailed.add(task.requestId());
                }
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
                0L, reportedActive - locallyOwnedAfterTerminals);
        return Math.max(unknownDetailed.size(), scalarUnknown);
    }

    private StatusReconciliation reconcileEngineStatus(
            WorkerStatus.EngineObservation engine,
            Map<String, WorkerStatus.TaskObservation> finishedTasks,
            Function<List<ScheduledRequest>, OptionalLong> repredictor,
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
            Set<RequestEntry> terminalEntries =
                    settlements.keySet();
            Set<BatchReduction> changedBatches = java.util.Collections.newSetFromMap(
                    new IdentityHashMap<>());
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
            IdentityHashMap<BatchWork, OptionalLong> predictions =
                    new IdentityHashMap<>();
            IdentityHashMap<BatchWork, OptionalLong> remainingPredictions =
                    new IdentityHashMap<>();
            prepareBatchPredictionsUnderLock(
                    changedBatches, terminalEntries, repredictor, nowMs,
                    predictions, remainingPredictions);
            long reportedActive = saturatedAdd(
                    Math.max(0L, engine.waitingQueryLen()),
                    Math.max(0L, engine.runningQueryLen()));
            long locallyOwnedAfterTerminals = Math.max(
                    0L,
                    locallyOwnedCountUnderLock() - terminalEntries.size());
            IdentityHashMap<RequestEntry, Phase> individualPhases =
                    new IdentityHashMap<>();
            IdentityHashMap<BatchWork, Phase> batchPhases =
                    new IdentityHashMap<>();
            long nextUnknownEngineRequestCount =
                    prepareActiveObservationsUnderLock(
                            activeTasks, terminals, reportedActive,
                            locallyOwnedAfterTerminals,
                            individualPhases, batchPhases);
            outcome = prepareStatusReconciliationUnderLock(
                    engine.role(), terminals, activeTasks, reductions);

            // Everything below this boundary is assignment/removal against
            // exact prevalidated identities. Any invariant failure is captured
            // into the already materialized outcome and forces retirement.
            canonicalMutationStarted = true;
            for (Map.Entry<RequestEntry, TerminalObservation> settlement
                    : settlements.entrySet()) {
                RequestEntry entry = settlement.getKey();
                // WorkerStatus is an authoritative Engine terminal. Protection
                // only fences TTL/external cleanup while ownership is ambiguous;
                // it must not defer the canonical Engine reducer. Invalidating
                // the exact lease here makes its later release a total no-op.
                entry.protection = null;
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
                batch.predictedWorkMs = prediction;
                batch.remainingWorkMs = remainingPredictions.get(batch);
                batch.phaseBaseMs = nowMs;
                batch.touch(nowMs);
            });
            long previousUnknownEngineRequestCount =
                    unknownEngineRequestCount;
            unknownEngineRequestCount = nextUnknownEngineRequestCount;
            capacityReleased |= unknownEngineRequestCount
                    < previousUnknownEngineRequestCount;
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
                if (!entry.isDirect()) {
                    ScheduledRequest item = entry.isActive()
                            ? entry.activeItem : entry.committedItem;
                    if (item == null) {
                        invariantFailure = appendRetirementInvariant(
                                invariantFailure,
                                "queued retirement owner has no ScheduledRequest request_id="
                                        + entry.requestId);
                    } else {
                        ownedItems.add(item);
                        if (entry.isActive()) {
                            if (entry.stopTerminalPending) {
                                if (activeIndex.contains(item)) {
                                    invariantFailure = appendRetirementInvariant(
                                            invariantFailure,
                                            "stop-pending ACTIVE owner remains indexed request_id="
                                                    + entry.requestId);
                                }
                            } else {
                                canonicalActive.add(item);
                            }
                        }
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

            int observedRouteLeases = 0;
            int observedBatchLeases = 0;
            for (Reservation lease : leases) {
                if (lease instanceof RouteReservation) {
                    observedRouteLeases++;
                } else if (lease instanceof BatchReservation) {
                    observedBatchLeases++;
                } else {
                    invariantFailure = appendRetirementInvariant(
                            invariantFailure,
                            "retirement found an unknown Prefill lease type");
                }
                if (lease.state == LeaseState.OPEN) {
                    if (lease.generationHandoff == null) {
                        invariantFailure = appendRetirementInvariant(
                                invariantFailure,
                                "OPEN Prefill lease lost its generation handoff");
                    } else {
                        orphanedHandoffs.add(lease.generationHandoff);
                        invariantFailure = appendRetirementInvariant(
                                invariantFailure,
                                "retirement reached an OPEN Prefill lease");
                    }
                } else if (lease.state != LeaseState.OWNED
                        || lease.generationHandoff != null) {
                    invariantFailure = appendRetirementInvariant(
                            invariantFailure,
                            "canonical retirement owner has an invalid lease state");
                }
            }
            if (routeLeasesInUse != observedRouteLeases) {
                invariantFailure = appendRetirementInvariant(
                        invariantFailure,
                        "route capacity does not match canonical retirement leases");
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
                entry.protection = null;
                entry.deferredTerminal = null;
                entry.individualPhase = null;
                entry.stopTerminalPending = false;
            }
            for (Reservation lease : leases) {
                lease.generationHandoff = null;
                lease.state = LeaseState.CLOSED;
            }
            requests.clear();
            activeIndex.clear();
            routeLeasesInUse = 0;
            batchLeasesInUse = 0;
            unknownEngineRequestCount = 0L;
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
                        && entry.protection == null
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
                    retained |= entry.protection != null
                            || schedulerOwnsRequest.test(entry.requestId);
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

    public long pendingRequestCount() {
        lock.lock();
        try {
            return saturatedAdd(
                    liveRequestCountUnderLock(),
                    unknownEngineRequestCount);
        } finally {
            lock.unlock();
        }
    }

    public Snapshot snapshotUnderLock(Comparator<ScheduledRequest> activeOrder) {
        requireLock();
        long nowMs = clock.getAsLong();
        List<ScheduledRequest> active = new ArrayList<>();
        for (RequestEntry entry : requests.values()) {
            if (entry.isActive()) {
                active.add(entry.activeItem);
            }
        }
        active.sort(activeOrder);
        return new Snapshot(
                nowMs,
                active,
                committedSnapshotUnderLock(nowMs),
                saturatedAdd(liveRequestCountUnderLock(),
                        unknownEngineRequestCount));
    }

    public WorkSnapshot committedSnapshot() {
        lock.lock();
        try {
            return committedSnapshotUnderLock(clock.getAsLong());
        } finally {
            lock.unlock();
        }
    }

    private WorkSnapshot committedSnapshotUnderLock(long nowMs) {
        requireLock();
        List<WorkSnapshot.RequestWork> individual = new ArrayList<>();
        IdentityHashMap<BatchWork, List<Long>> batchMembers =
                new IdentityHashMap<>();
        for (RequestEntry entry : requests.values()) {
            if (entry.isActive()) {
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
        individual.sort(Comparator.comparingLong(
                WorkSnapshot.RequestWork::requestId));
        List<WorkSnapshot.BatchWork> batches =
                new ArrayList<>(batchMembers.size());
        for (Map.Entry<BatchWork, List<Long>> observed : batchMembers.entrySet()) {
            BatchWork batch = observed.getKey();
            observed.getValue().sort(Long::compareTo);
            batches.add(new WorkSnapshot.BatchWork(
                    batch.lease.batchId,
                    observed.getValue(),
                    batch.servicePhase,
                    batch.remainingAt(nowMs)));
        }
        batches.sort(Comparator.comparingLong(
                WorkSnapshot.BatchWork::batchId));
        return new WorkSnapshot(
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
        entry.deferredTerminal = null;
        if (!requests.remove(entry.requestId, entry)) {
            throw new IllegalStateException(
                    "terminal request is not canonical request_id="
                            + entry.requestId);
        }
        if (lease != null) {
            closeOwnedLeaseUnderLock(lease);
        }
        if (lastBatchMember && completions != null) {
            completions.add(batch.completion());
        }
        return lease != null;
    }

    private void refreshRemainingBatchPredictionUnderLock(
            BatchReduction reduction,
            Function<List<ScheduledRequest>, OptionalLong> repredictor) {
        requireLock();
        if (reduction == null) {
            return;
        }
        BatchWork batch = reduction.batch;
        List<ScheduledRequest> survivors = reduction.liveItems();
        if (survivors.isEmpty()) {
            return;
        }
        long nowMs = clock.getAsLong();
        OptionalLong oldRemaining = batch.remainingAt(nowMs);
        OptionalLong prediction;
        try {
            prediction = repredictor.apply(survivors);
            if (prediction.isPresent() && prediction.getAsLong() < 0L) {
                throw new IllegalArgumentException(
                        "batch reprediction must be non-negative");
            }
        } catch (RuntimeException predictionFailure) {
            org.flexlb.util.Logger.error(
                    "Prefill batch reprediction failed; remaining work is unknown",
                    predictionFailure);
            prediction = OptionalLong.empty();
        }
        OptionalLong newRemaining = OptionalLong.empty();
        if (prediction.isPresent()) {
            long remaining = prediction.getAsLong();
            if (batch.predictedWorkMs.isPresent() && oldRemaining.isPresent()) {
                long consumed = Math.max(
                        0L,
                        batch.predictedWorkMs.getAsLong()
                                - oldRemaining.getAsLong());
                remaining = Math.max(0L, remaining - consumed);
            }
            newRemaining = OptionalLong.of(remaining);
        }
        batch.predictedWorkMs = prediction;
        batch.remainingWorkMs = newRemaining;
        batch.phaseBaseMs = nowMs;
        batch.touch(nowMs);
    }

    private void invalidateRemainingBatchPredictionUnderLock(
            BatchReduction reduction) {
        requireLock();
        if (reduction != null && !reduction.isEmpty()) {
            BatchWork batch = reduction.batch;
            long nowMs = clock.getAsLong();
            batch.predictedWorkMs = OptionalLong.empty();
            batch.remainingWorkMs = OptionalLong.empty();
            batch.phaseBaseMs = nowMs;
            batch.touch(nowMs);
        }
    }

    private Map<Long, TerminalObservation> terminalObservationsUnderLock(
            Map<String, WorkerStatus.TaskObservation> finishedTasks) {
        requireLock();
        Map<Long, TerminalObservation> terminals = new HashMap<>();
        for (WorkerStatus.TaskObservation task : finishedTasks.values()) {
            RequestEntry entry = requests.get(task.requestId());
            if (entry == null || entry.isActive()
                    || !matchesObservedBatch(entry, task.batchId())) {
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
            RoleType role,
            RequestEntry entry,
            TerminalObservation terminal) {
        if (entry == null || entry.committedItem == null
                || !terminal.workerObserved) {
            return null;
        }
        boolean fusionTerminal = role == RoleType.PDFUSION;
        if (terminal.errorCode == 0L && !fusionTerminal) {
            // A successful Prefill completion ends only the Prefill stage. The
            // canonical registry still settles it, but the P/D RequestSlot is
            // deliberately left live for Decode.
            return null;
        }
        WorkerStatusFact.Kind kind = terminal.errorCode == 0L
                ? WorkerStatusFact.Kind.COMPLETED
                : terminal.preemptionProgress
                        == PriorityPreemptionProgress.CANCELED
                    && terminal.errorCode == PRIORITY_PREEMPTED_ERROR_CODE
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

    private int liveRequestCountUnderLock() {
        return requests.size();
    }

    private int locallyOwnedCountUnderLock() {
        int count = 0;
        for (RequestEntry entry : requests.values()) {
            if (!entry.isActive()) {
                count++;
            }
        }
        return count;
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
        lock.lock();
        try {
            if (lease.state == LeaseState.OPEN) {
                generationHandoff = closeOpenLeaseUnderLock(lease);
            }
        } finally {
            lock.unlock();
        }
        if (generationHandoff == null) {
            return;
        }
        Throwable failure = null;
        try {
            generationHandoff.close();
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
            Reservation lease) {
        requireLock();
        return lease.generationHandoff;
    }

    /** Move the exact handoff out of the OPEN admission before commit publishes. */
    private void moveGenerationHandoffToOwnedUnderLock(
            Reservation lease,
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

    /** OPEN rollback closes both quota and the still-admission-owned handoff. */
    private EndpointGenerationLifecycle.HandoffPermit
            closeOpenLeaseUnderLock(Reservation lease) {
        requireLock();
        if (lease.state != LeaseState.OPEN
                || lease.generationHandoff == null) {
            throw new IllegalStateException(
                    "Prefill OPEN rollback lost its generation handoff");
        }
        RequestEntry owner = openLeaseOwnerUnderLock(lease);
        if (owner != null && !owner.isActive()) {
            throw new IllegalStateException(
                    "OPEN Prefill lease has a non-ACTIVE canonical owner");
        }
        EndpointGenerationLifecycle.HandoffPermit generationHandoff =
                lease.generationHandoff;
        lease.generationHandoff = null;
        lease.state = LeaseState.CLOSED;
        decrementLeaseCapacityUnderLock(lease);
        if (owner != null) {
            owner.reservation = null;
        }
        return generationHandoff;
    }

    /** OWNED terminal returns quota only; its handoff already has another owner. */
    private void closeOwnedLeaseUnderLock(Reservation lease) {
        requireLock();
        if (lease.state != LeaseState.OWNED
                || lease.generationHandoff != null) {
            throw new IllegalStateException(
                    "Prefill OWNED terminal still owns an admission handoff");
        }
        lease.state = LeaseState.CLOSED;
        decrementLeaseCapacityUnderLock(lease);
    }

    private void decrementLeaseCapacityUnderLock(Reservation lease) {
        requireLock();
        if (lease instanceof RouteReservation) {
            if (routeLeasesInUse <= 0) {
                throw new IllegalStateException(
                        "Prefill route capacity accounting underflow");
            }
            routeLeasesInUse--;
        } else if (lease instanceof BatchReservation) {
            if (batchLeasesInUse <= 0) {
                throw new IllegalStateException(
                        "Prefill batch capacity accounting underflow");
            }
            batchLeasesInUse--;
        } else {
            throw new IllegalStateException("unknown Prefill capacity lease type");
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
