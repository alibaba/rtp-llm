package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.DecisionGroupHandler;
import org.flexlb.balance.scheduler.BatchItem;
import org.flexlb.balance.scheduler.DeliveryCapacityAdmission;
import org.flexlb.balance.scheduler.InflightEvictor;
import org.flexlb.balance.scheduler.WorkerBatcher;
import org.flexlb.balance.strategy.FormulaPredictor;
import org.flexlb.balance.strategy.LearningPredictor;
import org.flexlb.balance.strategy.PrefillTimePredictor;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.RoutingConfig;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.dao.route.RoleType;
import org.flexlb.enums.PriorityPreemptionProgress;
import org.flexlb.enums.TaskPhase;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.lang.invoke.VarHandle;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
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
    private static final int COMBINED_WAIT_SNAPSHOT_MAX_ATTEMPTS = 4;

    enum WaitSnapshotStage {
        AFTER_REQUEST_QUEUED_READ,
        BEFORE_CACHE_PUBLISH
    }

    @FunctionalInterface
    interface WaitSnapshotHook {
        void onStage(WaitSnapshotStage stage);
    }

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

    private record BatchMemberProtection(FinishedObservation deferredTerminal) {}

    private final PrefillTimePredictor predictor;
    private final ConcurrentHashMap<Long, BatchInflight> inflightBatches = new ConcurrentHashMap<>();
    /** Reserved plus registered QUEUE batch slots; DIRECT wait records are excluded. */
    private final AtomicInteger queueBatchCapacityUsage = new AtomicInteger();
    private final EndpointGenerationLifecycle generationLifecycle =
            new EndpointGenerationLifecycle();
    private final ConcurrentHashMap<Long, ConcurrentHashMap<Long, BatchMemberProtection>>
            batchMemberProtections = new ConcurrentHashMap<>();
    /**
     * QUEUE batch members admitted out of ACTIVE, including callback-owned
     * members and members registered in a real {@link BatchInflight} lifecycle.
     */
    private final AtomicInteger admittedBatchRequestCount = new AtomicInteger();
    private final WorkerBatcher batcher;
    private final PrefillRequestLedger requestLedger;
    private final BatchSchedulerReporter reporter;

    /** Active Engine tasks not represented in either local lifecycle ledger. */
    private volatile long engineUntrackedRequestCount = 0;

    /**
     * Independent batch-ledger epoch used with the request-ledger epoch to form a
     * coherent combined wait snapshot without acquiring locks across ledgers.
     */
    private final AtomicLong batchWaitMutationsStarted = new AtomicLong();
    private final AtomicLong batchWaitMutationsCompleted = new AtomicLong();
    private final AtomicReference<WaitSnapshotHook> waitSnapshotHook = new AtomicReference<>();

    public PrefillEndpoint(WorkerStatus status, FlexlbConfig config,
                           DecisionGroupHandler decisionHandler,
                           DeliveryCapacityAdmission capacityAdmission,
                           BatchSchedulerReporter reporter) {
        super(status);
        this.reporter = reporter;
        this.predictor = createPredictor(config);
        this.batcher = createBatcher(
                config, decisionHandler, capacityAdmission, reporter);
        AtomicReference<WaitSnapshotHook> snapshotHook = waitSnapshotHook;
        this.requestLedger = new PrefillRequestLedger(
                batcher::signalDeliveryCapacityAvailable,
                System::currentTimeMillis,
                stage -> notifyWaitSnapshotHook(snapshotHook, stage));
        this.batcher.start();
    }

    private WorkerBatcher createBatcher(FlexlbConfig config,
                                        DecisionGroupHandler decisionHandler,
                                        DeliveryCapacityAdmission capacityAdmission,
                                        BatchSchedulerReporter reporter) {
        return new WorkerBatcher(status.getIpPort(), this, config,
                decisionHandler, capacityAdmission, reporter);
    }

    public WorkerBatcher getBatcher() {
        return batcher;
    }

    @Override
    public void close() {
        if (!generationLifecycle.tryBeginRetirement()) {
            if (generationLifecycle.currentThreadOwnsRetirement()
                    || generationLifecycle.currentThreadOwnsAcceptedHandoff()) {
                return;
            }
            generationLifecycle.awaitRetirement();
            return;
        }

        Throwable shutdownFailure = null;
        try {
            batcher.shutdown();
        } catch (Throwable failure) {
            shutdownFailure = failure;
        }

        Throwable retirementCause = shutdownFailure;
        generationLifecycle.runWhenAcceptedHandoffsDrain(
                () -> completeGenerationRetirement(retirementCause));
        if (generationLifecycle.currentThreadOwnsAcceptedHandoff()) {
            rethrowEndpointRetirementFailure(shutdownFailure);
            return;
        }
        generationLifecycle.awaitRetirement();
    }

    /** Runs exactly once, either on close or on the final handoff-releasing thread. */
    private void completeGenerationRetirement(Throwable initialFailure) {
        Throwable retirementFailure = initialFailure;
        try {
            requestLedger.retireDirectRequests();
        } catch (Throwable directRetirementFailure) {
            retirementFailure = appendRetirementFailure(
                    retirementFailure, directRetirementFailure);
        }
        try {
            super.close();
        } catch (Throwable endpointCloseFailure) {
            retirementFailure = appendRetirementFailure(
                    retirementFailure, endpointCloseFailure);
        } finally {
            generationLifecycle.completeRetirement(retirementFailure);
        }
    }

    private static Throwable appendRetirementFailure(
            Throwable first,
            Throwable next) {
        if (first == null) {
            return next;
        }
        if (first != next) {
            first.addSuppressed(next);
        }
        return first;
    }

    private static void rethrowEndpointRetirementFailure(Throwable failure) {
        if (failure instanceof RuntimeException runtimeFailure) {
            throw runtimeFailure;
        }
        if (failure instanceof Error error) {
            throw error;
        }
        if (failure != null) {
            throw new IllegalStateException(
                    "Prefill endpoint retirement failed", failure);
        }
    }

    public long batcherWaitMs() {
        return batcher.queueWaitMs();
    }

    /** Priority-aware estimate of additional fixed-window collection delay. */
    public long batcherEstimatedWaitMs(int priority, long requestId) {
        return batcher.queueManager().estimateWaitMs(priority, requestId);
    }

    private static PrefillTimePredictor createPredictor(FlexlbConfig config) {
        RoutingConfig.ExecutionTimeEstimatorConfig estimator = config.getRouter()
                .getRoles().getPrefill().getExecutionTimeEstimator();
        if (estimator instanceof RoutingConfig.LearningEstimatorConfig) {
            return new LearningPredictor();
        }
        RoutingConfig.FormulaEstimatorConfig formula =
                (RoutingConfig.FormulaEstimatorConfig) estimator;
        return new FormulaPredictor(formula.getExpression());
    }

    /** Result of reserving the endpoint half of one QUEUE batch admission. */
    public sealed interface QueueBatchSlotResult permits QueueBatchSlotReserved,
            QueueBatchSlotUnavailable, QueueBatchSlotAdmissionFailed {
    }

    public record QueueBatchSlotReserved(QueueBatchSlotReservation reservation)
            implements QueueBatchSlotResult {
        public QueueBatchSlotReserved {
            Objects.requireNonNull(reservation, "reservation");
        }
    }

    public record QueueBatchSlotUnavailable(
            DeliveryCapacityAdmission.CapacityAvailability availability)
            implements QueueBatchSlotResult {
        public QueueBatchSlotUnavailable {
            Objects.requireNonNull(availability, "availability");
        }
    }

    public record QueueBatchSlotAdmissionFailed(Throwable cause)
            implements QueueBatchSlotResult {
        public QueueBatchSlotAdmissionFailed {
            Objects.requireNonNull(cause, "cause");
        }
    }

    /** Atomically reserve the endpoint lifecycle slot for one QUEUE batch. */
    public QueueBatchSlotResult tryReserveQueueBatchSlot(
            BatchItem head,
            int maximumInflightBatches) {
        Objects.requireNonNull(head, "head");
        EndpointGenerationLifecycle.HandoffPermit handoffPermit =
                generationLifecycle.tryAcquireHandoff();
        if (handoffPermit == null) {
            return new QueueBatchSlotAdmissionFailed(
                    new EndpointGenerationRetiredException(
                            "Prefill endpoint generation retired before batch reservation"));
        }
        while (true) {
            int current = queueBatchCapacityUsage.get();
            if (current == Integer.MAX_VALUE
                    || (maximumInflightBatches > 0
                    && current >= maximumInflightBatches)) {
                handoffPermit.close();
                return new QueueBatchSlotUnavailable(
                        () -> generationLifecycle.isRetiringOrRetired()
                                || queueBatchCapacityUsage.get() < Integer.MAX_VALUE
                                && (maximumInflightBatches <= 0
                                || queueBatchCapacityUsage.get()
                                < maximumInflightBatches));
            }
            if (!queueBatchCapacityUsage.compareAndSet(current, current + 1)) {
                continue;
            }
            return new QueueBatchSlotReserved(
                    new QueueBatchSlotReservation(head, handoffPermit));
        }
    }

    /** Number of real QUEUE batch slots currently reserved or registered. */
    public int getQueueBatchCapacityUsage() {
        return queueBatchCapacityUsage.get();
    }

    /** Register one individually delivered DIRECT request in the request ledger. */
    public void registerDirectRequest(long requestId, long predictedMs) {
        EndpointGenerationLifecycle.HandoffPermit handoffPermit =
                generationLifecycle.tryAcquireHandoff();
        if (handoffPermit == null) {
            throw new EndpointGenerationRetiredException(
                    "Prefill endpoint generation retired before DIRECT registration");
        }
        try {
            if (!requestLedger.registerDirectRequest(requestId, predictedMs)) {
                throw new IllegalStateException(
                        "DIRECT request already has a live Prefill owner request_id="
                                + requestId);
            }
        } finally {
            handoffPermit.close();
        }
    }

    private void registerReservedQueueBatch(
            QueueBatchSlotReservation reservation,
            long batchId,
            long predictedMs,
            List<BatchItem> requests) {
        BatchInflight batch = new BatchInflight(
                predictedMs, requests, reservation::releaseFromBatchLifecycle);
        beginBatchWaitMutation();
        try {
            BatchInflight previous = inflightBatches.putIfAbsent(batchId, batch);
            if (previous != null) {
                throw new IllegalStateException(
                        "batch lifecycle already registered batch_id=" + batchId);
            }
        } finally {
            endBatchWaitMutation();
        }
    }

    private void releaseQueueBatchCapacityUnit() {
        int remaining = queueBatchCapacityUsage.decrementAndGet();
        if (remaining < 0) {
            queueBatchCapacityUsage.incrementAndGet();
            throw new IllegalStateException("QUEUE batch capacity released more than once");
        }
        batcher.signalDeliveryCapacityAvailable();
    }

    public final class QueueBatchSlotReservation {

        private enum State {
            RESERVED,
            BATCH_LIFECYCLE,
            RELEASED
        }

        private final BatchItem head;
        private final EndpointGenerationLifecycle.HandoffPermit handoffPermit;
        private State state = State.RESERVED;
        private List<BatchItem> admittedRequests = List.of();
        private boolean batchLoadPublicationOpen;
        private long registeredBatchId = -1L;
        private boolean batchLifecycleEstablished;
        private boolean deliveryHandoffComplete;

        private QueueBatchSlotReservation(
                BatchItem head,
                EndpointGenerationLifecycle.HandoffPermit handoffPermit) {
            this.head = head;
            this.handoffPermit = handoffPermit;
        }

        public BatchItem head() {
            return head;
        }

        /**
         * Publish callback-owned members before the active queue releases them.
         * Closing the publication after lifecycle transfer or abandonment
         * completes the coherent load transition.
         */
        public synchronized DeliveryCapacityAdmission.BatchLoadPublication
                beginBatchLoadPublication(List<BatchItem> requests) {
            if (state != State.RESERVED || !admittedRequests.isEmpty()
                    || batchLoadPublicationOpen) {
                throw new IllegalStateException(
                        "QUEUE batch callback load ownership was already established");
            }
            List<BatchItem> admitted = List.copyOf(requests);
            if (admitted.isEmpty()) {
                throw new IllegalArgumentException(
                        "QUEUE batch callback ownership requires requests");
            }
            AtomicBoolean closed = new AtomicBoolean();
            DeliveryCapacityAdmission.BatchLoadPublication publication = () -> {
                if (closed.compareAndSet(false, true)) {
                    synchronized (QueueBatchSlotReservation.this) {
                        batchLoadPublicationOpen = false;
                    }
                    endBatchWaitMutation();
                }
            };
            beginBatchWaitMutation();
            boolean published = false;
            try {
                admittedRequests = admitted;
                batchLoadPublicationOpen = true;
                admittedBatchRequestCount.addAndGet(admitted.size());
                published = true;
            } finally {
                if (!published) {
                    endBatchWaitMutation();
                }
            }
            return publication;
        }

        public synchronized void transferToBatchLifecycle(
                long batchId,
                long predictedMs,
                List<BatchItem> requests) {
            if (state != State.RESERVED) {
                throw new IllegalStateException(
                        "QUEUE batch capacity is not available for registration");
            }
            if (admittedRequests.isEmpty()) {
                throw new IllegalStateException(
                        "QUEUE batch load was not published before lifecycle transfer");
            }
            List<BatchItem> registeredRequests = List.copyOf(requests);
            validateRegisteredBatchMembers(registeredRequests);
            registerReservedQueueBatch(
                    this, batchId, predictedMs, registeredRequests);
            int membersNotRegistered = admittedRequests.size() - registeredRequests.size();
            if (membersNotRegistered > 0) {
                beginBatchWaitMutation();
                try {
                    admittedBatchRequestCount.addAndGet(-membersNotRegistered);
                } finally {
                    endBatchWaitMutation();
                }
            }
            // Membership validation and callback-load reconciliation are now
            // complete. The real BatchInflight owns the payload; the slot token
            // must not retain BatchItem graphs for the rest of the batch life.
            admittedRequests = List.of();
            registeredBatchId = batchId;
            state = State.BATCH_LIFECYCLE;
            batchLifecycleEstablished = true;
        }

        public synchronized void release() {
            if (state != State.RESERVED) {
                return;
            }
            state = State.RELEASED;
            try {
                if (!admittedRequests.isEmpty()) {
                    beginBatchWaitMutation();
                    try {
                        admittedBatchRequestCount.addAndGet(-admittedRequests.size());
                        admittedRequests = List.of();
                    } finally {
                        endBatchWaitMutation();
                    }
                }
                releaseQueueBatchCapacityUnit();
            } finally {
                completeDeliveryHandoffPermit();
            }
        }

        public synchronized void completeDeliveryHandoff() {
            if (!batchLifecycleEstablished) {
                throw new IllegalStateException(
                        "QUEUE batch lifecycle was not established");
            }
            completeDeliveryHandoffPermit();
        }

        private void completeDeliveryHandoffPermit() {
            if (!deliveryHandoffComplete) {
                deliveryHandoffComplete = true;
                handoffPermit.close();
            }
        }

        private synchronized void releaseFromBatchLifecycle() {
            if (state == State.RELEASED) {
                return;
            }
            if (state != State.BATCH_LIFECYCLE) {
                throw new IllegalStateException(
                        "QUEUE batch capacity has no registered batch lifecycle"
                                + " batch_id=" + registeredBatchId);
            }
            state = State.RELEASED;
            releaseQueueBatchCapacityUnit();
        }

        private void validateRegisteredBatchMembers(List<BatchItem> requests) {
            if (requests.isEmpty() || requests.size() > admittedRequests.size()) {
                throw new IllegalArgumentException(
                        "registered QUEUE batch must be a non-empty admitted subset");
            }
            int nextAdmittedIndex = 0;
            for (BatchItem registered : requests) {
                while (nextAdmittedIndex < admittedRequests.size()
                        && admittedRequests.get(nextAdmittedIndex) != registered) {
                    nextAdmittedIndex++;
                }
                if (nextAdmittedIndex == admittedRequests.size()) {
                    throw new IllegalArgumentException(
                            "registered QUEUE batch contains an unadmitted request");
                }
                nextAdmittedIndex++;
            }
        }
    }

    public void releaseBatch(long batchId) {
        long statusMs = System.currentTimeMillis();
        AtomicReference<BatchInflight> removedBatch = new AtomicReference<>();
        beginBatchWaitMutation();
        try {
            inflightBatches.compute(batchId, (id, batch) -> {
                // Keep the lock order consistent with protection/calibration:
                // inflight batch key first, batch-member protection key second.
                ConcurrentHashMap<Long, BatchMemberProtection> protectedRequests =
                        batchMemberProtections.get(id);
                if (batch == null || protectedRequests == null
                        || protectedRequests.isEmpty()) {
                    batchMemberProtections.remove(id);
                    if (batch != null) {
                        admittedBatchRequestCount.addAndGet(-batch.requests().size());
                        removedBatch.set(batch);
                    }
                    return null;
                }

                // A transport failure can race an Engine fence installation.
                // Retain exactly those members whose protection owner won the
                // batch-key linearization point; unprotected siblings are safe to
                // release immediately.
                List<BatchItem> survivors = batch.requests().stream()
                        .filter(item -> protectedRequests.containsKey(item.requestId()))
                        .toList();
                int removed = batch.requests().size() - survivors.size();
                if (survivors.isEmpty()) {
                    batchMemberProtections.remove(id, protectedRequests);
                    admittedBatchRequestCount.addAndGet(-batch.requests().size());
                    removedBatch.set(batch);
                    return null;
                }
                if (removed == 0) {
                    batch.touch(statusMs);
                    batch.observeFailure();
                    return batch;
                }
                batch.touch(statusMs);
                batch.observeFailure();
                long newPredMs = predictRepackedBatchMs(survivors);
                BatchInflight repacked = batch.repack(newPredMs, survivors);
                admittedBatchRequestCount.addAndGet(-removed);
                return repacked;
            });
        } finally {
            endBatchWaitMutation();
        }
        BatchInflight removed = removedBatch.get();
        if (removed != null) {
            removed.releaseCapacitySlot();
        }
    }

    /** Outcome of reserving one request-scoped Prefill delivery slot. */
    public enum RequestCapacityReservationStatus {
        ACQUIRED,
        CAPACITY_FULL,
        REQUEST_ALREADY_TRACKED,
        ENDPOINT_RETIRED
    }

    /** Explicit result of {@link #acquireRequestCapacityReservation}. */
    public record RequestCapacityReservationAcquisition(
            RequestCapacityReservationStatus status,
            RequestCapacityReservation reservation) {
        public RequestCapacityReservationAcquisition {
            Objects.requireNonNull(status, "status");
            if ((status == RequestCapacityReservationStatus.ACQUIRED)
                    != (reservation != null)) {
                throw new IllegalArgumentException(
                        "only ACQUIRED may carry a Prefill capacity reservation");
            }
        }
    }

    /**
     * Exact-entry token for capacity held before delivery becomes externally
     * visible. A composite admission prepares this entry first, performs its
     * final hard-capacity ownership transition, then transfers the token to
     * ordinary lifecycle ownership. Before that transfer, abort affects only
     * this exact generation.
     */
    public static final class RequestCapacityReservation {
        private enum State {
            RESERVED,
            PREPARED_FOR_DELIVERY,
            DELIVERY_OWNED,
            HANDOFF_COMPLETE,
            CLOSED
        }

        private final PrefillRequestLedger.RequestCapacityReservation delegate;
        private final EndpointGenerationLifecycle.HandoffPermit handoffPermit;
        private State state = State.RESERVED;

        private RequestCapacityReservation(
                PrefillRequestLedger.RequestCapacityReservation delegate,
                EndpointGenerationLifecycle.HandoffPermit handoffPermit) {
            this.delegate = delegate;
            this.handoffPermit = handoffPermit;
        }

        public synchronized boolean prepareForDelivery() {
            if (state != State.RESERVED) {
                return false;
            }
            boolean prepared = false;
            try {
                prepared = delegate.prepareForDelivery();
                state = prepared
                        ? State.PREPARED_FOR_DELIVERY
                        : State.CLOSED;
                return prepared;
            } finally {
                if (!prepared) {
                    handoffPermit.close();
                }
            }
        }

        public synchronized boolean release() {
            if (state != State.RESERVED) {
                return false;
            }
            state = State.CLOSED;
            try {
                return delegate.release();
            } finally {
                handoffPermit.close();
            }
        }

        public synchronized boolean abortBeforeDelivery() {
            if (state != State.RESERVED
                    && state != State.PREPARED_FOR_DELIVERY) {
                return false;
            }
            state = State.CLOSED;
            try {
                return delegate.abortBeforeDelivery();
            } finally {
                handoffPermit.close();
            }
        }

        public synchronized void completePreparedDeliveryTransfer() {
            if (state != State.PREPARED_FOR_DELIVERY) {
                throw new IllegalStateException(
                        "Prefill request capacity was not prepared for delivery");
            }
            delegate.completePreparedDeliveryTransfer();
            state = State.DELIVERY_OWNED;
        }

        public synchronized void completeDeliveryHandoff() {
            if (state == State.HANDOFF_COMPLETE || state == State.CLOSED) {
                return;
            }
            if (state != State.DELIVERY_OWNED) {
                throw new IllegalStateException(
                        "Prefill request delivery ownership was not established");
            }
            state = State.HANDOFF_COMPLETE;
            handoffPermit.close();
        }

    }

    /** Reserve the hard route-request slot without exposing the request. */
    public RequestCapacityReservationAcquisition acquireRequestCapacityReservation(
            long requestId, long predictMs, int maxPerWorker) {
        EndpointGenerationLifecycle.HandoffPermit handoffPermit =
                generationLifecycle.tryAcquireHandoff();
        if (handoffPermit == null) {
            return new RequestCapacityReservationAcquisition(
                    RequestCapacityReservationStatus.ENDPOINT_RETIRED, null);
        }
        PrefillRequestLedger.RequestCapacityReservationAcquisition acquisition;
        try {
            acquisition = requestLedger.acquireCapacityReservation(
                    requestId, predictMs, maxPerWorker);
        } catch (Throwable failure) {
            handoffPermit.close();
            throw failure;
        }
        RequestCapacityReservationStatus status = switch (acquisition.status()) {
            case ACQUIRED -> RequestCapacityReservationStatus.ACQUIRED;
            case CAPACITY_FULL -> RequestCapacityReservationStatus.CAPACITY_FULL;
            case REQUEST_ALREADY_TRACKED ->
                    RequestCapacityReservationStatus.REQUEST_ALREADY_TRACKED;
        };
        RequestCapacityReservation reservation;
        if (acquisition.reservation() == null) {
            handoffPermit.close();
            reservation = null;
        } else {
            reservation = new RequestCapacityReservation(
                    acquisition.reservation(), handoffPermit);
        }
        return new RequestCapacityReservationAcquisition(status, reservation);
    }

    /**
     * Idempotently release an individually-accounted request.
     *
     * @return {@code true} only when this call removed the live ledger entry
     */
    public boolean releaseRequest(long requestId) {
        return requestLedger.release(requestId);
    }

    /**
     * Protect one route-request ledger entry while an EngineFence reconciles
     * ambiguous delivery ownership.
     *
     * <p>The flag lives on the request entry and is mutated under the same fixed
     * stripe as progress, terminal settlement, and TTL eviction. There is no
     * auxiliary set to leak after an authoritative release/status terminal. This
     * method never acquires the batcher queue lock or calls back into the scheduler.
     *
     * @return {@code true} when the request is still locally accounted (including
     *         an already-protected request), otherwise {@code false}
     */
    public boolean beginEngineFenceProtection(long requestId) {
        return requestLedger.protect(requestId);
    }

    /**
     * End request-scoped EngineFence protection without refreshing its TTL age.
     * A request that was already released or authoritatively settled is a no-op.
     *
     * @return {@code true} only when a live protection flag was cleared
     */
    public boolean endEngineFenceProtection(long requestId) {
        return requestLedger.unprotect(requestId);
    }

    /** Advisory QUEUE_ROUTE snapshot; reservation acquisition is the hard gate. */
    public int availableRequestSlots(int maxPerWorker) {
        return requestLedger.available(maxPerWorker);
    }

    /**
     * Requests owned by a local Prefill lifecycle: admitted QUEUE batch members
     * plus individually tracked DIRECT and QUEUE_ROUTE requests.
     */
    public int getLocallyOwnedRequestCount() {
        return admittedBatchRequestCount.get() + requestLedger.count();
    }

    /** Individually-accounted DIRECT and QUEUE_ROUTE requests. */
    public int getIndividuallyTrackedRequestCount() {
        return requestLedger.count();
    }

    /** QUEUE route requests currently consuming the configured hard cap. */
    public int getQueueRouteCapacityUsage() {
        return requestLedger.queueRouteCount();
    }

    /**
     * Handle partial batch failure: remove failed requests from a batch and recompute prediction.
     *
     */
    public void repackBatch(long batchId, Set<Long> failedRequestIds) {
        long statusMs = System.currentTimeMillis();
        AtomicReference<BatchInflight> removedBatch = new AtomicReference<>();
        beginBatchWaitMutation();
        try {
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
                    admittedBatchRequestCount.addAndGet(-old.requests().size());
                    removedBatch.set(old);
                    return null; // removes entry from map
                }
                long newPredMs = predictRepackedBatchMs(survivors);
                BatchInflight repacked = old.repack(newPredMs, survivors);
                admittedBatchRequestCount.addAndGet(-removed);
                return repacked;
            });
        } finally {
            endBatchWaitMutation();
        }
        BatchInflight removed = removedBatch.get();
        if (removed != null) {
            removed.releaseCapacitySlot();
        }
    }

    @Override
    public void onWorkerStatusUpdate(WorkerStatus ws, WorkerStatusResponse resp) {
        try {
            super.onWorkerStatusUpdate(ws, resp);
            Set<Long> activeRequestsOutsideRequestLedger =
                    calibrate(resp.getFinishedTaskInfo(), resp.getRunningTaskInfo());
            updateEngineUntrackedRequestCount(
                    resp, activeRequestsOutsideRequestLedger);
        } finally {
            // WorkerStatus-derived capacity is an advisory scheduling input.
            // Predictor model updates publish their own change at the exact
            // learning boundary, including updates outside this status path.
            batcher.signalSchedulingInputsChanged();
        }
    }

    /**
     * Full calibration against worker status report.
     */
    private Set<Long> calibrate(Map<String, TaskInfo> finishedTaskInfo,
                                Map<String, TaskInfo> runningTaskInfo) {
        long statusMs = System.currentTimeMillis();

        int finishedSize = finishedTaskInfo != null ? finishedTaskInfo.size() : 0;
        int runningSize = runningTaskInfo != null ? runningTaskInfo.size() : 0;
        if (logger.isDebugEnabled()
                && (finishedSize > 0 || !inflightBatches.isEmpty() || requestLedger.count() > 0)) {
            logger.debug("Prefill calibrate: finishedTasks={}, runningTasks={}, "
                            + "inflightBatches={}, individuallyTrackedRequests={}",
                    finishedSize, runningSize, inflightBatches.size(), requestLedger.count());
        }

        // Phase 1: settle individually tracked requests directly by request id,
        // then collect
        // terminal observations owned by real batches. Checking the request ledger
        // first also tolerates Engine versions that attach a synthetic batch id to an
        // individually submitted request.
        Map<Long, List<FinishedObservation>> finishedByBatch = new HashMap<>();

        if (finishedTaskInfo != null) {
            for (TaskInfo task : finishedTaskInfo.values()) {
                if (task == null) {
                    continue;
                }
                if (settleRequest(task)) {
                    continue;
                }
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
        Set<Long> activeRequestsOutsideRequestLedger = new HashSet<>();
        if (runningTaskInfo != null) {
            for (TaskInfo task : runningTaskInfo.values()) {
                if (task == null) {
                    continue;
                }
                if (observeRequestProgress(task, statusMs)) {
                    continue;
                }
                if (!isPriorityCancelOverlayOnly(task)) {
                    activeRequestsOutsideRequestLedger.add(task.getRequestId());
                }
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
                for (TaskInfo task : entry.getValue()) {
                    if (!currentRequestIds.contains(task.getRequestId())) {
                        continue;
                    }
                    observedCurrentMember = true;
                    observedRunningMember |= task.getPhase() == TaskPhase.RUNNING;
                }
                if (!observedCurrentMember) {
                    return batch;
                }
                beginBatchWaitMutation();
                try {
                    if (observedRunningMember) {
                        batch.markRunning(statusMs);
                    } else {
                        batch.markQueued(statusMs);
                    }
                } finally {
                    endBatchWaitMutation();
                }
                return batch;
            });
        }

        // Phase 4: check active requests outside the request ledger for anomalies.
        for (Map.Entry<Long, List<TaskInfo>> entry : activeByBatch.entrySet()) {
            if (!inflightBatches.containsKey(entry.getKey())) {
                for (TaskInfo task : entry.getValue()) {
                    logger.debug("Prefill calibrate: running request reqId={} batchId={} not in inflight",
                            task.getRequestId(), entry.getKey());
                }
            }
        }
        return activeRequestsOutsideRequestLedger;
    }

    private boolean settleRequest(TaskInfo task) {
        long requestId = task.getRequestId();
        if (!requestLedger.settle(requestId)) {
            return false;
        }
        if (task.getErrorCode() != 0) {
            logger.debug("Prefill calibrate: request failure reqId={} error={}",
                    requestId, task.getErrorMessage());
        }
        return true;
    }

    private boolean observeRequestProgress(TaskInfo task, long statusMs) {
        return requestLedger.observe(task.getRequestId(),
                task.getPhase() == TaskPhase.RUNNING, statusMs)
                != PrefillRequestLedger.ProgressOwnership.NOT_TRACKED;
    }

    private void settleFinishedMembers(long batchId,
                                       List<FinishedObservation> observations,
                                       long statusMs) {
        AtomicReference<BatchInflight> completed = new AtomicReference<>();
        beginBatchWaitMutation();
        try {
            inflightBatches.computeIfPresent(batchId, (id, batch) ->
                    applyFinishedObservations(
                            id, batch, observations, statusMs, true, completed));
        } finally {
            endBatchWaitMutation();
        }

        BatchInflight completedBatch = completed.get();
        if (completedBatch != null) {
            completedBatch.releaseCapacitySlot();
            reportBatchCompletion(batchId, completedBatch);
        }
    }

    private BatchInflight applyFinishedObservations(long batchId,
                                                     BatchInflight batch,
                                                     List<FinishedObservation> observations,
                                                     long statusMs,
                                                     boolean deferProtectedMembers,
                                                     AtomicReference<BatchInflight> completed) {
        Set<Long> localRequestIds = batch.requests().stream()
                .map(BatchItem::requestId)
                .collect(Collectors.toSet());
        Set<Long> finishedIds = new HashSet<>();
        int foreignCount = 0;

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
            if (deferProtectedMembers
                    && deferIfBatchMemberProtected(batchId, observation)) {
                continue;
            }

            finishedIds.add(requestId);
            if (observation.errorCode() != 0) {
                logger.debug("Prefill calibrate: batch failure batchId={} reqId={} error={}",
                        batchId, requestId, observation.errorMessage());
            }
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
        int finishedMemberCount = batch.requests().size() - survivors.size();
        if (survivors.isEmpty()) {
            admittedBatchRequestCount.addAndGet(-finishedMemberCount);
            completed.set(batch);
            return null;
        }

        long newPredMs = predictRepackedBatchMs(survivors);
        BatchInflight repackedBatch = batch.repack(newPredMs, survivors);
        // All operations that can fail complete before the ledger count changes.
        // The surrounding mutation epoch keeps readers from observing the small
        // counter-to-map publication interval inside ConcurrentHashMap.compute.
        admittedBatchRequestCount.addAndGet(-finishedMemberCount);
        return repackedBatch;
    }

    /**
     * Membership settlement must not depend on the optional cost estimator.
     * If prediction fails, the batch still loses the finished members while
     * its wait estimate becomes unavailable. No monotonicity is assumed from
     * a predictor, so only {@link Long#MAX_VALUE} is a guaranteed non-low
     * replacement.
     */
    private long predictRepackedBatchMs(List<BatchItem> survivingRequests) {
        try {
            return (long) predictor.predictBatchMs(survivingRequests);
        } catch (Throwable predictionFailure) {
            logger.error("Prefill batch repack prediction failed; marking wait unavailable "
                            + "engine={} surviving_requests={}",
                    getIp(), survivingRequests.size(), predictionFailure);
            return Long.MAX_VALUE;
        }
    }

    /**
     * Reconcile a finished task whose Engine status omitted the original batch id.
     *
     * <p>Individually delivered requests were already reconciled through the
     * request ledger. Production priority-cancel terminals may report
     * {@code batch_id=-1} even though the Master committed the request as a member
     * of a real batch. In that case scan the live ledger for the unique owning
     * batch and remove only the matching member. The member is revalidated inside
     * the map compute, so a concurrent release/repack/TTL eviction is an idempotent
     * no-op rather than a counter double-decrement. No reverse index is retained,
     * keeping every existing ledger mutation path consistent automatically.
     */
    private void reconcileFinishedWithoutBatchId(FinishedObservation observation, long statusMs) {
        long requestId = observation.requestId();
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

    private void updateEngineUntrackedRequestCount(
            WorkerStatusResponse response,
            Set<Long> activeRequestsOutsideRequestLedger) {
        // Real batches are few and need a membership set. Individual-request
        // membership was
        // classified while applying the same WorkerStatus observation, avoiding
        // a second request-ledger lookup or a copy of all live route ids.
        Set<Long> localBatchRequestIds = new HashSet<>();
        for (BatchInflight batch : inflightBatches.values()) {
            for (BatchItem request : batch.requests()) {
                localBatchRequestIds.add(request.requestId());
            }
        }

        activeRequestsOutsideRequestLedger.removeAll(localBatchRequestIds);

        long reportedActive = Math.max(0, response.getWaitingQueryLen())
                + Math.max(0, response.getRunningQueryLen());
        long scalarLowerBound = Math.max(
                0, reportedActive - Math.max(0, getLocallyOwnedRequestCount()));
        // The protobuf converter represents an absent detail list as an empty map,
        // while older/newer Engine variants may still populate only the scalar
        // counts. Keep the request-id union when details exist and conservatively
        // retain the scalar lower bound when the detail list is empty or partial.
        engineUntrackedRequestCount = Math.max(
                activeRequestsOutsideRequestLedger.size(), scalarLowerBound);
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
        for (int attempt = 0; attempt < COMBINED_WAIT_SNAPSHOT_MAX_ATTEMPTS; attempt++) {
            long batchVersionBefore = batchWaitMutationVersion();
            if (batchVersionBefore < 0) {
                Thread.onSpinWait();
                continue;
            }
            long locallyOwned = getLocallyOwnedRequestCount();
            long activeQueue = batcher.queueSize();
            long untrackedEngine = engineUntrackedRequestCount;
            long batchVersionAfter = batchWaitMutationVersion();
            if (batchVersionBefore == batchVersionAfter) {
                return saturatedAdd(
                        saturatedAdd(locallyOwned, activeQueue),
                        Math.max(0L, untrackedEngine));
            }
            Thread.onSpinWait();
        }
        // Continuous ownership transfer cannot yield a coherent non-blocking
        // snapshot. Route away conservatively instead of publishing a torn low
        // Prefill load.
        return Long.MAX_VALUE;
    }

    // ==================== Wait Time ====================

    /**
     * Real wait time: combined estimate for batch and individual request ledgers.
     */
    public long realWaitTimeMs() {
        return estimateWaitingTimeMs(System.currentTimeMillis());
    }

    public int getInflightBatchCount() {
        return inflightBatches.size();
    }

    /**
     * Evict inflight batches not observed for longer than {@code ttlMs}.
     * Called periodically by the scheduler to clean up stale prefill entries.
     *
     * @return number of batches evicted
     */
    public int evictExpiredBatches(long ttlMs) {
        return evictExpiredBatches(ttlMs, ignored -> false);
    }

    /** Evict only batches with no request generation still owned by the scheduler. */
    public int evictExpiredBatches(long ttlMs,
                                   LongPredicate schedulerOwnsRequest) {
        long nowMs = System.currentTimeMillis();
        AtomicInteger evictedCount = new AtomicInteger();
        for (Long batchId : inflightBatches.keySet()) {
            BatchInflight candidate = inflightBatches.get(batchId);
            if (candidate == null
                    || hasProtectedBatchMember(batchId)
                    || batchHasOwnedRequest(candidate, schedulerOwnsRequest)
                    || nowMs - candidate.lastObservedAtMs() <= ttlMs) {
                continue;
            }
            AtomicReference<BatchInflight> evicted = new AtomicReference<>();
            beginBatchWaitMutation();
            try {
                inflightBatches.computeIfPresent(batchId, (id, batch) -> {
                    if (hasProtectedBatchMember(id)
                            || batchHasOwnedRequest(batch, schedulerOwnsRequest)
                            || nowMs - batch.lastObservedAtMs() <= ttlMs) {
                        return batch;
                    }
                    evicted.set(batch);
                    return null;
                });
            } finally {
                endBatchWaitMutation();
            }
            BatchInflight removed = evicted.get();
            if (removed != null) {
                admittedBatchRequestCount.addAndGet(-removed.requests().size());
                removed.releaseCapacitySlot();
                evictedCount.incrementAndGet();
            }
        }
        return evictedCount.get();
    }

    private static boolean batchHasOwnedRequest(
            BatchInflight batch, LongPredicate schedulerOwnsRequest) {
        for (BatchItem item : batch.requests()) {
            if (schedulerOwnsRequest.test(item.requestId())) {
                return true;
            }
        }
        return false;
    }

    /**
     * Evict individually-accounted requests that have not appeared in WorkerStatus
     * for longer than {@code ttlMs}.
     *
     * <p>The stale check is repeated while holding the request's stripe. Progress
     * observation, explicit release, and TTL removal are therefore linearizable and
     * an observation racing the first optimistic check cannot be evicted as stale.
     */
    public int evictExpiredRequests(long ttlMs) {
        return requestLedger.evict(ttlMs);
    }

    /** Evict route-request entries which have no live scheduler generation. */
    public int evictExpiredRequests(long ttlMs,
                                    LongPredicate schedulerOwnsRequest) {
        return requestLedger.evict(ttlMs, schedulerOwnsRequest);
    }

    /** Evict stale entries from both Prefill accounting ledgers. */
    public int evictExpiredInflight(long ttlMs) {
        return evictExpiredBatches(ttlMs) + evictExpiredRequests(ttlMs);
    }

    /** Evict endpoint orphans without racing scheduler-owned generations. */
    public int evictExpiredInflight(long ttlMs,
                                    LongPredicate schedulerOwnsRequest) {
        return evictExpiredBatches(ttlMs, schedulerOwnsRequest)
                + evictExpiredRequests(ttlMs, schedulerOwnsRequest);
    }

    /**
     * Protect an ACK-ambiguous batch member from settlement and age-only eviction.
     *
     * <p>The ownership check and guard publication run inside the same
     * {@code inflightBatches} key computation used by WorkerStatus settlement,
     * explicit release, and TTL revalidation. A {@code true} result therefore
     * proves that every later destructive mutation for this batch observes the
     * guard; {@code false} proves that settlement won first or the request did not
     * belong to that batch, and no guard was installed.</p>
     */
    public boolean tryProtectBatchMember(long batchId, long requestId) {
        long nowMs = System.currentTimeMillis();
        AtomicBoolean protectedMember = new AtomicBoolean();
        inflightBatches.computeIfPresent(batchId, (id, batch) -> {
            boolean owned = false;
            for (BatchItem item : batch.requests()) {
                if (item.requestId() == requestId) {
                    owned = true;
                    break;
                }
            }
            if (!owned) {
                return batch;
            }
            batchMemberProtections.compute(id, (ignored, requests) -> {
                ConcurrentHashMap<Long, BatchMemberProtection> states = requests != null
                        ? requests : new ConcurrentHashMap<>();
                states.putIfAbsent(requestId, new BatchMemberProtection(null));
                return states;
            });
            batch.touch(nowMs);
            protectedMember.set(true);
            return batch;
        });
        return protectedMember.get();
    }

    /** Release one batch member's protection after authoritative settlement. */
    public void releaseBatchMemberProtection(long batchId, long requestId) {
        long statusMs = System.currentTimeMillis();
        AtomicReference<BatchInflight> completed = new AtomicReference<>();
        beginBatchWaitMutation();
        try {
            inflightBatches.compute(batchId, (id, batch) -> {
                AtomicReference<FinishedObservation> deferredTerminal = new AtomicReference<>();
                batchMemberProtections.computeIfPresent(id, (ignored, requests) -> {
                    BatchMemberProtection state = requests.remove(requestId);
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
                // The protection was removed under the same inflight-key critical section,
                // so apply the cached terminal directly instead of trying to defer it again.
                return applyFinishedObservations(
                        id, batch, List.of(observation), statusMs, false, completed);
            });
        } finally {
            endBatchWaitMutation();
        }
        BatchInflight completedBatch = completed.get();
        if (completedBatch != null) {
            completedBatch.releaseCapacitySlot();
            reportBatchCompletion(batchId, completedBatch);
        }
    }

    private boolean deferIfBatchMemberProtected(
            long batchId, FinishedObservation observation) {
        AtomicBoolean deferred = new AtomicBoolean(false);
        batchMemberProtections.computeIfPresent(batchId, (ignored, requests) -> {
            requests.computeIfPresent(observation.requestId(), (requestId, state) -> {
                deferred.set(true);
                FinishedObservation existing = state.deferredTerminal();
                return new BatchMemberProtection(existing == null
                        ? observation : existing.merge(observation));
            });
            return requests.isEmpty() ? null : requests;
        });
        return deferred.get();
    }

    private boolean hasProtectedBatchMember(long batchId) {
        ConcurrentHashMap<Long, BatchMemberProtection> requests =
                batchMemberProtections.get(batchId);
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
     * Called periodically by {@link org.flexlb.balance.scheduler.PriorityScheduler}.
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
        reporter.reportInflightRequestCount(RoleType.PREFILL.name(), getIp(), getLocallyOwnedRequestCount());
        long nowMs = System.currentTimeMillis();
        long maxAgeMs = Math.max(
                InflightEvictor.maxAgeMs(inflightBatches, nowMs),
                requestLedger.maxAge(nowMs));
        reporter.reportInflightMaxAgeMs(RoleType.PREFILL.name(), getIp(), maxAgeMs);
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
            try {
                PrefillTimePredictor.LearningResult learningResult = predictor.learn(
                        batch.originalFeatures(), predictedMs, actualMs);
                if (learningResult
                        == PrefillTimePredictor.LearningResult.MODEL_UPDATED) {
                    batcher.signalSchedulingInputsChanged();
                }
            } catch (RuntimeException learningFailure) {
                logger.warn("batch predictor learning failed after settlement: batchId={} engine={}",
                        batchId, getIp(), learningFailure);
            }
        }

        // These are post-settlement observers. Isolate them individually so
        // a metrics outage cannot suppress the scheduler's WorkerStatus
        // reducer or prevent the remaining observations.
        try {
            reporter.reportBatchPredictedTimeMs(RoleType.PREFILL.name(), getIp(), predictedMs);
        } catch (RuntimeException telemetryFailure) {
            logger.warn("batch predicted-time metric failed: batchId={} engine={}",
                    batchId, getIp(), telemetryFailure);
        }
        try {
            reporter.reportBatchActualTimeMs(RoleType.PREFILL.name(), getIp(), actualMs);
        } catch (RuntimeException telemetryFailure) {
            logger.warn("batch actual-time metric failed: batchId={} engine={}",
                    batchId, getIp(), telemetryFailure);
        }
        try {
            reporter.reportBatchPredictGapMs(RoleType.PREFILL.name(), getIp(), gapMs);
        } catch (RuntimeException telemetryFailure) {
            logger.warn("batch prediction-gap metric failed: batchId={} engine={}",
                    batchId, getIp(), telemetryFailure);
        }
    }

    private long estimateWaitingTimeMs(long nowMs) {
        for (int attempt = 0; attempt < COMBINED_WAIT_SNAPSHOT_MAX_ATTEMPTS; attempt++) {
            long batchVersionBefore = batchWaitMutationVersion();
            long requestVersionBefore = requestLedger.mutationVersion();
            if (batchVersionBefore < 0 || requestVersionBefore < 0) {
                Thread.onSpinWait();
                continue;
            }

            long batchWaitMs = computeBatchWaitingTimeMs(nowMs);
            long requestWaitMs = requestLedger.estimate(nowMs);
            if (requestWaitMs == Long.MAX_VALUE) {
                return Long.MAX_VALUE;
            }

            // Validate in reverse order so both component reads are bracketed by
            // their own monotonic epochs. This stays lock-free across ledgers while
            // preventing a BATCH/QUEUE transition from publishing a torn low sum.
            long requestVersionAfter = requestLedger.mutationVersion();
            long batchVersionAfter = batchWaitMutationVersion();
            if (batchVersionBefore == batchVersionAfter
                    && requestVersionBefore == requestVersionAfter) {
                return saturatedAdd(batchWaitMs, requestWaitMs);
            }
            Thread.onSpinWait();
        }

        // Continuous cross-ledger mutation cannot produce a coherent bounded
        // snapshot without blocking admission. Route away conservatively.
        return Long.MAX_VALUE;
    }

    private long computeBatchWaitingTimeMs(long nowMs) {
        if (inflightBatches.isEmpty()) {
            return 0;
        }
        long batchPredMs = 0;
        long earliestBatchProgressBaseMs = Long.MAX_VALUE;
        for (BatchInflight batch : inflightBatches.values()) {
            batchPredMs = saturatedAdd(batchPredMs, Math.max(0, batch.predictTimeMs()));
            if (batchPredMs == Long.MAX_VALUE) {
                return Long.MAX_VALUE;
            }
            // An inflight batch begins aging at lifecycle commit and
            // WorkerStatus may subsequently re-anchor it.
            earliestBatchProgressBaseMs = Math.min(
                    earliestBatchProgressBaseMs, batch.progressBaseMs());
        }
        long batchWaitMs = earliestBatchProgressBaseMs == Long.MAX_VALUE
                ? 0
                : Math.max(0, batchPredMs
                        - Math.max(0, nowMs - earliestBatchProgressBaseMs));
        return batchWaitMs;
    }

    private void beginBatchWaitMutation() {
        batchWaitMutationsStarted.incrementAndGet();
        VarHandle.storeStoreFence();
    }

    private void endBatchWaitMutation() {
        VarHandle.storeStoreFence();
        batchWaitMutationsCompleted.incrementAndGet();
    }

    private long batchWaitMutationVersion() {
        VarHandle.loadLoadFence();
        long started = batchWaitMutationsStarted.get();
        long completed = batchWaitMutationsCompleted.get();
        return started == completed ? completed : -1;
    }

    /** Package-private deterministic interleaving hook; always null in production. */
    void setWaitSnapshotHookForTest(WaitSnapshotHook hook) {
        waitSnapshotHook.set(hook);
    }

    private static void notifyWaitSnapshotHook(
            AtomicReference<WaitSnapshotHook> hookReference,
            PrefillRequestLedger.WaitSnapshotStage ledgerStage) {
        WaitSnapshotHook hook = hookReference.get();
        if (hook != null) {
            WaitSnapshotStage endpointStage = switch (ledgerStage) {
                case AFTER_QUEUED_READ -> WaitSnapshotStage.AFTER_REQUEST_QUEUED_READ;
                case BEFORE_CACHE_PUBLISH -> WaitSnapshotStage.BEFORE_CACHE_PUBLISH;
            };
            hook.onStage(endpointStage);
        }
    }

    private static long saturatedAdd(long left, long right) {
        return left > Long.MAX_VALUE - right ? Long.MAX_VALUE : left + right;
    }

}
