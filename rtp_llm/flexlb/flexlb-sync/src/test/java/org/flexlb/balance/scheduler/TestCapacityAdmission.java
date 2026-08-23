package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillEndpoint;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Supplier;

/** Test fixtures for the explicit capacity-first delivery protocol. */
public final class TestCapacityAdmission {

    private static final AtomicLong TEST_BATCH_IDS = new AtomicLong(1_000_000L);

    private TestCapacityAdmission() {
    }

    /**
     * Capacity admission for tests whose subject does not include endpoint
     * accounting. Every request receives its own stateful reservation.
     */
    public static DeliveryCapacityAdmission alwaysAvailable() {
        return new DeliveryCapacityAdmission() {
            @Override
            public AdmissionResult tryReserveItemCapacity(BatchItem item) {
                return new CapacityReserved(new TestCapacityReservation(item));
            }

            @Override
            public BatchCapacityResult tryReserveBatchCapacity(BatchItem head) {
                return new BatchCapacityReserved(
                        new TestBatchCapacityReservation(head));
            }
        };
    }

    /**
     * Per-request test reservations plus the endpoint's real QUEUE group-slot
     * accounting. The supplier supports endpoint construction cycles in tests.
     */
    public static DeliveryCapacityAdmission withEndpointBatchCapacity(
            Supplier<PrefillEndpoint> endpointSupplier) {
        Objects.requireNonNull(endpointSupplier, "endpointSupplier");
        DeliveryCapacityAdmission itemCapacity = alwaysAvailable();
        return new DeliveryCapacityAdmission() {
            @Override
            public AdmissionResult tryReserveItemCapacity(BatchItem item) {
                return itemCapacity.tryReserveItemCapacity(item);
            }

            @Override
            public BatchCapacityResult tryReserveBatchCapacity(BatchItem head) {
                PrefillEndpoint endpoint = Objects.requireNonNull(
                        endpointSupplier.get(), "test Prefill endpoint");
                return adaptEndpointBatchSlot(head,
                        endpoint.tryReserveQueueBatchSlot(
                                head, head.maxInflightBatchesPerPrefillWorker()));
            }
        };
    }

    /** Route a scheduler-mock group admission to the head's real endpoint. */
    public static DeliveryCapacityAdmission.BatchCapacityResult
            tryReserveEndpointBatchCapacity(BatchItem head) {
        PrefillEndpoint endpoint = Objects.requireNonNull(
                head.prefillEp(), "batch head Prefill endpoint");
        return adaptEndpointBatchSlot(head,
                endpoint.tryReserveQueueBatchSlot(
                        head, head.maxInflightBatchesPerPrefillWorker()));
    }

    /** Build a callback payload backed by test-only reservations. */
    public static AdmittedDecisionGroup admit(List<BatchItem> items) {
        return admit(alwaysAvailable(), items);
    }

    /**
     * Register a real QUEUE batch lifecycle through the endpoint's group-slot
     * reservation contract. Endpoint tests use this instead of bypassing hard
     * capacity accounting with a production fallback.
     */
    public static void registerQueueBatchLifecycle(
            PrefillEndpoint endpoint,
            long batchId,
            long predictedMs,
            List<BatchItem> requests) {
        if (requests == null || requests.isEmpty()) {
            throw new IllegalArgumentException(
                    "QUEUE batch fixture requires at least one request");
        }
        BatchItem head = requests.get(0);
        PrefillEndpoint.QueueBatchSlotResult result =
                endpoint.tryReserveQueueBatchSlot(
                        head, head.maxInflightBatchesPerPrefillWorker());
        if (!(result instanceof PrefillEndpoint.QueueBatchSlotReserved reserved)) {
            throw new IllegalStateException(
                    "test QUEUE batch was not capacity-admitted: " + result);
        }
        try {
            try (DeliveryCapacityAdmission.BatchLoadPublication ignored =
                         reserved.reservation().beginBatchLoadPublication(requests)) {
                // The fixture has no active queue; closing publishes the other
                // side of the same ownership transition.
            }
            reserved.reservation().transferToBatchLifecycle(
                    batchId, predictedMs, requests);
            reserved.reservation().completeDeliveryHandoff();
        } catch (RuntimeException | Error failure) {
            try {
                reserved.reservation().release();
            } catch (Throwable cleanupFailure) {
                failure.addSuppressed(cleanupFailure);
            }
            throw failure;
        }
    }

    private static DeliveryCapacityAdmission.BatchCapacityResult adaptEndpointBatchSlot(
            BatchItem head,
            PrefillEndpoint.QueueBatchSlotResult result) {
        if (result instanceof PrefillEndpoint.QueueBatchSlotReserved reserved) {
            return new DeliveryCapacityAdmission.BatchCapacityReserved(
                    new EndpointBackedTestBatchCapacityReservation(
                            head, reserved.reservation()));
        }
        if (result instanceof PrefillEndpoint.QueueBatchSlotUnavailable unavailable) {
            return new DeliveryCapacityAdmission.BatchCapacityUnavailable(
                    DeliveryCapacityAdmission.CapacityResource.PREFILL_BATCH,
                    unavailable.availability());
        }
        return new DeliveryCapacityAdmission.BatchAdmissionFailed(
                ((PrefillEndpoint.QueueBatchSlotAdmissionFailed) result).cause());
    }

    /** Acquire and transfer one exact QUEUE route-request reservation. */
    public static boolean commitRouteRequest(
            PrefillEndpoint endpoint,
            long requestId,
            long predictedMs,
            int maximumInflightRequests) {
        PrefillEndpoint.RequestCapacityReservationAcquisition acquisition =
                endpoint.acquireRequestCapacityReservation(
                        requestId, predictedMs, maximumInflightRequests);
        if (acquisition.status()
                != PrefillEndpoint.RequestCapacityReservationStatus.ACQUIRED) {
            return false;
        }
        PrefillEndpoint.RequestCapacityReservation reservation =
                acquisition.reservation();
        if (!reservation.prepareForDelivery()) {
            reservation.release();
            return false;
        }
        reservation.completePreparedDeliveryTransfer();
        reservation.completeDeliveryHandoff();
        return true;
    }

    /**
     * Reserve capacity through the supplied production admission contract and
     * build the exact callback payload accepted by {@link PriorityScheduler}.
     */
    public static AdmittedDecisionGroup admit(
            DeliveryCapacityAdmission admission,
            List<BatchItem> items) {
        if (items == null || items.isEmpty()) {
            throw new IllegalArgumentException(
                    "test admitted group requires at least one request");
        }
        List<DeliveryCapacityAdmission.ItemCapacityReservation> reservations =
                new ArrayList<>(items.size());
        DeliveryCapacityAdmission.BatchCapacityReservation batchReservation = null;
        try {
            if (items.get(0).deliveryMode() == DeliveryMode.BATCH_ENQUEUE) {
                DeliveryCapacityAdmission.BatchCapacityResult batchResult =
                        admission.tryReserveBatchCapacity(items.get(0));
                if (!(batchResult
                        instanceof DeliveryCapacityAdmission.BatchCapacityReserved reserved)) {
                    throw new IllegalStateException(
                            "test batch was not capacity-admitted: " + batchResult);
                }
                batchReservation = reserved.reservation();
            }
            for (BatchItem item : items) {
                DeliveryCapacityAdmission.AdmissionResult result =
                        admission.tryReserveItemCapacity(item);
                if (!(result instanceof DeliveryCapacityAdmission.CapacityReserved reserved)) {
                    throw new IllegalStateException(
                            "test request was not capacity-admitted: " + result);
                }
                reservations.add(reserved.reservation());
            }
            AdmittedDecisionGroup admitted = AdmittedDecisionGroup.create(
                    items, reservations, batchReservation);
            if (batchReservation != null) {
                DeliveryCapacityAdmission.BatchLoadPublicationResult publicationResult =
                        admitted.establishBatchLoadPublication();
                if (publicationResult
                        instanceof DeliveryCapacityAdmission.BatchLoadPublicationFailed failed) {
                    throw new IllegalStateException(
                            "test batch load publication failed", failed.cause());
                }
                try (DeliveryCapacityAdmission.BatchLoadPublication ignored =
                             ((DeliveryCapacityAdmission.BatchLoadPublicationEstablished)
                                     publicationResult).publication()) {
                    // Direct handler tests start at callback ownership.
                }
            }
            return admitted;
        } catch (RuntimeException | Error failure) {
            for (DeliveryCapacityAdmission.ItemCapacityReservation reservation : reservations) {
                try {
                    reservation.release();
                } catch (Throwable cleanupFailure) {
                    failure.addSuppressed(cleanupFailure);
                }
            }
            if (batchReservation != null) {
                try {
                    batchReservation.release();
                } catch (Throwable cleanupFailure) {
                    failure.addSuppressed(cleanupFailure);
                }
            }
            throw failure;
        }
    }

    /**
     * Invoke a handler through the synchronous ownership boundary used by
     * {@link BatcherContext}. Tests which call a scheduler handler directly
     * must finalize unresolved request tokens and the group handoff permit.
     */
    public static void runDeliveryCallback(
            DecisionGroupHandler handler,
            AdmittedDecisionGroup group,
            DecisionGroupMetadata metadata) {
        Throwable callbackFailure = null;
        try {
            handler.onDecisionGroupAdmitted(group, metadata);
        } catch (Throwable failure) {
            callbackFailure = failure;
        } finally {
            Throwable missingBatchLifecycle = callbackFailure != null
                    ? callbackFailure
                    : new IllegalStateException(
                            "test callback did not establish a batch lifecycle");
            AdmittedDecisionGroup.BatchCapacityCleanup batchCleanup =
                    group.terminateUntransferredBatchCapacity(
                            missingBatchLifecycle);
            try {
                group.completeTransferredBatchHandoff();
            } catch (Throwable handoffFailure) {
                if (callbackFailure == null) {
                    callbackFailure = handoffFailure;
                } else if (callbackFailure != handoffFailure) {
                    callbackFailure.addSuppressed(handoffFailure);
                }
            }

            Throwable unresolvedDefault = callbackFailure != null
                    ? callbackFailure
                    : batchCleanup.untransferred()
                            ? batchCleanup.failure()
                            : new IllegalStateException(
                                    "test callback left admitted request unresolved");
            Map<BatchItem, Throwable> unresolved = new LinkedHashMap<>();
            for (AdmittedDecisionGroup.AdmittedItem member : group.members()) {
                Throwable failure = member.terminateIfUnresolved(
                        unresolvedDefault);
                if (failure == null && batchCleanup.untransferred()) {
                    failure = batchCleanup.failure();
                }
                if (failure != null) {
                    unresolved.put(member.request(), failure);
                }
            }
            for (Map.Entry<BatchItem, Throwable> failure : unresolved.entrySet()) {
                try {
                    handler.onDeliveryFailure(
                            failure.getKey(), failure.getValue());
                } catch (Throwable ignored) {
                    // Ownership is finalized; preserve the callback failure.
                }
            }
        }
        if (callbackFailure instanceof RuntimeException runtimeFailure) {
            throw runtimeFailure;
        }
        if (callbackFailure instanceof Error error) {
            throw error;
        }
        if (callbackFailure != null) {
            throw new IllegalStateException(
                    "test decision-group callback failed", callbackFailure);
        }
    }

    /** Resolve every member as a successful callback would. */
    public static void complete(AdmittedDecisionGroup group) {
        for (AdmittedDecisionGroup.AdmittedItem member : group.members()) {
            if (!member.transferCapacityToEndpointLifecycle()) {
                throw new IllegalStateException(
                        "test capacity ownership was lost for request "
                                + member.request().requestId());
            }
        }
        if (group.deliveryMode() == DeliveryMode.BATCH_ENQUEUE) {
            group.transferBatchCapacityToLifecycle(
                    TEST_BATCH_IDS.incrementAndGet(), 0L, group.requests());
        }
        for (AdmittedDecisionGroup.AdmittedItem member : group.members()) {
            if (!member.completeDeliveryHandoff()) {
                throw new IllegalStateException(
                        "test callback did not resolve request "
                                + member.request().requestId());
            }
        }
    }

    private static final class TestCapacityReservation
            implements DeliveryCapacityAdmission.ItemCapacityReservation {

        private enum State {
            RESERVED,
            ENDPOINT_LIFECYCLE_OWNED,
            RELEASED
        }

        private final BatchItem item;
        private State state = State.RESERVED;

        private TestCapacityReservation(BatchItem item) {
            this.item = item;
        }

        @Override
        public BatchItem item() {
            return item;
        }

        @Override
        public synchronized boolean transferToEndpointLifecycle() {
            if (state == State.ENDPOINT_LIFECYCLE_OWNED) {
                return true;
            }
            if (state != State.RESERVED) {
                return false;
            }
            state = State.ENDPOINT_LIFECYCLE_OWNED;
            return true;
        }

        @Override
        public synchronized void release() {
            if (state == State.RESERVED) {
                state = State.RELEASED;
            }
        }
    }

    private static final class TestBatchCapacityReservation
            implements DeliveryCapacityAdmission.BatchCapacityReservation {

        private enum State {
            RESERVED,
            REGISTERED,
            RELEASED
        }

        private final BatchItem head;
        private State state = State.RESERVED;

        private TestBatchCapacityReservation(BatchItem head) {
            this.head = head;
        }

        @Override
        public BatchItem head() {
            return head;
        }

        @Override
        public DeliveryCapacityAdmission.BatchLoadPublicationResult
                establishBatchLoadPublication(
                List<BatchItem> requests) {
            return new DeliveryCapacityAdmission.BatchLoadPublicationEstablished(
                    () -> { });
        }

        @Override
        public synchronized BatchDispatcher.SubmissionPermit transferToBatchLifecycle(
                long batchId, long predictedMs, List<BatchItem> requests) {
            if (state != State.RESERVED) {
                throw new IllegalStateException(
                        "test batch reservation is not available for registration");
            }
            if (requests.isEmpty()) {
                throw new IllegalArgumentException(
                        "test batch lifecycle requires admitted requests");
            }
            state = State.REGISTERED;
            return noOpSubmissionPermit();
        }

        @Override
        public synchronized void completeDeliveryHandoff() {
            if (state != State.REGISTERED) {
                throw new IllegalStateException(
                        "test batch lifecycle was not registered before handoff completion");
            }
        }

        @Override
        public synchronized void release() {
            if (state == State.RESERVED) {
                state = State.RELEASED;
            }
        }
    }

    private static final class EndpointBackedTestBatchCapacityReservation
            implements DeliveryCapacityAdmission.BatchCapacityReservation {

        private final BatchItem head;
        private final PrefillEndpoint.QueueBatchSlotReservation endpointSlot;

        private EndpointBackedTestBatchCapacityReservation(
                BatchItem head,
                PrefillEndpoint.QueueBatchSlotReservation endpointSlot) {
            this.head = head;
            this.endpointSlot = endpointSlot;
        }

        @Override
        public BatchItem head() {
            return head;
        }

        @Override
        public DeliveryCapacityAdmission.BatchLoadPublicationResult
                establishBatchLoadPublication(
                List<BatchItem> requests) {
            try {
                return new DeliveryCapacityAdmission.BatchLoadPublicationEstablished(
                        endpointSlot.beginBatchLoadPublication(requests));
            } catch (Throwable publicationFailure) {
                return new DeliveryCapacityAdmission.BatchLoadPublicationFailed(
                        publicationFailure);
            }
        }

        @Override
        public BatchDispatcher.SubmissionPermit transferToBatchLifecycle(
                long batchId, long predictedMs, List<BatchItem> requests) {
            endpointSlot.transferToBatchLifecycle(batchId, predictedMs, requests);
            return noOpSubmissionPermit();
        }

        @Override
        public void completeDeliveryHandoff() {
            endpointSlot.completeDeliveryHandoff();
        }

        @Override
        public void release() {
            endpointSlot.release();
        }
    }

    private static BatchDispatcher.SubmissionPermit noOpSubmissionPermit() {
        return new BatchDispatcher.SubmissionPermit() {
            @Override
            public void submit(List<BatchItem> items,
                               PrefillEndpoint prefillEndpoint,
                               long batchId,
                               long predictedMs,
                               String reason,
                               DispatchCallback callback) {
            }

            @Override
            public void release() {
            }
        };
    }
}
