package org.flexlb.balance.scheduler;

import java.util.List;
import java.util.Objects;

/**
 * Reserves every hard capacity unit required before a candidate request can
 * become part of a final decision and enter the delivery callback.
 *
 * <p>Grouping proposes an ordered candidate list. This contract decides how
 * much of its prefix can cross from the active queue into callback ownership.
 * A capacity rejection never removes the rejecting request and never invokes
 * {@link DecisionGroupHandler#onDecisionGroupAdmitted} for it.
 */
public interface DeliveryCapacityAdmission {

    /** Try to reserve all endpoint capacity required by {@code item}. */
    AdmissionResult tryReserveItemCapacity(BatchItem item);

    /**
     * Reserve the one group-scoped Prefill batch slot required by a BATCH
     * candidate. Implementations that do not support batch delivery fail
     * closed; route-only admission remains a functional-interface use case.
     */
    default BatchCapacityResult tryReserveBatchCapacity(BatchItem head) {
        return new BatchAdmissionFailed(new IllegalStateException(
                "batch capacity admission is not configured"));
    }

    /** Result of reserving one group-scoped QUEUE batch slot. */
    sealed interface BatchCapacityResult permits BatchCapacityReserved,
            BatchCapacityUnavailable, BatchOwnershipLost, BatchAdmissionFailed {
    }

    record BatchCapacityReserved(BatchCapacityReservation reservation)
            implements BatchCapacityResult {
        public BatchCapacityReserved {
            Objects.requireNonNull(reservation, "reservation");
        }
    }

    record BatchCapacityUnavailable(
            CapacityResource resource,
            CapacityAvailability availability)
            implements BatchCapacityResult {
        public BatchCapacityUnavailable {
            Objects.requireNonNull(resource, "resource");
            Objects.requireNonNull(availability, "availability");
        }
    }

    enum BatchOwnershipLost implements BatchCapacityResult {
        INSTANCE
    }

    record BatchAdmissionFailed(Throwable cause) implements BatchCapacityResult {
        public BatchAdmissionFailed {
            Objects.requireNonNull(cause, "cause");
        }
    }

    /** Result of one non-blocking capacity reservation attempt. */
    sealed interface AdmissionResult permits CapacityReserved,
            CapacityUnavailable, OwnershipLost, AdmissionFailed {
    }

    /** Every hard capacity unit is reserved for this exact request generation. */
    record CapacityReserved(ItemCapacityReservation reservation)
            implements AdmissionResult {
        public CapacityReserved {
            Objects.requireNonNull(reservation, "reservation");
        }
    }

    /** The request remains active and waits in its original queue order. */
    record CapacityUnavailable(
            CapacityResource resource,
            CapacityAvailability availability)
            implements AdmissionResult {
        public CapacityUnavailable {
            Objects.requireNonNull(resource, "resource");
            Objects.requireNonNull(availability, "availability");
        }
    }

    /**
     * Non-blocking live wait predicate for the exact resource that rejected
     * admission. It is evaluated while the worker queue lock is held and must
     * neither acquire endpoint mutation locks nor invoke callbacks.
     */
    interface CapacityAvailability {
        boolean isAvailable();

        /** Subscribe to the exact transition which may make this resource available. */
        default void addListener(Runnable listener) {
        }

        /** Remove a listener previously installed by the blocked worker. */
        default void removeListener(Runnable listener) {
        }
    }

    /** Cancellation, expiration, preemption, or shutdown already owns the request. */
    enum OwnershipLost implements AdmissionResult {
        INSTANCE
    }

    /** Capacity preparation failed for a reason that must terminate the request. */
    record AdmissionFailed(Throwable cause) implements AdmissionResult {
        public AdmissionFailed {
            Objects.requireNonNull(cause, "cause");
        }
    }

    enum CapacityResource {
        PREFILL_BATCH,
        PREFILL_REQUEST,
        DECODE_ENGINE,
        BATCH_DISPATCHER
    }

    /**
     * Request-generation-scoped capacity ownership.
     *
     * <p>{@link #transferToEndpointLifecycle()} transfers the reserved units to
     * the normal endpoint
     * lifecycle before the delivery subsystem accepts ownership.
     * {@link #release()} abandons an untransferred reservation. Both operations
     * are idempotent; neither performs another capacity check.
     */
    interface ItemCapacityReservation {

        BatchItem item();

        /**
         * Transfer the reservation to endpoint lifecycle ownership.
         *
         * @return {@code true} when this exact request generation still owns
         *         every reserved unit, otherwise {@code false}
         */
        boolean transferToEndpointLifecycle();

        /**
         * Release endpoint-generation protection after the synchronous
         * callback has handed ownership to delivery or completed local cleanup.
         */
        default void completeDeliveryHandoff() {
        }

        /** Release only this exact untransferred reservation. */
        void release();
    }

    /**
     * Exact group-scoped ownership of one QUEUE EnqueueBatch slot.
     * Registration transfers the endpoint slot to Prefill batch lifecycle and
     * returns the already-accepted local dispatcher task. Release abandons both
     * resources before that transfer.
     */
    interface BatchCapacityReservation {

        BatchItem head();

        /**
         * Establish conservative callback-owned load before ACTIVE removal.
         * Failure is typed because retrying the same ACTIVE head cannot repair
         * a publication invariant failure. A failed result owns no publication
         * state and leaves the reservation releasable.
         */
        BatchLoadPublicationResult establishBatchLoadPublication(
                List<BatchItem> requests);

        BatchDispatcher.SubmissionPermit transferToBatchLifecycle(
                long batchId, long predictedMs, List<BatchItem> requests);

        /** Mark callback-to-delivery ownership handoff or local cleanup complete. */
        void completeDeliveryHandoff();

        void release();
    }

    sealed interface BatchLoadPublicationResult
            permits BatchLoadPublicationEstablished,
            BatchLoadPublicationFailed {
    }

    record BatchLoadPublicationEstablished(BatchLoadPublication publication)
            implements BatchLoadPublicationResult {
        public BatchLoadPublicationEstablished {
            Objects.requireNonNull(publication, "publication");
        }
    }

    record BatchLoadPublicationFailed(Throwable cause)
            implements BatchLoadPublicationResult {
        public BatchLoadPublicationFailed {
            Objects.requireNonNull(cause, "cause");
        }
    }

    /**
     * Keeps endpoint load snapshots conservative from ACTIVE removal until
     * the callback either publishes the real batch lifecycle or abandons it.
     */
    @FunctionalInterface
    interface BatchLoadPublication extends AutoCloseable {
        @Override
        void close();
    }

}
