package org.flexlb.balance.scheduler;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Objects;

/**
 * A homogeneous logical decision group whose members already own every hard
 * capacity reservation required to enter delivery.
 *
 * <p>The payload is the ownership boundary between {@link BatcherContext} and
 * {@link DecisionGroupHandler}. A handler must claim and resolve each live
 * member exactly once. After the callback returns, the context terminates every
 * member whose ownership token remains unresolved; the callback is never
 * retried.
 */
public final class AdmittedDecisionGroup {

    private final List<AdmittedItem> members;
    private final List<BatchItem> requests;
    private final DeliveryMode deliveryMode;
    private final DeliveryCapacityAdmission.BatchCapacityReservation batchReservation;
    private BatchCapacityState batchCapacityState;

    private enum BatchCapacityState {
        NOT_REQUIRED,
        CALLBACK_OWNED,
        BATCH_LIFECYCLE_OWNED,
        DELIVERY_HANDOFF_COMPLETE,
        RELEASED
    }

    static AdmittedDecisionGroup create(
            List<BatchItem> requests,
            List<DeliveryCapacityAdmission.ItemCapacityReservation> reservations,
            DeliveryCapacityAdmission.BatchCapacityReservation batchReservation) {
        Objects.requireNonNull(requests, "requests");
        Objects.requireNonNull(reservations, "reservations");
        if (requests.isEmpty() || requests.size() != reservations.size()) {
            throw new IllegalArgumentException(
                    "admitted group requires one reservation per request");
        }

        DeliveryMode mode = requests.get(0).deliveryMode();
        if ((mode == DeliveryMode.BATCH_ENQUEUE) != (batchReservation != null)) {
            throw new IllegalArgumentException(
                    "BATCH_ENQUEUE requires exactly one group batch-capacity reservation");
        }
        List<AdmittedItem> admitted = new ArrayList<>(requests.size());
        for (int index = 0; index < requests.size(); index++) {
            BatchItem request = Objects.requireNonNull(
                    requests.get(index), "admitted request");
            if (request.deliveryMode() != mode) {
                throw new IllegalArgumentException(
                        "admitted decision group must use one delivery mode");
            }
            DeliveryCapacityAdmission.ItemCapacityReservation reservation =
                    Objects.requireNonNull(reservations.get(index),
                            "capacity reservation");
            // BatcherContext validates reservation.item() exactly once while
            // acquiring the token. Do not call an external token accessor
            // again while constructing the ownership payload.
            admitted.add(new AdmittedItem(request, reservation));
        }
        return new AdmittedDecisionGroup(
                requests, admitted, mode, batchReservation);
    }

    private AdmittedDecisionGroup(
            List<BatchItem> requests,
            List<AdmittedItem> members,
            DeliveryMode deliveryMode,
            DeliveryCapacityAdmission.BatchCapacityReservation batchReservation) {
        this.requests = Collections.unmodifiableList(new ArrayList<>(requests));
        this.members = Collections.unmodifiableList(members);
        this.deliveryMode = deliveryMode;
        this.batchReservation = batchReservation;
        this.batchCapacityState = batchReservation == null
                ? BatchCapacityState.NOT_REQUIRED
                : BatchCapacityState.CALLBACK_OWNED;
    }

    public List<AdmittedItem> members() {
        return members;
    }

    public List<BatchItem> requests() {
        return requests;
    }

    DeliveryMode deliveryMode() {
        return deliveryMode;
    }

    /**
     * Establish conservative callback load as a typed admission-finalization
     * result. A custom reservation cannot escape an exception back into the
     * worker retry loop.
     */
    DeliveryCapacityAdmission.BatchLoadPublicationResult
            establishBatchLoadPublication() {
        if (batchReservation == null) {
            return new DeliveryCapacityAdmission.BatchLoadPublicationFailed(
                    new IllegalStateException(
                            "route decisions have no batch load ownership"));
        }
        try {
            DeliveryCapacityAdmission.BatchLoadPublicationResult result =
                    batchReservation.establishBatchLoadPublication(requests);
            return result != null
                    ? result
                    : new DeliveryCapacityAdmission.BatchLoadPublicationFailed(
                            new IllegalStateException(
                                    "batch load publication returned no result"));
        } catch (Throwable publicationFailure) {
            return new DeliveryCapacityAdmission.BatchLoadPublicationFailed(
                    publicationFailure);
        }
    }

    /**
     * Transfer the already-reserved group slot to one exact Prefill batch
     * lifecycle. No capacity check occurs at this boundary.
     */
    public synchronized BatchDispatcher.SubmissionPermit
            transferBatchCapacityToLifecycle(
            long batchId,
            long predictedMs,
            List<BatchItem> batchRequests) {
        if (batchCapacityState != BatchCapacityState.CALLBACK_OWNED) {
            throw new IllegalStateException(
                    "batch capacity is not owned by this callback");
        }
        BatchDispatcher.SubmissionPermit submissionPermit =
                batchReservation.transferToBatchLifecycle(
                batchId, predictedMs, List.copyOf(batchRequests));
        batchCapacityState = BatchCapacityState.BATCH_LIFECYCLE_OWNED;
        return submissionPermit;
    }

    /** Release the group slot if the callback never established a batch lifecycle. */
    BatchCapacityCleanup terminateUntransferredBatchCapacity(Throwable failure) {
        Objects.requireNonNull(failure, "failure");
        synchronized (this) {
            if (batchCapacityState != BatchCapacityState.CALLBACK_OWNED) {
                return BatchCapacityCleanup.NOT_REQUIRED;
            }
            batchCapacityState = BatchCapacityState.RELEASED;
        }
        Throwable terminalFailure = failure;
        try {
            batchReservation.release();
        } catch (Throwable releaseFailure) {
            if (releaseFailure != terminalFailure) {
                terminalFailure.addSuppressed(releaseFailure);
            }
        }
        return new BatchCapacityCleanup(true, terminalFailure);
    }

    /** Release the endpoint-generation handoff permit after callback completion. */
    void completeTransferredBatchHandoff() {
        synchronized (this) {
            if (batchCapacityState != BatchCapacityState.BATCH_LIFECYCLE_OWNED) {
                return;
            }
            batchReservation.completeDeliveryHandoff();
            batchCapacityState = BatchCapacityState.DELIVERY_HANDOFF_COMPLETE;
        }
    }

    record BatchCapacityCleanup(boolean untransferred, Throwable failure) {
        private static final BatchCapacityCleanup NOT_REQUIRED =
                new BatchCapacityCleanup(false, null);
    }

    /** Capacity ownership for one exact request generation. */
    public static final class AdmittedItem {

        private enum State {
            CALLBACK_OWNED,
            ENDPOINT_LIFECYCLE_OWNED,
            DELIVERY_HANDOFF_COMPLETE,
            EXISTING_REQUEST_OWNER,
            TERMINATED
        }

        private final BatchItem request;
        private final DeliveryCapacityAdmission.ItemCapacityReservation reservation;
        private State state = State.CALLBACK_OWNED;
        private Throwable unresolvedFailure;

        private AdmittedItem(
                BatchItem request,
                DeliveryCapacityAdmission.ItemCapacityReservation reservation) {
            this.request = request;
            this.reservation = reservation;
        }

        public BatchItem request() {
            return request;
        }

        /** Transfer the already-reserved capacity to endpoint lifecycle ownership. */
        public synchronized boolean transferCapacityToEndpointLifecycle() {
            if (state != State.CALLBACK_OWNED) {
                throw new IllegalStateException(
                        "admitted request capacity is not available for transfer request_id="
                                + request.requestId());
            }
            if (!reservation.transferToEndpointLifecycle()) {
                return false;
            }
            state = State.ENDPOINT_LIFECYCLE_OWNED;
            return true;
        }

        /**
         * Preserve an item-scoped failure before delivery handoff without
         * aborting siblings which own independent reservations.
         * The context releases and terminates this unresolved member after the
         * callback returns.
         */
        public synchronized void recordFailureBeforeDeliveryHandoff(Throwable failure) {
            Objects.requireNonNull(failure, "failure");
            if (state != State.CALLBACK_OWNED
                    && state != State.ENDPOINT_LIFECYCLE_OWNED) {
                throw new IllegalStateException(
                    "unresolved delivery failure cannot be recorded after handoff"
                                + " request_id=" + request.requestId());
            }
            if (unresolvedFailure == null) {
                unresolvedFailure = failure;
            } else if (unresolvedFailure != failure) {
                unresolvedFailure.addSuppressed(failure);
            }
        }

        /**
         * Resolve callback ownership after endpoint lifecycle ownership has
         * been established. This closes generation-handoff protection; it does
         * not recheck or release hard capacity.
         */
        public synchronized boolean completeDeliveryHandoff() {
            if (state != State.ENDPOINT_LIFECYCLE_OWNED) {
                return false;
            }
            reservation.completeDeliveryHandoff();
            state = State.DELIVERY_HANDOFF_COMPLETE;
            return true;
        }

        /**
         * Resolve a reservation when cancellation, expiration, shutdown, or a
         * prior terminal reducer already owns the request. This path performs
         * capacity cleanup but deliberately does not invoke a second terminal
         * callback.
         *
         * @return a cleanup failure for logging, otherwise {@code null}
         */
        public Throwable settleWithExistingRequestOwner() {
            boolean endpointLifecycleOwned;
            synchronized (this) {
                if (state == State.EXISTING_REQUEST_OWNER
                        || state == State.DELIVERY_HANDOFF_COMPLETE
                        || state == State.TERMINATED) {
                    return null;
                }
                endpointLifecycleOwned = state == State.ENDPOINT_LIFECYCLE_OWNED;
                state = State.EXISTING_REQUEST_OWNER;
            }
            Throwable firstFailure = null;
            if (endpointLifecycleOwned) {
                try {
                    reservation.completeDeliveryHandoff();
                } catch (Throwable handoffFailure) {
                    firstFailure = handoffFailure;
                }
            }
            try {
                reservation.release();
            } catch (Throwable releaseFailure) {
                if (firstFailure == null) {
                    firstFailure = releaseFailure;
                } else if (firstFailure != releaseFailure) {
                    firstFailure.addSuppressed(releaseFailure);
                }
            }
            return firstFailure;
        }

        /** Resolve and release any member left behind by the callback. */
        Throwable terminateIfUnresolved(Throwable failure) {
            Objects.requireNonNull(failure, "failure");
            Throwable terminalFailure;
            boolean endpointLifecycleOwned;
            synchronized (this) {
                if (state == State.DELIVERY_HANDOFF_COMPLETE
                        || state == State.EXISTING_REQUEST_OWNER
                        || state == State.TERMINATED) {
                    return null;
                }
                endpointLifecycleOwned = state == State.ENDPOINT_LIFECYCLE_OWNED;
                state = State.TERMINATED;
                terminalFailure = unresolvedFailure != null
                        ? unresolvedFailure : failure;
                if (unresolvedFailure != null
                        && failure != unresolvedFailure) {
                    unresolvedFailure.addSuppressed(failure);
                }
            }
            if (endpointLifecycleOwned) {
                try {
                    reservation.completeDeliveryHandoff();
                } catch (Throwable handoffFailure) {
                    if (handoffFailure != terminalFailure) {
                        terminalFailure.addSuppressed(handoffFailure);
                    }
                }
            }
            try {
                reservation.release();
            } catch (Throwable releaseFailure) {
                if (releaseFailure != terminalFailure) {
                    terminalFailure.addSuppressed(releaseFailure);
                }
            }
            return terminalFailure;
        }
    }
}
