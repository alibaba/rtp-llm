package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.delivery.DeliveryItem;
import org.flexlb.balance.delivery.PrefillAdmissionPort;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillWorkLedger;
import org.flexlb.balance.projection.RouteProjection;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.OptionalLong;
import java.util.function.LongSupplier;

import static org.flexlb.balance.scheduler.PrefillAdmissionResources.PreparedState.CLOSED;
import static org.flexlb.balance.scheduler.PrefillAdmissionResources.PreparedState.COMMITTED;
import static org.flexlb.balance.scheduler.PrefillAdmissionResources.PreparedState.PREPARED;
import static org.flexlb.balance.scheduler.PrefillAdmissionResources.accepted;
import static org.flexlb.balance.scheduler.PrefillAdmissionResources.failed;
import static org.flexlb.balance.scheduler.PrefillAdmissionResources.missingEndpoint;
import static org.flexlb.balance.scheduler.PrefillAdmissionResources.prepareCommitted;
import static org.flexlb.balance.scheduler.PrefillAdmissionResources.prepareMember;
import static org.flexlb.balance.scheduler.PrefillAdmissionResources.preserveRejectedCause;
import static org.flexlb.balance.scheduler.PrefillAdmissionResources.rejected;
import static org.flexlb.balance.scheduler.PrefillAdmissionResources.rejectedPrefill;
import static org.flexlb.balance.scheduler.PrefillAdmissionResources.restorePreparedTail;
import static org.flexlb.balance.scheduler.PrefillAdmissionResources.rollbackMember;
import static org.flexlb.balance.scheduler.PrefillAdmissionResources.rollbackReservation;
import static org.flexlb.balance.scheduler.PrefillAdmissionResources.sameIdentitySequence;
import static org.flexlb.balance.scheduler.PrefillAdmissionResources.throwRollbackFailure;

/**
 * Batch dispatcher admission.
 *
 * <p>The transport correlation is allocated exactly once when the head is
 * prepared. All Prefill and Decode leaves remain hidden behind the returned
 * transaction capability.</p>
 */
public final class BatchPrefillAdmission implements PrefillAdmissionPort {

    private static final RouteProjection.AdmissionBlockSemantics
            CAPACITY_BLOCK = new RouteProjection.AdmissionBlockSemantics(
                    "DELIVERY_CAPACITY_BATCH_ADMISSION",
                    RouteProjection.AfterProbeAdmission.BLOCKED,
                    "DELIVERY_CAPACITY_BATCH_ADMISSION");

    private final LongSupplier batchIds;

    public BatchPrefillAdmission(LongSupplier batchIds) {
        this.batchIds = Objects.requireNonNull(batchIds, "batchIds");
    }

    @Override
    public CapacityBoundary.Attempt<PreparedAdmission> prepare(
            DeliveryItem head,
            long predictedMs) {
        final BatchItem exactHead;
        final long batchId;
        try {
            exactHead = (BatchItem) head;
            batchId = batchIds.getAsLong();
            if (batchId <= 0L) {
                throw new IllegalStateException(
                        "batch id supplier returned a non-positive id");
            }
        } catch (RuntimeException | Error failure) {
            return failed(failure);
        }

        PrefillEndpoint prefill = exactHead.prefillEp();
        if (prefill == null) {
            return failed(missingEndpoint("Prefill", exactHead));
        }
        final PreparedTransaction transaction;
        final CapacityBoundary.Attempt<PreparedAdmission> acceptedTransaction;
        try {
            transaction = new PreparedTransaction(batchId, prefill);
            acceptedTransaction = accepted(transaction);
        } catch (Throwable failure) {
            return failed(failure);
        }
        PrefillWorkLedger.BatchReservationResult reservationResult;
        try {
            reservationResult = prefill.reserveBatch(
                    exactHead,
                    batchId,
                    exactHead.maxInflightDeliveriesPerPrefillWorker());
        } catch (RuntimeException | Error failure) {
            return failed(failure);
        }
        if (reservationResult.status()
                != PrefillWorkLedger.CapacityStatus.ACQUIRED) {
            return rejectedPrefill(
                    exactHead,
                    reservationResult.status(),
                    new CapacityBoundary.Unavailable(
                            prefill.batchAdmissionAvailability(
                                    exactHead.maxInflightDeliveriesPerPrefillWorker()),
                            CAPACITY_BLOCK));
        }

        PrefillWorkLedger.BatchReservation reservation =
                reservationResult.reservation();
        final CapacityBoundary.Attempt<PrefillAdmissionResources.Member>
                memberAttempt;
        try {
            memberAttempt = prepareMember(exactHead);
        } catch (Throwable failure) {
            return failed(rollbackReservation(reservation, failure));
        }
        if (memberAttempt
                instanceof CapacityBoundary.Attempt.Rejected<
                        PrefillAdmissionResources.Member> rejectedMember) {
            Throwable rollbackFailure = rollbackReservation(
                    reservation, null);
            if (rollbackFailure != null) {
                preserveRejectedCause(
                        rollbackFailure, rejectedMember.boundary());
                return failed(rollbackFailure);
            }
            return rejected(rejectedMember.boundary());
        }
        PrefillAdmissionResources.Member headMember =
                ((CapacityBoundary.Attempt.Accepted<
                        PrefillAdmissionResources.Member>) memberAttempt)
                        .value();
        try {
            transaction.bind(reservation, headMember);
        } catch (Throwable bindFailure) {
            Throwable failure = rollbackMember(headMember, bindFailure);
            failure = rollbackReservation(reservation, failure);
            return failed(failure);
        }
        return acceptedTransaction;
    }

    private static final class PreparedTransaction
            implements PreparedAdmission {
        private final long batchId;
        private final PrefillEndpoint prefill;
        private PrefillWorkLedger.BatchReservation reservation;
        private List<PrefillAdmissionResources.Member> members;
        private PrefillAdmissionResources.PreparedState state = PREPARED;

        private PreparedTransaction(
                long batchId,
                PrefillEndpoint prefill) {
            this.batchId = batchId;
            this.prefill = prefill;
            members = new ArrayList<>(1);
            members.add(null);
        }

        /** Bind only references into storage allocated before acquisition. */
        private void bind(
                PrefillWorkLedger.BatchReservation exactReservation,
                PrefillAdmissionResources.Member head) {
            reservation = exactReservation;
            members.set(0, head);
        }

        @Override
        public OptionalLong correlationId() {
            return OptionalLong.of(batchId);
        }

        @Override
        public synchronized CapacityBoundary.Attempt<DeliveryItem> append(
                DeliveryItem exactNextItem,
                long predictedMs) {
            requirePrepared("append");
            final BatchItem exact;
            try {
                exact = (BatchItem) exactNextItem;
            } catch (RuntimeException | Error failure) {
                return failed(failure);
            }

            final int originalSize = members.size();
            final CapacityBoundary.Attempt<DeliveryItem> acceptedItem;
            try {
                acceptedItem = accepted(exact);
                members.add(null);
            } catch (Throwable failure) {
                return failed(restorePreparedTail(
                        members, originalSize, failure));
            }

            final CapacityBoundary.Attempt<PrefillAdmissionResources.Member>
                    attempt;
            try {
                attempt = prepareMember(exact);
            } catch (Throwable failure) {
                return failed(restorePreparedTail(
                        members, originalSize, failure));
            }
            if (attempt
                    instanceof CapacityBoundary.Attempt.Rejected<
                            PrefillAdmissionResources.Member> rejectedMember) {
                Throwable rollbackFailure = restorePreparedTail(
                        members, originalSize, null);
                if (rollbackFailure != null) {
                    preserveRejectedCause(
                            rollbackFailure, rejectedMember.boundary());
                    return failed(rollbackFailure);
                }
                return rejected(rejectedMember.boundary());
            }
            PrefillAdmissionResources.Member member =
                    ((CapacityBoundary.Attempt.Accepted<
                            PrefillAdmissionResources.Member>) attempt).value();
            try {
                members.set(originalSize, member);
            } catch (Throwable bindFailure) {
                Throwable failure = restorePreparedTail(
                        members, originalSize, bindFailure);
                failure = rollbackMember(member, failure);
                return failed(failure);
            }
            return acceptedItem;
        }

        @Override
        public synchronized CommittedAdmission commitUnderLock(
                List<DeliveryItem> exactItems,
                long predictedMs) {
            requirePrepared("commit");
            if (!sameIdentitySequence(members, exactItems)) {
                throw new IllegalArgumentException(
                        "batch commit does not match prepared identities");
            }
            PrefillAdmissionResources.CommittedAdmissionAdapter committedOwner =
                    prepareCommitted(members, 1);
            PrefillWorkLedger.CommittedHandoff handoff =
                    reservation.commitUnderLock(exactItems, predictedMs);
            committedOwner.bind(handoff);
            state = COMMITTED;
            reservation = null;
            members = null;
            return committedOwner;
        }

        @Override
        public void close() {
            PrefillWorkLedger.Reservation exactReservation;
            List<PrefillAdmissionResources.Member> exactMembers;
            synchronized (this) {
                if (state != PREPARED) {
                    return;
                }
                state = CLOSED;
                exactReservation = reservation;
                reservation = null;
                exactMembers = members;
                members = List.of();
            }
            Throwable rollbackFailure = null;
            for (PrefillAdmissionResources.Member member : exactMembers) {
                rollbackFailure = rollbackMember(
                        member, rollbackFailure);
            }
            rollbackFailure = rollbackReservation(
                    exactReservation, rollbackFailure);
            if (rollbackFailure != null) {
                throwRollbackFailure(rollbackFailure);
            }
        }

        private void requirePrepared(String operation) {
            if (state != PREPARED) {
                throw new IllegalStateException(
                        operation
                                + " requires PREPARED batch admission");
            }
        }
    }
}
