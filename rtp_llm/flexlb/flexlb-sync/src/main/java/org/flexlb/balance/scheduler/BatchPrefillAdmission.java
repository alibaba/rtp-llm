package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.delivery.CapacityBoundary.Attempt;
import org.flexlb.balance.delivery.CapacityBoundary.Attempt.Accepted;
import org.flexlb.balance.delivery.CapacityBoundary.Attempt.Rejected;
import org.flexlb.balance.delivery.DeliveryItem;
import org.flexlb.balance.delivery.PrefillAdmissionPort;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillWorkLedger;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.balance.scheduler.PrefillAdmissionResources.CommittedAdmissionAdapter;
import org.flexlb.balance.scheduler.PrefillAdmissionResources.Member;
import org.flexlb.balance.scheduler.PrefillAdmissionResources.PreparedState;

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
    public Attempt<PreparedAdmission> prepare(
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
        try {
            return new PreparedTransaction(batchId, prefill)
                    .prepareHead(exactHead);
        } catch (Throwable failure) {
            return failed(failure);
        }
    }

    private static final class PreparedTransaction
            implements PreparedAdmission {
        private final long batchId;
        private final PrefillEndpoint prefill;
        private final Attempt<PreparedAdmission> acceptedTransaction;
        private PrefillWorkLedger.BatchReservation reservation;
        private ArrayList<Member> members;
        private Member pendingMember;
        private PreparedState state = PREPARED;

        private PreparedTransaction(
                long batchId,
                PrefillEndpoint prefill) {
            this.batchId = batchId;
            this.prefill = prefill;
            members = new ArrayList<>(1);
            acceptedTransaction = accepted(this);
        }

        private Attempt<PreparedAdmission> prepareHead(BatchItem head) {
            try {
                PrefillWorkLedger.BatchReservationResult result =
                        prefill.reserveBatch(
                                head,
                                batchId,
                                head.maxInflightDeliveriesPerPrefillWorker());
                if (result.status()
                        != PrefillWorkLedger.CapacityStatus.ACQUIRED) {
                    return rejectedPrefill(
                            head,
                            result.status(),
                            new CapacityBoundary.Unavailable(
                                    prefill.batchAdmissionAvailability(
                                            head.maxInflightDeliveriesPerPrefillWorker()),
                                    CAPACITY_BLOCK));
                }

                reservation = result.reservation();
                return switch (prepareAndAddMember(
                        head, acceptedTransaction)) {
                    case Accepted<PreparedAdmission> accepted -> accepted;
                    case Rejected<PreparedAdmission> rejected ->
                            abort(rejected.boundary());
                };
            } catch (Throwable failure) {
                return abort(failure);
            }
        }

        @Override
        public OptionalLong correlationId() {
            return OptionalLong.of(batchId);
        }

        @Override
        public synchronized Attempt<DeliveryItem> append(
                DeliveryItem exactNextItem,
                long predictedMs) {
            requirePrepared("append");
            final BatchItem exact;
            try {
                exact = (BatchItem) exactNextItem;
            } catch (RuntimeException | Error failure) {
                return failed(failure);
            }

            try {
                Attempt<DeliveryItem> acceptedItem = accepted(exact);
                members.ensureCapacity(Math.addExact(members.size(), 1));
                return prepareAndAddMember(exact, acceptedItem);
            } catch (Throwable failure) {
                Member exactPending = pendingMember;
                pendingMember = null;
                return failed(rollbackMember(exactPending, failure));
            }
        }

        /** Capture before the no-allocation list bind. */
        private <T> Attempt<T> prepareAndAddMember(
                BatchItem item,
                Attempt<T> acceptedResult) {
            return switch (prepareMember(item)) {
                case Accepted<Member>(var member) -> {
                    pendingMember = member;
                    members.add(pendingMember);
                    pendingMember = null;
                    yield acceptedResult;
                }
                case Rejected<Member>(var boundary) -> rejected(boundary);
            };
        }

        private <T> Attempt<T> abort(CapacityBoundary boundary) {
            Throwable rollbackFailure = rollbackAll(null);
            if (rollbackFailure == null) {
                return rejected(boundary);
            }
            preserveRejectedCause(rollbackFailure, boundary);
            return failed(rollbackFailure);
        }

        private <T> Attempt<T> abort(Throwable failure) {
            return failed(rollbackAll(failure));
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
            CommittedAdmissionAdapter committedOwner =
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
            Throwable rollbackFailure = rollbackAll(null);
            if (rollbackFailure != null) {
                throwRollbackFailure(rollbackFailure);
            }
        }

        /** Detach once, then attempt every rollback leaf. */
        private Throwable rollbackAll(Throwable priorFailure) {
            PrefillWorkLedger.Reservation exactReservation;
            ArrayList<Member> exactMembers;
            Member exactPending;
            synchronized (this) {
                if (state != PREPARED) {
                    return priorFailure;
                }
                state = CLOSED;
                exactReservation = reservation;
                reservation = null;
                exactMembers = members;
                members = null;
                exactPending = pendingMember;
                pendingMember = null;
            }

            Throwable failure = rollbackMember(
                    exactPending, priorFailure);
            for (Member member : exactMembers) {
                failure = rollbackMember(member, failure);
            }
            return rollbackReservation(exactReservation, failure);
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
