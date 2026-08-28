package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.delivery.CapacityBoundary.Attempt;
import org.flexlb.balance.scheduler.ScheduledRequest;
import org.flexlb.balance.delivery.PrefillAdmissionPort;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.balance.scheduler.PrefillAdmissionResources.CommittedAdmissionOwner;
import org.flexlb.balance.scheduler.PrefillAdmissionResources.Member;
import org.flexlb.dao.route.RoleType;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.OptionalLong;
import java.util.function.LongSupplier;

import static org.flexlb.balance.scheduler.PrefillAdmissionResources.accepted;
import static org.flexlb.balance.scheduler.PrefillAdmissionResources.createCommittedOwner;
import static org.flexlb.balance.scheduler.PrefillAdmissionResources.failed;
import static org.flexlb.balance.scheduler.PrefillAdmissionResources.missingEndpoint;
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
                    "DELIVERY_CAPACITY_BATCH_ADMISSION",
                    RoleType.PREFILL);

    private final LongSupplier batchIds;

    public BatchPrefillAdmission(LongSupplier batchIds) {
        this.batchIds = Objects.requireNonNull(batchIds, "batchIds");
    }

    @Override
    public Attempt<PreparedAdmission> tryBegin(
            ScheduledRequest firstCandidate) {
        final ScheduledRequest exact;
        final long batchId;
        try {
            exact = firstCandidate;
            batchId = batchIds.getAsLong();
            if (batchId <= 0L) {
                throw new IllegalStateException(
                        "batch id supplier returned a non-positive id");
            }
        } catch (RuntimeException | Error failure) {
            return failed(failure);
        }

        PrefillEndpoint prefill = exact.prefillEp();
        if (prefill == null) {
            return failed(missingEndpoint("Prefill", exact));
        }
        try {
            return accepted(new PreparedTransaction(batchId, prefill));
        } catch (Throwable failure) {
            return failed(failure);
        }
    }

    private static final class PreparedTransaction
            implements PreparedAdmission {
        private final long batchId;
        private final PrefillEndpoint prefill;
        private PrefillState.BatchReservation reservation;
        private ArrayList<Member> members;
        private Member pendingMember;
        private boolean ownershipResolved;

        private PreparedTransaction(
                long batchId,
                PrefillEndpoint prefill) {
            this.batchId = batchId;
            this.prefill = prefill;
            members = new ArrayList<>(1);
        }

        @Override
        public OptionalLong correlationId() {
            return OptionalLong.of(batchId);
        }

        @Override
        public synchronized Attempt<ScheduledRequest> tryAppend(
                ScheduledRequest exactNextItem,
                long predictedMs) {
            requirePrepared("append");
            final ScheduledRequest exact;
            try {
                exact = exactNextItem;
            } catch (RuntimeException | Error failure) {
                return failed(failure);
            }

            try {
                Attempt<ScheduledRequest> acceptedItem = accepted(exact);
                members.ensureCapacity(Math.addExact(members.size(), 1));
                boolean first = members.isEmpty();
                if (first) {
                    PrefillState.BatchReservationResult result =
                            prefill.reserveBatch(
                                    exact,
                                    batchId,
                                    exact.maxInflightDeliveriesPerPrefillWorker());
                    if (result.status()
                            != PrefillState.CapacityStatus.ACQUIRED) {
                        return rejectedPrefill(
                                exact,
                                result.status(),
                                CapacityBoundary.unavailable(
                                        prefill.batchAdmissionAvailability(
                                                exact.maxInflightDeliveriesPerPrefillWorker()),
                                        CAPACITY_BLOCK));
                    }
                    reservation = result.reservation();
                }
                Attempt<ScheduledRequest> result =
                        prepareAndAddMember(exact, acceptedItem);
                if (first && !result.accepted()) {
                    return abort(result.boundary());
                }
                return result;
            } catch (Throwable failure) {
                if (members.isEmpty()) {
                    return abort(failure);
                }
                Member exactPending = pendingMember;
                pendingMember = null;
                return failed(rollbackMember(exactPending, failure));
            }
        }

        /** Capture before the no-allocation list bind. */
        private <T> Attempt<T> prepareAndAddMember(
                ScheduledRequest item,
                Attempt<T> acceptedResult) {
            Attempt<Member> attempt = prepareMember(item);
            if (!attempt.accepted()) {
                return rejected(attempt.boundary());
            }
            pendingMember = attempt.value();
            members.add(pendingMember);
            pendingMember = null;
            return acceptedResult;
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
        public synchronized CommittedAdmission commitPreparedUnderLock(
                List<ScheduledRequest> exactItems,
                long predictedMs) {
            requirePrepared("commit");
            if (!sameIdentitySequence(members, exactItems)) {
                throw new IllegalArgumentException(
                        "batch commit does not match prepared identities");
            }
            if (reservation == null || members.isEmpty()) {
                throw new IllegalStateException(
                        "batch commit requires at least one prepared member");
            }
            CommittedAdmissionOwner committedOwner =
                    createCommittedOwner(members, 1);
            PrefillState.CommittedHandoff handoff =
                    reservation.commit(exactItems, predictedMs);
            committedOwner.bindBatchHandoff(handoff);
            ownershipResolved = true;
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
            PrefillState.Reservation exactReservation;
            ArrayList<Member> exactMembers;
            Member exactPending;
            synchronized (this) {
                if (ownershipResolved) {
                    return priorFailure;
                }
                ownershipResolved = true;
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
            if (ownershipResolved) {
                throw new IllegalStateException(
                        operation
                                + " requires PREPARED batch admission");
            }
        }
    }
}
