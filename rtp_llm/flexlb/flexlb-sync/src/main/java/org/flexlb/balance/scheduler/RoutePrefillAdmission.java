package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.delivery.DeliveryItem;
import org.flexlb.balance.delivery.PrefillAdmissionPort;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillWorkLedger;
import org.flexlb.balance.projection.RouteProjection;

import java.util.ArrayList;
import java.util.List;
import java.util.OptionalLong;

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
 * Individual route admission.
 *
 * <p>Every group member owns its own Prefill route reservation. The adapter has no
 * transport correlation because the route response itself is the delivery
 * boundary.</p>
 */
public final class RoutePrefillAdmission implements PrefillAdmissionPort {

    private static final RouteProjection.AdmissionBlockSemantics
            CAPACITY_BLOCK = new RouteProjection.AdmissionBlockSemantics(
                    "DELIVERY_CAPACITY_PREFILL_REQUEST",
                    RouteProjection.AfterProbeAdmission.BLOCKED,
                    "DELIVERY_CAPACITY_PREFILL_REQUEST");

    public RoutePrefillAdmission() {
    }

    @Override
    public CapacityBoundary.Attempt<PreparedAdmission> prepare(
            DeliveryItem head,
            long predictedMs) {
        final BatchItem exactHead;
        try {
            exactHead = (BatchItem) head;
        } catch (RuntimeException | Error failure) {
            return failed(failure);
        }
        return prepareHead(exactHead, predictedMs);
    }

    private CapacityBoundary.Attempt<PreparedAdmission> prepareHead(
            BatchItem exactHead,
            long predictedMs) {
        PrefillEndpoint prefill = exactHead.prefillEp();
        if (prefill == null) {
            return failed(missingEndpoint("Prefill", exactHead));
        }
        final PreparedTransaction transaction;
        final CapacityBoundary.Attempt<PreparedAdmission> acceptedTransaction;
        try {
            transaction = new PreparedTransaction(prefill);
            acceptedTransaction = accepted(transaction);
        } catch (Throwable failure) {
            return failed(failure);
        }
        PrefillWorkLedger.RouteReservationResult reservationResult;
        try {
            reservationResult = prefill.reserveRoute(
                    exactHead,
                    predictedMs,
                    exactHead.maxInflightDeliveriesPerPrefillWorker());
        } catch (RuntimeException | Error failure) {
            return failed(failure);
        }
        if (reservationResult.status()
                != PrefillWorkLedger.CapacityStatus.ACQUIRED) {
            return rejectedPrefill(
                    exactHead,
                    reservationResult.status(),
                    routeCapacityFull(prefill, exactHead));
        }

        PrefillWorkLedger.RouteReservation reservation =
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

    private static CapacityBoundary.Unavailable routeCapacityFull(
            PrefillEndpoint prefill,
            BatchItem item) {
        return new CapacityBoundary.Unavailable(
                prefill.routeAdmissionAvailability(
                        item.maxInflightDeliveriesPerPrefillWorker()),
                CAPACITY_BLOCK);
    }

    private static final class PreparedTransaction
            implements PreparedAdmission {
        private final PrefillEndpoint prefill;
        private List<PrefillWorkLedger.RouteReservation> reservations;
        private List<PrefillAdmissionResources.Member> members;
        private PrefillAdmissionResources.PreparedState state = PREPARED;

        private PreparedTransaction(
                PrefillEndpoint prefill) {
            this.prefill = prefill;
            reservations = new ArrayList<>(1);
            reservations.add(null);
            members = new ArrayList<>(1);
            members.add(null);
        }

        /** Bind only references into storage allocated before acquisition. */
        private void bind(
                PrefillWorkLedger.RouteReservation headReservation,
                PrefillAdmissionResources.Member head) {
            reservations.set(0, headReservation);
            members.set(0, head);
        }

        @Override
        public OptionalLong correlationId() {
            return OptionalLong.empty();
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

            final int originalReservationCount = reservations.size();
            final int originalMemberCount = members.size();
            final CapacityBoundary.Attempt<DeliveryItem> acceptedItem;
            try {
                acceptedItem = accepted(exact);
                reservations.add(null);
                members.add(null);
            } catch (Throwable failure) {
                Throwable aggregate = restorePreparedTail(
                        members, originalMemberCount, failure);
                aggregate = restorePreparedTail(
                        reservations, originalReservationCount, aggregate);
                return failed(aggregate);
            }

            PrefillWorkLedger.RouteReservationResult reservationResult;
            try {
                reservationResult = prefill.reserveRoute(
                        exact,
                        predictedMs,
                        exact.maxInflightDeliveriesPerPrefillWorker());
            } catch (Throwable failure) {
                Throwable aggregate = restorePreparedTail(
                        members, originalMemberCount, failure);
                aggregate = restorePreparedTail(
                        reservations, originalReservationCount, aggregate);
                return failed(aggregate);
            }
            if (reservationResult.status()
                    != PrefillWorkLedger.CapacityStatus.ACQUIRED) {
                Throwable rollbackFailure = restorePreparedTail(
                        members, originalMemberCount, null);
                rollbackFailure = restorePreparedTail(
                        reservations,
                        originalReservationCount,
                        rollbackFailure);
                if (rollbackFailure != null) {
                    return failed(rollbackFailure);
                }
                return rejectedPrefill(
                        exact,
                        reservationResult.status(),
                        routeCapacityFull(prefill, exact));
            }
            PrefillWorkLedger.RouteReservation reservation =
                    reservationResult.reservation();
            final CapacityBoundary.Attempt<PrefillAdmissionResources.Member>
                    memberAttempt;
            try {
                memberAttempt = prepareMember(exact);
            } catch (Throwable failure) {
                Throwable aggregate = restorePreparedTail(
                        members, originalMemberCount, failure);
                aggregate = restorePreparedTail(
                        reservations, originalReservationCount, aggregate);
                aggregate = rollbackReservation(reservation, aggregate);
                return failed(aggregate);
            }
            if (memberAttempt
                    instanceof CapacityBoundary.Attempt.Rejected<
                            PrefillAdmissionResources.Member> rejectedMember) {
                Throwable rollbackFailure = restorePreparedTail(
                        members, originalMemberCount, null);
                rollbackFailure = restorePreparedTail(
                        reservations,
                        originalReservationCount,
                        rollbackFailure);
                rollbackFailure = rollbackReservation(
                        reservation, rollbackFailure);
                if (rollbackFailure != null) {
                    preserveRejectedCause(
                            rollbackFailure, rejectedMember.boundary());
                    return failed(rollbackFailure);
                }
                return rejected(rejectedMember.boundary());
            }
            PrefillAdmissionResources.Member member =
                    ((CapacityBoundary.Attempt.Accepted<
                            PrefillAdmissionResources.Member>) memberAttempt)
                            .value();
            try {
                reservations.set(originalReservationCount, reservation);
                members.set(originalMemberCount, member);
            } catch (Throwable bindFailure) {
                Throwable failure = restorePreparedTail(
                        members, originalMemberCount, bindFailure);
                failure = restorePreparedTail(
                        reservations, originalReservationCount, failure);
                failure = rollbackMember(member, failure);
                failure = rollbackReservation(reservation, failure);
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
                        "route commit does not match prepared identities");
            }
            assert !reservations.isEmpty()
                    : "route admission has no head reservation";
            assert reservations.size() == members.size()
                    : "route admission reservation/member count diverged";
            PrefillWorkLedger.RouteReservation headReservation =
                    reservations.get(0);
            PrefillAdmissionResources.CommittedAdmissionAdapter committedOwner =
                    prepareCommitted(members, members.size());
            List<PrefillWorkLedger.CommittedHandoff> handoffs =
                    headReservation.commitGroupUnderLock(
                            exactItems, reservations);
            committedOwner.bind(handoffs);
            state = COMMITTED;
            reservations = null;
            members = null;
            return committedOwner;
        }

        @Override
        public void close() {
            List<PrefillWorkLedger.RouteReservation> exactReservations;
            List<PrefillAdmissionResources.Member> exactMembers;
            synchronized (this) {
                if (state != PREPARED) {
                    return;
                }
                state = CLOSED;
                exactReservations = reservations;
                exactMembers = members;
                reservations = List.of();
                members = List.of();
            }
            Throwable rollbackFailure = null;
            for (PrefillAdmissionResources.Member member : exactMembers) {
                rollbackFailure = rollbackMember(
                        member, rollbackFailure);
            }
            for (PrefillWorkLedger.RouteReservation reservation
                    : exactReservations) {
                rollbackFailure = rollbackReservation(
                        reservation, rollbackFailure);
            }
            if (rollbackFailure != null) {
                throwRollbackFailure(rollbackFailure);
            }
        }

        private void requirePrepared(String operation) {
            if (state != PREPARED) {
                throw new IllegalStateException(
                        operation
                                + " requires PREPARED route admission");
            }
        }
    }
}
