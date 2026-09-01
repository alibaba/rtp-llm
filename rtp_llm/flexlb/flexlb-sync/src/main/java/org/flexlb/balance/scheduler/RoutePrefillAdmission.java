package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.scheduler.ScheduledRequest;
import org.flexlb.balance.delivery.PrefillAdmissionPort;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.dao.route.RoleType;

import java.util.ArrayList;
import java.util.List;
import java.util.OptionalLong;

import static org.flexlb.balance.scheduler.PrefillAdmissionResources.accepted;
import static org.flexlb.balance.scheduler.PrefillAdmissionResources.createCommittedOwner;
import static org.flexlb.balance.scheduler.PrefillAdmissionResources.failed;
import static org.flexlb.balance.scheduler.PrefillAdmissionResources.missingEndpoint;
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
                    "DELIVERY_CAPACITY_PREFILL_REQUEST",
                    RoleType.PREFILL);

    public RoutePrefillAdmission() {
    }

    @Override
    public CapacityBoundary.Attempt<PreparedAdmission> tryBegin(
            ScheduledRequest firstCandidate) {
        final ScheduledRequest exact;
        try {
            exact = firstCandidate;
        } catch (RuntimeException | Error failure) {
            return failed(failure);
        }
        PrefillEndpoint prefill = exact.prefillEp();
        if (prefill == null) {
            return failed(missingEndpoint("Prefill", exact));
        }
        try {
            return accepted(new PreparedTransaction(prefill));
        } catch (Throwable failure) {
            return failed(failure);
        }
    }

    private static CapacityBoundary routeCapacityFull(
            PrefillEndpoint prefill,
            ScheduledRequest item) {
        return CapacityBoundary.unavailable(
                prefill.routeAdmissionAvailability(
                        item.maxInflightDeliveriesPerPrefillWorker()),
                CAPACITY_BLOCK);
    }

    private static final class PreparedTransaction
            implements PreparedAdmission {
        private final PrefillEndpoint prefill;
        private List<PrefillState.RouteReservation> reservations;
        private List<PrefillAdmissionResources.Member> members;
        private boolean ownershipResolved;

        private PreparedTransaction(
                PrefillEndpoint prefill) {
            this.prefill = prefill;
            reservations = new ArrayList<>(1);
            members = new ArrayList<>(1);
        }

        @Override
        public OptionalLong correlationId() {
            return OptionalLong.empty();
        }

        @Override
        public synchronized CapacityBoundary.Attempt<ScheduledRequest> tryAppend(
                ScheduledRequest exactNextItem,
                long predictedMs) {
            requirePrepared("append");
            final ScheduledRequest exact;
            try {
                exact = exactNextItem;
            } catch (RuntimeException | Error failure) {
                return failed(failure);
            }

            final int originalReservationCount = reservations.size();
            final int originalMemberCount = members.size();
            final CapacityBoundary.Attempt<ScheduledRequest> acceptedItem;
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

            PrefillState.RouteReservationResult reservationResult;
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
                    != PrefillState.CapacityStatus.ACQUIRED) {
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
            PrefillState.RouteReservation reservation =
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
            if (!memberAttempt.accepted()) {
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
                            rollbackFailure, memberAttempt.boundary());
                    return failed(rollbackFailure);
                }
                return rejected(memberAttempt.boundary());
            }
            PrefillAdmissionResources.Member member =
                    memberAttempt.value();
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
        public synchronized CommittedAdmission commitPreparedUnderLock(
                List<ScheduledRequest> exactItems,
                long predictedMs) {
            requirePrepared("commit");
            if (!sameIdentitySequence(members, exactItems)) {
                throw new IllegalArgumentException(
                        "route commit does not match prepared identities");
            }
            if (reservations.isEmpty()
                    || reservations.size() != members.size()) {
                throw new IllegalStateException(
                        "route admission reservation/member ownership diverged");
            }
            PrefillState.RouteReservation headReservation =
                    reservations.get(0);
            PrefillAdmissionResources.CommittedAdmissionOwner committedOwner =
                    createCommittedOwner(members, members.size());
            List<PrefillState.CommittedHandoff> handoffs =
                    headReservation.commitGroup(
                            exactItems, reservations);
            committedOwner.bindRouteHandoffs(handoffs);
            ownershipResolved = true;
            reservations = null;
            members = null;
            return committedOwner;
        }

        @Override
        public void close() {
            List<PrefillState.RouteReservation> exactReservations;
            List<PrefillAdmissionResources.Member> exactMembers;
            synchronized (this) {
                if (ownershipResolved) {
                    return;
                }
                ownershipResolved = true;
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
            for (PrefillState.RouteReservation reservation
                    : exactReservations) {
                rollbackFailure = rollbackReservation(
                        reservation, rollbackFailure);
            }
            if (rollbackFailure != null) {
                throwRollbackFailure(rollbackFailure);
            }
        }

        private void requirePrepared(String operation) {
            if (ownershipResolved) {
                throw new IllegalStateException(
                        operation
                                + " requires PREPARED route admission");
            }
        }
    }
}
