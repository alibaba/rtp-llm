package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.delivery.DeliveryMetrics;
import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.balance.delivery.DeliveryStrategy;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.prediction.PrefillPredictionBoundary;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.dao.route.RoleType;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.OptionalLong;

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

/** Individual route admission, ownership, publication, and projection. */
public final class RouteDeliveryStrategy implements DeliveryStrategy {

    private static final RouteProjection.DeliveryProjection PROJECTION =
            new RouteProjectionPolicy();
    private static final RouteProjection.AdmissionBlockSemantics
            CAPACITY_BLOCK = new RouteProjection.AdmissionBlockSemantics(
                    "DELIVERY_CAPACITY_PREFILL_REQUEST",
                    RouteProjection.AfterProbeAdmission.BLOCKED,
                    "DELIVERY_CAPACITY_PREFILL_REQUEST",
                    RoleType.PREFILL);

    private final RequestRegistry requests;
    private final DeliveryMetrics telemetry;

    public RouteDeliveryStrategy(
            RequestRegistry requests,
            DeliveryMetrics telemetry) {
        this.requests = Objects.requireNonNull(requests, "requests");
        this.telemetry = Objects.requireNonNull(telemetry, "telemetry");
    }

    @Override
    public Transaction prepare(
            List<ScheduledRequest> candidates,
            PrefillTimePredictor.Evaluator evaluator,
            OptionalLong plannedPredictionMs) {
        if (candidates.isEmpty()) {
            throw new IllegalArgumentException(
                    "route delivery requires at least one candidate");
        }
        return prepareTransaction(candidates, evaluator);
    }

    private RouteTransaction prepareTransaction(
            List<ScheduledRequest> candidates,
            PrefillTimePredictor.Evaluator evaluator) {
        ScheduledRequest head = candidates.get(0);
        List<ScheduledRequest> admitted = new ArrayList<>(candidates.size());
        PrefillEndpoint prefill = head.prefillEp();
        if (prefill == null) {
            return RouteTransaction.blocked(
                    this,
                    head,
                    CapacityBoundary.failed(missingEndpoint("Prefill", head)));
        }
        RouteTransaction transaction = new RouteTransaction(this, prefill);
        ScheduledRequest blockedItem = null;
        CapacityBoundary blockedResult = null;
        try {
            for (ScheduledRequest item : candidates) {
                CapacityBoundary.Attempt<ScheduledRequest> attempt =
                        requests.prepareIfOwned(
                                item,
                                () -> transaction.append(
                                        item,
                                        predict(evaluator, item)))
                        .orElseGet(
                                RouteDeliveryStrategy::<ScheduledRequest>
                                        ownershipLost);
                if (!attempt.accepted()) {
                    blockedItem = item;
                    blockedResult = attempt.boundary();
                    break;
                }
                admitted.add(item);
            }
            if (admitted.isEmpty()) {
                Throwable cleanup = close(transaction);
                if (cleanup != null) {
                    throw cleanup;
                }
                return RouteTransaction.blocked(
                        this, blockedItem, blockedResult);
            }
            transaction.select(admitted, blockedItem, blockedResult);
            return transaction;
        } catch (Throwable failure) {
            Throwable cleanup = close(transaction);
            if (cleanup != null && cleanup != failure) {
                failure.addSuppressed(cleanup);
            }
            throw propagate(failure);
        }
    }

    private void deliver(
            List<ScheduledRequest> items,
            PrefillAdmissionResources.CommittedAdmissionOwner admission,
            int remainingQueueDepth) {
        Throwable deliveryFailure = null;
        List<ScheduledRequest> delivered = new ArrayList<>(items.size());
        try {
            for (ScheduledRequest item : items) {
                RequestRegistry.DeliveryClaim claim;
                try {
                    claim = requests.tryClaimRouteDelivery(
                            item,
                            () -> admission.transferToEndpoint(item));
                } catch (Throwable claimFailure) {
                    try {
                        requests.failPrepared(item, claimFailure);
                    } catch (Throwable terminalFailure) {
                        claimFailure.addSuppressed(terminalFailure);
                        deliveryFailure = append(
                                deliveryFailure, claimFailure);
                    }
                    continue;
                }
                if (claim == null) {
                    continue;
                }
                try {
                    requests.complete(claim, DeliveryResult.delivered());
                    delivered.add(item);
                } catch (Throwable completionFailure) {
                    deliveryFailure = append(
                            deliveryFailure, completionFailure);
                }
            }
        } finally {
            deliveryFailure = append(
                    deliveryFailure, close(admission));
        }
        if (!delivered.isEmpty()) {
            telemetry.routesDelivered(remainingQueueDepth, delivered);
        }
        if (deliveryFailure != null) {
            throw propagate(deliveryFailure);
        }
    }

    @Override
    public double projectGroupDurationMs(
            List<ScheduledRequest> items,
            PrefillTimePredictor.Evaluator evaluator) {
        double totalMs = 0.0;
        for (ScheduledRequest item : items) {
            totalMs += PrefillPredictionBoundary.predictSingleRequestMs(
                    evaluator, item.seqLen(), item.hitCache());
        }
        return PrefillPredictionBoundary.requireValidDecisionGroupMs(totalMs);
    }

    @Override
    public RouteProjection.DeliveryProjection projectionPolicy() {
        return PROJECTION;
    }

    private static long predict(
            PrefillTimePredictor.Evaluator evaluator,
            ScheduledRequest item) {
        return PrefillPredictionBoundary.predictSingleRequestMs(
                evaluator, item.seqLen(), item.hitCache());
    }

    private static CapacityBoundary routeCapacityFull(
            PrefillEndpoint prefill,
            ScheduledRequest item) {
        return CapacityBoundary.unavailable(
                prefill.routeAdmissionAvailability(
                        item.maxInflightDeliveriesPerPrefillWorker()),
                CAPACITY_BLOCK);
    }

    private static <T> CapacityBoundary.Attempt<T> ownershipLost() {
        return CapacityBoundary.Attempt.rejected(
                CapacityBoundary.OWNERSHIP_LOST);
    }

    private static Throwable close(AutoCloseable capability) {
        try {
            capability.close();
            return null;
        } catch (Throwable failure) {
            return failure;
        }
    }

    private static Throwable append(Throwable first, Throwable next) {
        if (next == null) {
            return first;
        }
        if (first == null) {
            return next;
        }
        if (first != next) {
            first.addSuppressed(next);
        }
        return first;
    }

    private static RuntimeException propagate(Throwable failure) {
        if (failure instanceof RuntimeException runtimeFailure) {
            return runtimeFailure;
        }
        if (failure instanceof Error error) {
            throw error;
        }
        return new IllegalStateException("route delivery failed", failure);
    }

    /** Ordered route callback payload and sole owner of its exact members. */
    private static final class RouteTransaction implements Transaction {
        private final RouteDeliveryStrategy owner;
        private final PrefillEndpoint prefill;
        private List<ScheduledRequest> items = List.of();
        private ScheduledRequest blockedItem;
        private CapacityBoundary blockedResult;
        private ArrayList<PrefillState.RouteReservation> reservations;
        private ArrayList<PrefillAdmissionResources.Member> members;
        private PrefillAdmissionResources.CommittedAdmissionOwner committed;

        private RouteTransaction(
                RouteDeliveryStrategy owner,
                PrefillEndpoint prefill) {
            this.owner = owner;
            this.prefill = prefill;
            reservations = new ArrayList<>(1);
            members = new ArrayList<>(1);
        }

        private RouteTransaction(
                RouteDeliveryStrategy owner,
                ScheduledRequest blockedItem,
                CapacityBoundary blockedResult) {
            this.owner = owner;
            this.prefill = null;
            this.blockedItem = blockedItem;
            this.blockedResult = blockedResult;
        }

        private static RouteTransaction blocked(
                RouteDeliveryStrategy owner,
                ScheduledRequest blockedItem,
                CapacityBoundary blockedResult) {
            return new RouteTransaction(owner, blockedItem, blockedResult);
        }

        private synchronized CapacityBoundary.Attempt<ScheduledRequest> append(
                ScheduledRequest exact,
                long predictedMs) {
            requirePrepared("append");
            final CapacityBoundary.Attempt<ScheduledRequest> acceptedItem;
            try {
                acceptedItem = accepted(exact);
                int capacity = Math.addExact(members.size(), 1);
                reservations.ensureCapacity(capacity);
                members.ensureCapacity(capacity);
            } catch (Throwable failure) {
                return failed(failure);
            }

            PrefillState.ReservationResult<PrefillState.RouteReservation> result;
            try {
                result = prefill.reserveRoute(
                        exact,
                        predictedMs,
                        exact.maxInflightDeliveriesPerPrefillWorker());
            } catch (Throwable failure) {
                return failed(failure);
            }
            if (result.status() != PrefillState.CapacityStatus.ACQUIRED) {
                return rejectedPrefill(
                        exact,
                        result.status(),
                        routeCapacityFull(prefill, exact));
            }

            PrefillState.RouteReservation reservation = result.reservation();
            final CapacityBoundary.Attempt<PrefillAdmissionResources.Member>
                    memberAttempt;
            try {
                memberAttempt = prepareMember(exact);
            } catch (Throwable failure) {
                return failed(rollbackReservation(reservation, failure));
            }
            if (!memberAttempt.accepted()) {
                Throwable rollbackFailure = rollbackReservation(
                        reservation, null);
                if (rollbackFailure != null) {
                    preserveRejectedCause(
                            rollbackFailure, memberAttempt.boundary());
                    return failed(rollbackFailure);
                }
                return rejected(memberAttempt.boundary());
            }
            reservations.add(reservation);
            members.add(memberAttempt.value());
            return acceptedItem;
        }

        private synchronized void select(
                List<ScheduledRequest> exactItems,
                ScheduledRequest exactBlockedItem,
                CapacityBoundary exactBlockedResult) {
            requirePrepared("select");
            items = exactItems;
            blockedItem = exactBlockedItem;
            blockedResult = exactBlockedResult;
        }

        @Override
        public List<ScheduledRequest> items() {
            return items;
        }

        @Override
        public ScheduledRequest blockedItem() {
            return blockedItem;
        }

        @Override
        public CapacityBoundary blockedResult() {
            return blockedResult;
        }

        @Override
        public synchronized void commitUnderLock() {
            requirePrepared("commit");
            if (!sameIdentitySequence(members, items)) {
                throw new IllegalArgumentException(
                        "route commit does not match prepared identities");
            }
            if (reservations.isEmpty()
                    || reservations.size() != members.size()) {
                throw new IllegalStateException(
                        "route admission reservation/member ownership diverged");
            }
            PrefillAdmissionResources.CommittedAdmissionOwner exactCommitted =
                    createCommittedOwner(members, members.size());
            List<PrefillState.CommittedHandoff> handoffs =
                    reservations.get(0).commitGroup(items, reservations);
            exactCommitted.bindRouteHandoffs(handoffs);
            committed = exactCommitted;
            reservations = null;
            members = null;
        }

        @Override
        public void handoff(
                String decisionReason, int remainingQueueDepth) {
            PrefillAdmissionResources.CommittedAdmissionOwner exactCommitted =
                    takeCommitted();
            owner.deliver(items, exactCommitted, remainingQueueDepth);
        }

        @Override
        public void abort(Throwable cause) {
            PrefillAdmissionResources.CommittedAdmissionOwner exactCommitted;
            synchronized (this) {
                if (reservations == null && committed == null) {
                    return;
                }
                if (committed == null) {
                    throw new IllegalStateException(
                            "route group has not committed its admission");
                }
                exactCommitted = committed;
                committed = null;
            }
            Throwable cleanup = RouteDeliveryStrategy.close(exactCommitted);
            if (cleanup != null) {
                throw propagate(cleanup);
            }
        }

        @Override
        public void close() {
            Throwable rollbackFailure = rollbackPrepared(null);
            if (rollbackFailure != null) {
                throwRollbackFailure(rollbackFailure);
            }
        }

        private synchronized PrefillAdmissionResources.CommittedAdmissionOwner
                takeCommitted() {
            PrefillAdmissionResources.CommittedAdmissionOwner exactCommitted =
                    committed;
            if (exactCommitted == null) {
                throw new IllegalStateException(
                        "route group no longer owns a committed admission");
            }
            committed = null;
            return exactCommitted;
        }

        private Throwable rollbackPrepared(Throwable priorFailure) {
            List<PrefillState.RouteReservation> exactReservations;
            List<PrefillAdmissionResources.Member> exactMembers;
            synchronized (this) {
                if (reservations == null) {
                    return priorFailure;
                }
                exactReservations = reservations;
                exactMembers = members;
                reservations = null;
                members = null;
            }
            Throwable failure = priorFailure;
            for (PrefillAdmissionResources.Member member : exactMembers) {
                failure = rollbackMember(member, failure);
            }
            for (PrefillState.RouteReservation reservation
                    : exactReservations) {
                failure = rollbackReservation(reservation, failure);
            }
            return failure;
        }

        private void requirePrepared(String operation) {
            if (reservations == null || members == null) {
                throw new IllegalStateException(
                        operation + " requires PREPARED route admission");
            }
        }
    }

    private static final class RouteProjectionPolicy
            implements RouteProjection.DeliveryProjection {

        @Override
        public RouteProjection.GroupPlanning planning(
                RouteProjection.Predictions predictions) {
            return new RouteProjection.GroupPlanning() {
                private List<GroupPlanner.Item> previous = List.of();
                private long[] offsets = new long[0];
                private int computedThrough = -1;

                @Override
                public double durationMs(
                        List<GroupPlanner.Item> prefix,
                        int requiredThroughIndex) {
                    if (requiredThroughIndex < 0
                            || requiredThroughIndex >= prefix.size()) {
                        throw new IndexOutOfBoundsException(requiredThroughIndex);
                    }
                    if (!isIdentityPrefix(previous, prefix)) {
                        previous = List.copyOf(prefix);
                        offsets = new long[prefix.size()];
                        computedThrough = -1;
                    } else if (offsets.length < prefix.size()) {
                        offsets = java.util.Arrays.copyOf(
                                offsets, prefix.size());
                        previous = List.copyOf(prefix);
                    }
                    while (computedThrough < requiredThroughIndex) {
                        int next = computedThrough + 1;
                        long prior = next == 0 ? 0L : offsets[next - 1];
                        offsets[next] = saturatedAdd(
                                prior,
                                predictions.itemDurationMs(prefix.get(next)));
                        computedThrough = next;
                    }
                    return offsets[requiredThroughIndex];
                }
            };
        }

        @Override
        public RouteProjection.GroupService service(
                GroupPlanner.Plan<GroupPlanner.Item> plan,
                RouteProjection.Predictions predictions) {
            return new RouteProjection.GroupService() {
                private final long[] offsets = new long[plan.items().size()];
                private int computedThrough = -1;

                @Override
                public long completionOffsetMs(int memberIndex) {
                    if (memberIndex < 0 || memberIndex >= plan.items().size()) {
                        throw new IndexOutOfBoundsException(memberIndex);
                    }
                    while (computedThrough < memberIndex) {
                        int next = computedThrough + 1;
                        long prior = next == 0 ? 0L : offsets[next - 1];
                        offsets[next] = saturatedAdd(
                                prior,
                                predictions.itemDurationMs(
                                        plan.items().get(next)));
                        computedThrough = next;
                    }
                    return offsets[memberIndex];
                }

                @Override
                public long totalDurationMs() {
                    return completionOffsetMs(plan.items().size() - 1);
                }
            };
        }

        private static boolean isIdentityPrefix(
                List<GroupPlanner.Item> previous,
                List<GroupPlanner.Item> next) {
            if (previous.size() > next.size()) {
                return false;
            }
            for (int index = 0; index < previous.size(); index++) {
                if (previous.get(index) != next.get(index)) {
                    return false;
                }
            }
            return true;
        }

        private static long saturatedAdd(long left, long right) {
            return left > Long.MAX_VALUE - right
                    ? Long.MAX_VALUE : left + right;
        }
    }
}
