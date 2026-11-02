package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.delivery.DeliveryMetrics;
import org.flexlb.balance.delivery.DeliveryStrategy;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.prediction.PrefillPredictionBoundary;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.balance.projection.WorkSnapshot;
import org.flexlb.balance.scheduler.RequestSlot.DeliveryClaim;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.OptionalLong;

import static org.flexlb.balance.scheduler.PrefillAdmissionResources.accepted;
import static org.flexlb.balance.scheduler.PrefillAdmissionResources.createCommittedOwner;
import static org.flexlb.balance.scheduler.PrefillAdmissionResources.failed;
import static org.flexlb.balance.scheduler.PrefillAdmissionResources.missingEndpoint;
import static org.flexlb.balance.scheduler.PrefillAdmissionResources.prepareMember;
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
                        requests.prepareRouteMember(item, transaction, evaluator);
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
            RouteTransaction transaction,
            PrefillAdmissionResources.CommittedAdmissionOwner admission,
            int remainingQueueDepth,
            WorkSnapshot precedingWork) {
        List<ScheduledRequest> items = transaction.items();
        Throwable deliveryFailure = null;
        List<ScheduledRequest> delivered = new ArrayList<>(items.size());
        List<ClaimedRoute> claimed = new ArrayList<>(items.size());
        try {
            for (ScheduledRequest item : items) {
                DeliveryClaim claim;
                try {
                    claim = requests.claimRouteDelivery(
                            item, admission);
                } catch (Throwable claimFailure) {
                    try {
                        requests.failDeliveryPreparation(item, claimFailure);
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
                claimed.add(new ClaimedRoute(item, claim));
            }
            long unstartedWorkMs = 0L;
            for (ClaimedRoute route : claimed) {
                ScheduledRequest item = route.item();
                try {
                    long itemWorkMs = transaction.predictions.get(item);
                    unstartedWorkMs = unstartedWorkMs > Long.MAX_VALUE - itemWorkMs
                            ? Long.MAX_VALUE : unstartedWorkMs + itemWorkMs;
                    requests.publishRoute(route.claim(), precedingWork, unstartedWorkMs);
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

    private record ClaimedRoute(ScheduledRequest item, DeliveryClaim claim) { }

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
    static final class RouteTransaction implements Transaction {
        private final RouteDeliveryStrategy owner;
        private final PrefillEndpoint prefill;
        private List<ScheduledRequest> items = List.of();
        private ScheduledRequest blockedItem;
        private CapacityBoundary blockedResult;
        private ArrayList<PrefillState.RouteReservation> reservations;
        private PrefillEndpoint.RouteCommitAdmission routeAdmission;
        private final java.util.IdentityHashMap<ScheduledRequest, Long> predictions = new java.util.IdentityHashMap<>();
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

        synchronized CapacityBoundary.Attempt<ScheduledRequest> append(
                ScheduledRequest exact,
                PrefillTimePredictor.Evaluator evaluator) {
            requirePrepared("append");
            long predictedMs = predict(evaluator, exact);
            final CapacityBoundary.Attempt<ScheduledRequest> acceptedItem;
            try {
                acceptedItem = accepted(exact);
                int capacity = Math.addExact(members.size(), 1);
                reservations.ensureCapacity(capacity);
                members.ensureCapacity(capacity);
                predictions.put(exact, predictedMs);
            } catch (Throwable failure) {
                return failed(failure);
            }

            final PrefillState.RouteReservation reservation;
            try {
                if (routeAdmission == null) {
                    routeAdmission = prefill.tryBeginRouteCommitAdmission();
                    if (routeAdmission == null) {
                        return failed(PrefillAdmissionResources.retired("Prefill", exact));
                    }
                }
                var result = routeAdmission.reserve(exact, predictedMs);
                if (result.status() != PrefillState.CapacityStatus.ACQUIRED) {
                    return rejectedPrefill(exact, result.status(), CapacityBoundary.deliveryUnavailable(
                            prefill.routeAdmissionAvailability()));
                }
                reservation = result.reservation();
            } catch (Throwable failure) {
                return failed(failure);
            }
            final CapacityBoundary.Attempt<PrefillAdmissionResources.Member>
                    memberAttempt;
            try {
                memberAttempt = prepareMember(exact);
            } catch (Throwable failure) {
                return failed(rollbackReservation(reservation, failure));
            }
            if (!memberAttempt.accepted()) {
                Throwable cleanup = rollbackReservation(reservation, null);
                return cleanup == null ? rejected(memberAttempt.boundary()) : failed(cleanup);
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
        public synchronized WorkSnapshot commitUnderLock() {
            requirePrepared("commit");
            if (!sameIdentitySequence(members, items)) {
                throw new IllegalArgumentException(
                        "route commit does not match prepared identities");
            }
            if (reservations.isEmpty() || reservations.size() != members.size()) {
                throw new IllegalStateException("route admission reservation/member ownership diverged");
            }
            PrefillAdmissionResources.CommittedAdmissionOwner exactCommitted = createCommittedOwner(members);
            PrefillState.CommittedHandoff handoff = routeAdmission.commit(items, reservations);
            exactCommitted.bindPrefillHandoff(handoff);
            committed = exactCommitted;
            routeAdmission = null;
            reservations = null;
            members = null;
            return handoff.precedingWork();
        }

        @Override
        public void handoff(
                String decisionReason, int remainingQueueDepth,
                WorkSnapshot precedingWork) {
            PrefillAdmissionResources.CommittedAdmissionOwner exactCommitted =
                    takeCommitted();
            owner.deliver(this, exactCommitted, remainingQueueDepth,
                    Objects.requireNonNull(precedingWork, "precedingWork"));
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
            PrefillEndpoint.RouteCommitAdmission exactAdmission;
            List<PrefillAdmissionResources.Member> exactMembers;
            synchronized (this) {
                if (reservations == null) {
                    return priorFailure;
                }
                exactReservations = reservations;
                exactAdmission = routeAdmission;
                exactMembers = members;
                reservations = null;
                routeAdmission = null;
                members = null;
            }
            Throwable failure = priorFailure;
            for (PrefillAdmissionResources.Member member : exactMembers) {
                failure = rollbackMember(member, failure);
            }
            for (PrefillState.RouteReservation reservation : exactReservations) {
                failure = rollbackReservation(reservation, failure);
            }
            // Release the generation only after every provisional send slot is returned.
            if (exactAdmission != null) {
                failure = RouteDeliveryStrategy.append(failure, RouteDeliveryStrategy.close(exactAdmission));
            }
            return failure;
        }

        private void requirePrepared(String operation) {
            if (reservations == null
                    || members == null) {
                throw new IllegalStateException(
                        operation + " requires PREPARED route admission");
            }
        }
    }

    private static final class RouteProjectionPolicy
            implements RouteProjection.DeliveryProjection {

        private static final ThreadLocal<RoutePlanning> PLANNING =
                ThreadLocal.withInitial(RoutePlanning::new);
        private static final ThreadLocal<RouteService> SERVICE =
                ThreadLocal.withInitial(RouteService::new);

        @Override
        public long singletonCompletionOffsetMs(
                long seqLen,
                long hitCache,
                RouteProjection.Predictions predictions) {
            return predictions.itemDurationMs(seqLen, hitCache);
        }

        @Override
        public RouteProjection.GroupPlanning planning(
                RouteProjection.Predictions predictions) {
            RoutePlanning planning = PLANNING.get();
            planning.reset(predictions);
            return planning;
        }

        @Override
        public RouteProjection.GroupService service(
                GroupPlanner.Plan<GroupPlanner.Item> plan,
                RouteProjection.Predictions predictions) {
            RouteService service = SERVICE.get();
            service.reset(plan, predictions);
            return service;
        }

        private static final class RoutePlanning
                implements RouteProjection.GroupPlanning {
            private GroupPlanner.Item[] previous = new GroupPlanner.Item[0];
            private long[] offsets = new long[0];
            private int previousSize;
            private int computedThrough;
            private RouteProjection.Predictions predictions;

            private void reset(RouteProjection.Predictions exactPredictions) {
                predictions = exactPredictions;
                previousSize = 0;
                computedThrough = -1;
            }

            @Override
            public double durationMs(
                    List<GroupPlanner.Item> prefix,
                    int requiredThroughIndex) {
                if (requiredThroughIndex < 0
                        || requiredThroughIndex >= prefix.size()) {
                    throw new IndexOutOfBoundsException(requiredThroughIndex);
                }
                if (!isIdentityPrefix(prefix)) {
                    ensureCapacity(prefix.size());
                    for (int index = 0; index < prefix.size(); index++) {
                        previous[index] = prefix.get(index);
                    }
                    previousSize = prefix.size();
                    computedThrough = -1;
                } else if (previousSize < prefix.size()) {
                    ensureCapacity(prefix.size());
                    for (int index = previousSize;
                            index < prefix.size(); index++) {
                        previous[index] = prefix.get(index);
                    }
                    previousSize = prefix.size();
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

            private boolean isIdentityPrefix(List<GroupPlanner.Item> next) {
                if (previousSize > next.size()) {
                    return false;
                }
                for (int index = 0; index < previousSize; index++) {
                    if (previous[index] != next.get(index)) {
                        return false;
                    }
                }
                return true;
            }

            private void ensureCapacity(int size) {
                if (previous.length >= size) {
                    return;
                }
                previous = java.util.Arrays.copyOf(previous, size);
                offsets = java.util.Arrays.copyOf(offsets, size);
            }
        }

        private static final class RouteService
                implements RouteProjection.GroupService {
            private GroupPlanner.Plan<GroupPlanner.Item> plan;
            private RouteProjection.Predictions predictions;
            private long[] offsets = new long[0];
            private int computedThrough;

            private void reset(
                    GroupPlanner.Plan<GroupPlanner.Item> exactPlan,
                    RouteProjection.Predictions exactPredictions) {
                plan = exactPlan;
                predictions = exactPredictions;
                if (offsets.length < exactPlan.items().size()) {
                    offsets = java.util.Arrays.copyOf(
                            offsets, exactPlan.items().size());
                }
                computedThrough = -1;
            }

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
        }

        private static long saturatedAdd(long left, long right) {
            return left > Long.MAX_VALUE - right
                    ? Long.MAX_VALUE : left + right;
        }
    }
}
