package org.flexlb.balance.delivery;

import org.flexlb.balance.scheduler.ScheduledRequest;

import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.prediction.PrefillPredictionBoundary;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.projection.RouteProjection;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.OptionalLong;

/** Individual route admission, ownership, publication, and projection. */
public final class RouteDeliveryStrategy implements DeliveryStrategy {

    private static final RouteProjection.DeliveryProjection PROJECTION =
            new RouteProjectionPolicy();

    private final PrefillAdmissionPort admissionPort;
    private final SlotDeliveryPort slotPort;
    private final DeliveryMetrics telemetry;

    public RouteDeliveryStrategy(
            PrefillAdmissionPort admissionPort,
            SlotDeliveryPort slotPort,
            DeliveryMetrics telemetry) {
        this.admissionPort = Objects.requireNonNull(
                admissionPort, "admissionPort");
        this.slotPort = Objects.requireNonNull(slotPort, "slotPort");
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
        CapacityBoundary.Attempt<PrefillAdmissionPort.PreparedAdmission>
                beginAttempt = admissionPort.tryBegin(head);
        if (!beginAttempt.accepted()) {
            return new RouteTransaction(
                    this,
                    List.of(),
                    null,
                    head,
                    beginAttempt.boundary());
        }
        PrefillAdmissionPort.PreparedAdmission prepared =
                beginAttempt.value();
        ScheduledRequest blockedItem = null;
        CapacityBoundary blockedResult = null;
        try {
            for (ScheduledRequest item : candidates) {
                CapacityBoundary.Attempt<ScheduledRequest> attempt =
                        slotPort.prepareIfOwned(
                                item,
                                () -> prepared.tryAppend(
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
                Throwable cleanup = close(prepared);
                if (cleanup != null) {
                    throw cleanup;
                }
                return new RouteTransaction(
                        this, List.of(), null, blockedItem, blockedResult);
            }
            return new RouteTransaction(
                    this, admitted, prepared, blockedItem, blockedResult);
        } catch (Throwable failure) {
            Throwable cleanup = close(prepared);
            if (cleanup != null && cleanup != failure) {
                failure.addSuppressed(cleanup);
            }
            throw propagate(failure);
        }
    }

    private void deliver(
            List<ScheduledRequest> items,
            PrefillAdmissionPort.CommittedAdmission admission,
            DeliveryMetadata metadata) {
        Throwable deliveryFailure = null;
        List<ScheduledRequest> delivered = new ArrayList<>(items.size());
        try {
            for (ScheduledRequest item : items) {
                SlotDeliveryPort.Claim claim;
                try {
                    claim = slotPort.tryClaimForDelivery(
                            item,
                            SlotDeliveryPort.Identity.commitConfirmation(),
                            () -> admission.transferToEndpoint(item));
                } catch (Throwable claimFailure) {
                    try {
                        slotPort.failPrepared(item, claimFailure);
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
                    slotPort.complete(
                            claim,
                            SlotDeliveryPort.Completion.delivered());
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
            telemetry.routesDelivered(metadata, delivered);
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
        private final List<ScheduledRequest> items;
        private final ScheduledRequest blockedItem;
        private final CapacityBoundary blockedResult;
        private PrefillAdmissionPort.PreparedAdmission prepared;
        private PrefillAdmissionPort.CommittedAdmission committed;

        private RouteTransaction(
                RouteDeliveryStrategy owner,
                List<ScheduledRequest> items,
                PrefillAdmissionPort.PreparedAdmission prepared,
                ScheduledRequest blockedItem,
                CapacityBoundary blockedResult) {
            this.owner = owner;
            this.items = items;
            this.prepared = prepared;
            this.blockedItem = blockedItem;
            this.blockedResult = blockedResult;
            if (items.isEmpty() != (prepared == null)) {
                throw new IllegalArgumentException(
                        "route transaction owns preparation exactly when active");
            }
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
            PrefillAdmissionPort.PreparedAdmission exactPrepared = prepared;
            if (exactPrepared == null) {
                throw new IllegalStateException(
                        "route group no longer owns a prepared admission");
            }
            committed = exactPrepared.commitPreparedUnderLock(items, 0L);
            prepared = null;
        }

        @Override
        public void handoff(DeliveryMetadata metadata) {
            PrefillAdmissionPort.CommittedAdmission exactCommitted =
                    takeCommitted();
            owner.deliver(items, exactCommitted, metadata);
        }

        @Override
        public void abort(Throwable cause) {
            PrefillAdmissionPort.CommittedAdmission exactCommitted;
            synchronized (this) {
                if (prepared == null && committed == null) {
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
        public synchronized void close() {
            PrefillAdmissionPort.PreparedAdmission exactPrepared = prepared;
            if (exactPrepared == null) {
                return;
            }
            prepared = null;
            Throwable cleanup = RouteDeliveryStrategy.close(exactPrepared);
            if (cleanup != null) {
                throw propagate(cleanup);
            }
        }

        private synchronized PrefillAdmissionPort.CommittedAdmission
                takeCommitted() {
            PrefillAdmissionPort.CommittedAdmission exactCommitted = committed;
            if (exactCommitted == null) {
                throw new IllegalStateException(
                        "route group no longer owns a committed admission");
            }
            committed = null;
            return exactCommitted;
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
