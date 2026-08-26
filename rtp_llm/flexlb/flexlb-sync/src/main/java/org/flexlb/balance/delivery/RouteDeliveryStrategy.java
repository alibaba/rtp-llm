package org.flexlb.balance.delivery;

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
    private final DeliveryTelemetry telemetry;

    public RouteDeliveryStrategy(
            PrefillAdmissionPort admissionPort,
            SlotDeliveryPort slotPort,
            DeliveryTelemetry telemetry) {
        this.admissionPort = Objects.requireNonNull(
                admissionPort, "admissionPort");
        this.slotPort = Objects.requireNonNull(slotPort, "slotPort");
        this.telemetry = Objects.requireNonNull(telemetry, "telemetry");
    }

    @Override
    public <R> R admitAndDeliver(
            List<DeliveryItem> candidates,
            DeliveryMetadata metadata,
            PrefillTimePredictor.Evaluator evaluator,
            OptionalLong plannedPrediction,
            DeliveryContext<R> context) {
        if (candidates.isEmpty()
                || !context.selectionStillOwned(candidates)) {
            return context.noAction();
        }

        Prefix prefix = reservePrefix(candidates, evaluator);
        if (prefix.items().isEmpty()) {
            return context.commitBoundary(prefix.boundary());
        }

        final AdmittedGroup group;
        try {
            group = new AdmittedGroup(
                    this, prefix.items(), prefix.prepared());
        } catch (Throwable failure) {
            Throwable cleanup = close(prefix.prepared());
            if (cleanup != null && cleanup != failure) {
                failure.addSuppressed(cleanup);
            }
            throw propagate(failure);
        }
        DeliveryContext.SelectionCommit<R> selection;
        try {
            selection = context.commitSelection(
                    new RouteCommit(group),
                    prefix.boundary(),
                    metadata.reason());
        } catch (Throwable failure) {
            Throwable cleanup = group.resolveCommitFailure(failure);
            if (cleanup != null && cleanup != failure) {
                failure.addSuppressed(cleanup);
            }
            throw propagate(failure);
        }
        if (selection
                instanceof DeliveryContext.SelectionCommit.NotCommitted<?>) {
            Throwable cleanup = group.releaseUncommitted();
            if (cleanup != null) {
                throw propagate(cleanup);
            }
            return selection.loopResult();
        }
        DeliveryContext.SelectionCommit.Committed<R> committed =
                (DeliveryContext.SelectionCommit.Committed<R>) selection;
        context.publishCommittedDelivery(
                committed.owner(), metadata);
        return committed.loopResult();
    }

    private Prefix reservePrefix(
            List<DeliveryItem> candidates,
            PrefillTimePredictor.Evaluator evaluator) {
        DeliveryItem head = candidates.get(0);
        CapacityBoundary.Attempt<PrefillAdmissionPort.PreparedAdmission>
                headAttempt = slotPort.prepareIfOwned(
                        head,
                        () -> admissionPort.prepare(
                                head, predict(evaluator, head)))
                        .orElseGet(() -> RouteDeliveryStrategy
                                .<PrefillAdmissionPort.PreparedAdmission>
                                ownershipLost());
        if (headAttempt
                instanceof CapacityBoundary.Attempt.Rejected<
                        PrefillAdmissionPort.PreparedAdmission> rejected) {
            return new Prefix(
                    List.of(),
                    null,
                    new DeliveryContext.SelectionBoundary(
                            head, rejected.boundary()));
        }
        PrefillAdmissionPort.PreparedAdmission prepared =
                ((CapacityBoundary.Attempt.Accepted<
                        PrefillAdmissionPort.PreparedAdmission>)
                        headAttempt).value();
        if (prepared.correlationId().isPresent()) {
            IllegalStateException failure = new IllegalStateException(
                    "route admission returned an external correlation id");
            Throwable cleanup = close(prepared);
            if (cleanup != null && cleanup != failure) {
                failure.addSuppressed(cleanup);
            }
            throw failure;
        }
        List<DeliveryItem> admitted = new ArrayList<>(candidates.size());
        admitted.add(head);
        try {
            for (int index = 1; index < candidates.size(); index++) {
                DeliveryItem item = candidates.get(index);
                CapacityBoundary.Attempt<DeliveryItem> attempt =
                        slotPort.prepareIfOwned(
                                item,
                                () -> prepared.append(
                                        item, predict(evaluator, item)))
                                .orElseGet(() -> RouteDeliveryStrategy
                                        .<DeliveryItem>ownershipLost());
                if (attempt
                        instanceof CapacityBoundary.Attempt.Accepted<
                                DeliveryItem>) {
                    admitted.add(item);
                    continue;
                }
                return new Prefix(
                        admitted,
                        prepared,
                        new DeliveryContext.SelectionBoundary(
                                item,
                                ((CapacityBoundary.Attempt.Rejected<DeliveryItem>)
                                        attempt).boundary()));
            }
            return new Prefix(admitted, prepared, null);
        } catch (Throwable failure) {
            Throwable cleanup = close(prepared);
            if (cleanup != null && cleanup != failure) {
                failure.addSuppressed(cleanup);
            }
            throw propagate(failure);
        }
    }

    private void deliver(
            List<DeliveryItem> items,
            PrefillAdmissionPort.CommittedAdmission admission,
            DeliveryMetadata metadata) {
        Throwable deliveryFailure = null;
        List<DeliveryItem> delivered = new ArrayList<>(items.size());
        try {
            for (DeliveryItem item : items) {
                SlotDeliveryPort.Claim claim;
                try {
                    claim = slotPort.tryCommit(
                            item,
                            SlotDeliveryPort.Identity.commitConfirmation(),
                            pointOfNoReturn -> admission.transfer(
                                    item, pointOfNoReturn));
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
                            SlotDeliveryPort.Completion.Delivered.INSTANCE);
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
            List<DeliveryItem> items,
            PrefillTimePredictor.Evaluator evaluator) {
        double totalMs = 0.0;
        for (DeliveryItem item : items) {
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
            DeliveryItem item) {
        return PrefillPredictionBoundary.predictSingleRequestMs(
                evaluator, item.seqLen(), item.hitCache());
    }

    private static <T> CapacityBoundary.Attempt<T> ownershipLost() {
        return new CapacityBoundary.Attempt.Rejected<>(
                CapacityBoundary.OwnershipLost.INSTANCE);
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

    /** Invocation-local adapter for this strategy's canonical queue commit. */
    private record RouteCommit(AdmittedGroup group)
            implements DeliveryContext.CanonicalCommit {

        @Override
        public List<DeliveryItem> items() {
            return group.items();
        }

        @Override
        public AdmittedGroup commitUnderLock() {
            return group.commitUnderLockInternal();
        }
    }

    /** Ordered route callback payload and sole owner of its exact members. */
    public static final class AdmittedGroup implements CommittedDelivery {
        private enum State {
            PREPARED,
            COMMITTED,
            CLOSED
        }

        private final RouteDeliveryStrategy owner;
        private final List<DeliveryItem> items;
        private PrefillAdmissionPort.PreparedAdmission prepared;
        private PrefillAdmissionPort.CommittedAdmission committed;
        private State state = State.PREPARED;

        private AdmittedGroup(
                RouteDeliveryStrategy owner,
                List<DeliveryItem> items,
                PrefillAdmissionPort.PreparedAdmission prepared) {
            this.owner = owner;
            this.items = items;
            this.prepared = prepared;
            assert !this.items.isEmpty()
                    : "admitted route group requires members";
        }

        @Override
        public List<DeliveryItem> items() {
            return items;
        }

        synchronized AdmittedGroup commitUnderLockInternal() {
            if (state != State.PREPARED) {
                throw new IllegalStateException(
                        "route group cannot commit in " + state);
            }
            committed = prepared.commitUnderLock(items, 0L);
            prepared = null;
            state = State.COMMITTED;
            return this;
        }

        @Override
        public void deliver(DeliveryMetadata metadata) {
            PrefillAdmissionPort.CommittedAdmission exactCommitted =
                    takeCommitted();
            owner.deliver(items, exactCommitted, metadata);
        }

        @Override
        public void fail(Throwable cause) {
            PrefillAdmissionPort.CommittedAdmission exactCommitted;
            synchronized (this) {
                if (state == State.CLOSED) {
                    return;
                }
                if (state != State.COMMITTED || committed == null) {
                    throw new IllegalStateException(
                            "route group cannot fail in " + state);
                }
                exactCommitted = committed;
                committed = null;
                state = State.CLOSED;
            }
            Throwable cleanup = close(exactCommitted);
            if (cleanup != null) {
                throw propagate(cleanup);
            }
        }

        private synchronized Throwable releaseUncommitted() {
            if (state != State.PREPARED) {
                return new IllegalStateException(
                        "route group cannot roll back in " + state);
            }
            PrefillAdmissionPort.PreparedAdmission exactPrepared = prepared;
            prepared = null;
            state = State.CLOSED;
            return close(exactPrepared);
        }

        private Throwable resolveCommitFailure(Throwable cause) {
            synchronized (this) {
                if (state == State.CLOSED) {
                    return null;
                }
                if (state == State.PREPARED) {
                    return releaseUncommitted();
                }
            }
            try {
                fail(cause);
                return null;
            } catch (Throwable cleanupFailure) {
                return cleanupFailure;
            }
        }

        private synchronized PrefillAdmissionPort.CommittedAdmission
                takeCommitted() {
            if (state != State.COMMITTED || committed == null) {
                throw new IllegalStateException(
                        "route group cannot deliver in " + state);
            }
            PrefillAdmissionPort.CommittedAdmission exactCommitted = committed;
            committed = null;
            state = State.CLOSED;
            return exactCommitted;
        }
    }

    private record Prefix(
            List<DeliveryItem> items,
            PrefillAdmissionPort.PreparedAdmission prepared,
            DeliveryContext.SelectionBoundary boundary) {
        private Prefix {
            assert items.isEmpty() == (prepared == null)
                    : "route prefix must own preparation exactly when non-empty";
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
