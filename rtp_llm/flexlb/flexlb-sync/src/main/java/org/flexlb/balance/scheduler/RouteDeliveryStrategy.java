package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.prediction.PrefillPredictionBoundary;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.balance.projection.WorkSnapshot;
import org.flexlb.balance.scheduler.RequestContext.DeliveryClaim;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.util.Failures;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.OptionalLong;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.math.LongMath.saturatedAdd;

/** Individual route admission, ownership, publication, and projection. */
public final class RouteDeliveryStrategy implements DeliveryStrategy {

    private static final RouteProjection.DeliveryProjection PROJECTION =
            new RouteProjectionPolicy();
    private final DeliveryMetricsReporter telemetry;

    public RouteDeliveryStrategy(
            DeliveryMetricsReporter telemetry) {
        this.telemetry = Objects.requireNonNull(telemetry, "telemetry");
    }

    @Override
    public DeliveryTransaction prepare(
            List<RequestRoute> candidates,
            PrefillTimePredictor.Evaluator evaluator,
            OptionalLong plannedPredictionMs) {
        checkArgument(!candidates.isEmpty(), "route delivery requires at least one candidate");
        return DeliveryStrategy.prepareMembers(candidates, evaluator, null, null);
    }

    @Override
    public void deliver(
            DeliveryTransaction transaction, String decisionReason,
            int remainingQueueDepth,
            WorkSnapshot precedingWork, PrefillTimePredictor.Evaluator evaluator) {
        Objects.requireNonNull(precedingWork, "precedingWork");
        Throwable deliveryFailure = null;
        List<RequestRoute> delivered = new ArrayList<>(transaction.members.size());
        // Preserve committed member indices; claim the whole group before publishing any response.
        DeliveryClaim[] claims = new DeliveryClaim[transaction.members.size()];
        PrefillState.CommittedHandoff handoff = transaction.takeCommitted();
        try {
            for (int index = 0; index < transaction.members.size(); index++) {
                var member = transaction.members.get(index);
                RequestRoute item = member.route();
                DeliveryClaim claim;
                try {
                    claim = item.ctx().scheduler().claimDelivery(item, DeliveryClaimKind.ROUTE_DECISION, 0L,
                            member.decode());
                } catch (Throwable claimFailure) {
                    try {
                        item.ctx().scheduler().failDeliveryPreparation(item, claimFailure);
                    } catch (Throwable terminalFailure) {
                        claimFailure.addSuppressed(terminalFailure);
                        deliveryFailure = Failures.append(
                                deliveryFailure, claimFailure);
                    }
                    continue;
                }
                claims[index] = claim;
            }
            long unstartedWorkMs = 0L;
            for (int index = 0; index < claims.length; index++) {
                DeliveryClaim claim = claims[index];
                if (claim == null) {
                    continue;
                }
                RequestRoute item = claim.item;
                try {
                    long itemWorkMs = transaction.routePredictions[index];
                    unstartedWorkMs = saturatedAdd(unstartedWorkMs, itemWorkMs);
                    item.ctx().scheduler().publishRoute(claim, precedingWork, unstartedWorkMs);
                    delivered.add(item);
                } catch (Throwable completionFailure) {
                    deliveryFailure = Failures.append(
                            deliveryFailure, completionFailure);
                }
            }
        } finally {
            DeliveryTransaction.closeCommitted(transaction.members, handoff);
        }
        if (!delivered.isEmpty()) {
            telemetry.reportDelivery(0L, null, remainingQueueDepth, delivered, 0L);
        }
        if (deliveryFailure != null) {
            throw Failures.propagate(deliveryFailure, "route delivery failed");
        }
    }

    @Override
    public GroupPlanner.PrefixPrediction<RequestRoute> newGroupPredictor(
            PrefillTimePredictor.Evaluator evaluator) {
        return new GroupPlanner.PrefixPrediction<>() {
            private double totalMs;

            @Override
            public double append(RequestRoute added, List<RequestRoute> items) {
                totalMs += PrefillPredictionBoundary.predictSingleRequestMs(
                        evaluator, added.seqLen(), added.hitCache());
                return PrefillPredictionBoundary.requireValidDecisionGroupMs(totalMs);
            }
        };
    }

    @Override
    public RouteProjection.DeliveryProjection projectionPolicy() {
        return PROJECTION;
    }

    private static final class RouteProjectionPolicy
            implements RouteProjection.DeliveryProjection {

        private static final ThreadLocal<RouteCursor> PLANNING =
                ThreadLocal.withInitial(RouteCursor::new);

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
            RouteCursor planning = PLANNING.get();
            planning.reset(predictions);
            return planning;
        }

        @Override
        public long completionOffsetMs(List<GroupPlanner.Item> items, int memberIndex,
                RouteProjection.Predictions predictions, RouteProjection.GroupPlanning planning) {
            Objects.checkIndex(memberIndex, items.size());
            // GroupPlanning belongs to this invocation and its exact selected prefix.
            if (planning instanceof RouteCursor cursor && cursor.predictions == predictions) {
                long cached = cursor.cachedDurationMs(memberIndex);
                if (cached >= 0L) { return cached; }
            }
            long durationMs = 0L;
            for (int i = 0; i <= memberIndex; i++) {
                durationMs = saturatedAdd(durationMs, predictions.itemDurationMs(items.get(i)));
            }
            return durationMs;
        }

        private static final class RouteCursor
                implements RouteProjection.GroupPlanning {
            private long durationMs;
            private long previousDurationMs;
            private int computedThrough;
            private RouteProjection.Predictions predictions;

            private void reset(RouteProjection.Predictions exactPredictions) {
                predictions = exactPredictions;
                computedThrough = -1;
                durationMs = 0L;
                previousDurationMs = 0L;
            }

            @Override
            public double durationMs(List<GroupPlanner.Item> prefix, int requiredThroughIndex) {
                if (requiredThroughIndex < 0 || requiredThroughIndex >= prefix.size()) {
                    throw new IndexOutOfBoundsException(requiredThroughIndex);
                }
                checkArgument(requiredThroughIndex >= computedThrough, "planning index must not decrease");
                while (computedThrough < requiredThroughIndex) {
                    int next = computedThrough + 1;
                    long itemMs = predictions.itemDurationMs(prefix.get(next));
                    previousDurationMs = durationMs;
                    durationMs = saturatedAdd(
                            durationMs, itemMs);
                    computedThrough = next;
                }
                return durationMs;
            }

            private long cachedDurationMs(int memberIndex) {
                if (memberIndex == computedThrough) { return durationMs; }
                // The last tentative member may have exceeded the budget and
                // been removed from the selected group.
                if (memberIndex == computedThrough - 1) { return previousDurationMs; }
                return -1L;
            }

        }

    }
}
