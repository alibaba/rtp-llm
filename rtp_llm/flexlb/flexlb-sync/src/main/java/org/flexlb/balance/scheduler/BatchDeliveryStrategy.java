package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.scheduler.DefaultBatchDispatcher.PreparedSubmission;
import org.flexlb.balance.scheduler.DefaultBatchDispatcher.BatchSender;
import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.prediction.PrefillBatchFeatures;
import org.flexlb.balance.prediction.PrefillPredictionBoundary;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.balance.projection.WorkSnapshot;
import org.flexlb.balance.scheduler.RequestContext.DeliveryClaim;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.util.Failures;

import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.OptionalLong;
import java.util.function.BiConsumer;
import java.util.function.LongSupplier;
import java.util.function.Supplier;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;

/** EnqueueBatch admission, ownership, transport, telemetry, and projection. */
public final class BatchDeliveryStrategy implements DeliveryStrategy {

    private static final RouteProjection.DeliveryProjection PROJECTION =
            new BatchProjection();
    private final Supplier<CapacityBoundary.Attempt<PreparedSubmission>>
            prepareSubmission;
    private final LongSupplier batchIds;
    private final DeliveryMetricsReporter telemetry;

    public BatchDeliveryStrategy(
            Supplier<CapacityBoundary.Attempt<PreparedSubmission>>
                    prepareSubmission,
            LongSupplier batchIds,
            DeliveryMetricsReporter telemetry) {
        this.prepareSubmission = Objects.requireNonNull(
                prepareSubmission, "prepareSubmission");
        this.batchIds = Objects.requireNonNull(batchIds, "batchIds");
        this.telemetry = Objects.requireNonNull(telemetry, "telemetry");
    }

    @Override
    public DeliveryTransaction prepare(
            List<RequestRoute> candidates,
            PrefillTimePredictor.Evaluator evaluator,
            OptionalLong plannedPredictionMs) {
        checkArgument(!candidates.isEmpty(), "batch delivery requires at least one candidate");
        var transaction = DeliveryStrategy.prepareMembers(candidates, evaluator, prepareSubmission, batchIds);
        try {
            if (!transaction.items().isEmpty()) {
                transaction.batch.predictedMs = plannedPredictionMs.isPresent()
                        && transaction.items().size() == candidates.size()
                        ? plannedPredictionMs.getAsLong()
                        : PrefillPredictionBoundary.predictCommittedBatchMs(evaluator,
                                PrefillBatchFeatures.from(transaction.items(), RequestRoute::seqLen, RequestRoute::hitCache));
            }
            return transaction;
        } catch (Throwable failure) {
            throw Failures.propagate(Failures.append(failure, Failures.close(transaction)), "batch preparation failed");
        }
    }

    @Override
    public GroupPlanner.PrefixPrediction<RequestRoute> newGroupPredictor(
            PrefillTimePredictor.Evaluator evaluator) {
        PrefillTimePredictor.BatchPrediction prediction = evaluator.newBatchPrediction();
        return (added, items) -> PrefillPredictionBoundary.requireValidDecisionGroupMs(
                prediction.append(added.seqLen(), added.hitCache()));
    }

    @Override
    public RouteProjection.DeliveryProjection projectionPolicy() {
        return PROJECTION;
    }

    @Override
    public void deliver(DeliveryTransaction transaction, String decisionReason,
            int remainingQueueDepth, WorkSnapshot precedingWork, PrefillTimePredictor.Evaluator evaluator) {
        Objects.requireNonNull(precedingWork, "precedingWork");
        // Transfer before submit: the dispatcher may execute inline.
        synchronized (transaction) {
            var submission = transaction.transferToExecutor();
            try {
                submission.submit(sender -> {
                    transaction.requireSubmitted();
                    try {
                        deliverCommitted(transaction, decisionReason, remainingQueueDepth, precedingWork, evaluator, sender);
                    } catch (Throwable failure) {
                        Failures.run(failure, () -> DeliveryStrategy.failUnsentDelivery(transaction, failure, true));
                        throw Failures.propagate(failure, "batch delivery failed");
                    }
                });
            } catch (Throwable failure) {
                Failures.run(failure, () -> DeliveryStrategy.failUnsentDelivery(transaction, failure, true));
                throw Failures.propagate(failure, "batch delivery failed");
            }
        }
    }

    private void deliverCommitted(
            DeliveryTransaction transaction,
            String decisionReason,
            int remainingQueueDepth,
            WorkSnapshot precedingWork,
            PrefillTimePredictor.Evaluator evaluator,
            BatchSender sender) {
        var batch = transaction.batch;
        List<RequestRoute> original = transaction.items();
        List<DeliveryClaim> claimed = new ArrayList<>(original.size());
        List<RequestRoute> submitted = List.of();
        DispatchGate gate = null;
        Throwable handoffFailure = null;
        boolean senderAccepted = false;
        long deliveredPredictionMs = batch.predictedMs;
        try {
            for (var member : transaction.members) {
                RequestRoute item = member.route();
                try {
                    DeliveryClaim claim =
                            item.ctx().scheduler().claimDelivery(item, DeliveryClaimKind.BATCH_ENQUEUE,
                                    batch.batchId, member.decode());
                    if (claim != null) { claimed.add(claim); }
                } catch (Throwable claimFailure) {
                    item.ctx().scheduler().failDeliveryPreparation(item, claimFailure);
                }
            }

            if (!claimed.isEmpty()) {
                submitted = claimed.stream().map(claim -> claim.item).toList();
                if (submitted.size() != original.size()) {
                    deliveredPredictionMs =
                            PrefillPredictionBoundary.predictCommittedBatchMs(
                                    evaluator,
                                    PrefillBatchFeatures.from(
                                            submitted,
                                            RequestRoute::seqLen,
                                            RequestRoute::hitCache));
                }
                for (DeliveryClaim claim : claimed) {
                    claim.item.ctx().scheduler().setDeliveryPrediction(claim, precedingWork, deliveredPredictionMs);
                }
                gate = new DispatchGate(
                        claimed);
                checkArgument(remainingQueueDepth >= 0, "remainingQueueDepth must be non-negative");
                sender.sendBatch(submitted, batch.batchId, deliveredPredictionMs,
                        decisionReason, gate);
                senderAccepted = true;
            }
        } catch (Throwable failure) {
            handoffFailure = failure;
        } finally {
            handoffFailure = Failures.append(handoffFailure, transaction.finishDelivery());
            if (gate != null) {
                handoffFailure = Failures.run(handoffFailure, gate::open);
            }
        }

        if (handoffFailure != null) {
            if (senderAccepted) {
                throw Failures.propagate(handoffFailure, "batch delivery failed");
            } else {
                Throwable completionFailure = null;
                for (DeliveryClaim claim : claimed) {
                    try {
                        claim.item.ctx().scheduler().completeDelivery(claim, DeliveryResult.notSent(handoffFailure));
                    } catch (Throwable failure) {
                        completionFailure = Failures.append(completionFailure, failure);
                    }
                }
                if (completionFailure != null) {
                    throw Failures.propagate(Failures.append(
                            handoffFailure, completionFailure), "batch delivery failed");
                }
            }
        }

        if (senderAccepted) {
            telemetry.reportDelivery(
                    batch.batchId,
                    decisionReason,
                    remainingQueueDepth,
                    submitted,
                    deliveredPredictionMs);
        }
    }

    private static final class DispatchGate
            implements BiConsumer<RequestRoute, DeliveryResult> {
        private final Map<RequestRoute, DeliveryClaim>
                claimsByItem;
        private boolean deferred = true;
        private List<Event> events;

        private DispatchGate(
                List<DeliveryClaim> members) {

            this.claimsByItem = new IdentityHashMap<>(members.size());
            for (DeliveryClaim claim : members) {
                DeliveryClaim previous = claimsByItem.put(
                        claim.item, claim);
                checkArgument(previous == null, "duplicate batch delivery identity");
            }
        }

        private void open() {
            List<Event> pending;
            synchronized (this) {
                deferred = false;
                pending = events;
                events = null;
            }
            if (pending != null) {
                Throwable failure = null;
                for (Event event : pending) {
                    failure = Failures.run(failure,
                            () -> invoke(event.item(), event.completion()));
                }
                if (failure != null) {
                    throw Failures.propagate(failure, "batch delivery failed");
                }
            }
        }

        @Override
        public void accept(
                RequestRoute item,
                DeliveryResult completion) {
            synchronized (this) {
                if (deferred) {
                    if (events == null) {
                        events = new ArrayList<>();
                    }
                    events.add(new Event(item, completion));
                    return;
                }
            }
            invoke(item, completion);
        }

        private void invoke(RequestRoute item, DeliveryResult completion) {
            DeliveryClaim claim = claimsByItem.get(item);
            checkState(claim != null, "batch completion referenced an unsubmitted identity");
            claim.item.ctx().scheduler().completeDelivery(claim, completion);
        }
    }

    private record Event(
            RequestRoute item,
            DeliveryResult completion) {
    }

    private static final class BatchProjection
            implements RouteProjection.DeliveryProjection {

        @Override
        public long singletonCompletionOffsetMs(
                long seqLen,
                long hitCache,
                RouteProjection.Predictions predictions) {
            // Committed prediction performs the same validated decision-group
            // evaluation before converting to lifecycle milliseconds. Running
            // the planning call first only evaluates the predictor twice for
            // every endpoint in the full-fleet singleton fast path.
            return predictions.singletonBatchDurationMs(seqLen, hitCache);
        }

        @Override
        public RouteProjection.GroupPlanning planning(
                RouteProjection.Predictions predictions) {
            return new AppendPlanning(predictions.newBatchPrediction());
        }

        @Override
        public long completionOffsetMs(List<GroupPlanner.Item> items, int memberIndex,
                RouteProjection.Predictions predictions, RouteProjection.GroupPlanning planning) {
            Objects.checkIndex(memberIndex, items.size());
            if (planning != null) {
                var cached = planning.predictedPrefixMs(memberIndex + 1);
                if (cached.isPresent()) {
                    return PrefillPredictionBoundary.committedDecisionGroupMs(cached.getAsDouble());
                }
            }
            return predictions.batchDurationMs(items.subList(0, memberIndex + 1));
        }

        /** One planner invocation; prefixes only grow and the probe boundary never moves backwards. */
        private static final class AppendPlanning implements RouteProjection.GroupPlanning {
            private final PrefillTimePredictor.BatchPrediction prediction;
            private int size;
            private double latestPredictionMs;
            private double previousPredictionMs;

            private AppendPlanning(PrefillTimePredictor.BatchPrediction prediction) {
                this.prediction = prediction;
            }

            @Override
            public double durationMs(List<GroupPlanner.Item> prefix, int through) {
                checkArgument(through >= 0 && through < prefix.size() && through + 1 >= size,
                        "Prediction requires a growing prefix");
                while (size <= through) {
                    GroupPlanner.Item item = prefix.get(size);
                    double next = prediction.append(item.seqLen(), item.hitCache());
                    previousPredictionMs = latestPredictionMs;
                    latestPredictionMs = next;
                    size++;
                }
                return latestPredictionMs;
            }

            @Override
            public java.util.OptionalDouble predictedPrefixMs(int prefixSize) {
                // Selection uses the latest prefix, or the preceding one when
                // the last append exceeded the budget. Other callers recompute.
                if (prefixSize > 0 && prefixSize == size) {
                    return java.util.OptionalDouble.of(latestPredictionMs);
                }
                if (prefixSize > 0 && prefixSize == size - 1) {
                    return java.util.OptionalDouble.of(previousPredictionMs);
                }
                return java.util.OptionalDouble.empty();
            }
        }

    }
}
