package org.flexlb.balance.projection;

import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.prediction.InvalidPrefillPredictionException;
import org.flexlb.balance.prediction.PrefillBatchFeatures;
import org.flexlb.balance.prediction.PrefillPredictionBoundary;
import org.flexlb.balance.prediction.PrefillTimePredictor;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.OptionalLong;
import java.util.PriorityQueue;

/** Pure frozen-snapshot TTFT projection shared by endpoint selection policies. */
final class RouteTimelineProjector {

    private static final String INVALID_PREDICTION_DETAIL =
            "PREDICTOR_RETURNED_INVALID_VALUE";
    private final RouteProjection.Probe incoming;
    private final long pendingRequestCount;

    RouteTimelineProjector(
            RouteProjection.Probe incoming,
            long pendingRequestCount) {
        this.incoming = incoming;
        this.pendingRequestCount = pendingRequestCount;
    }

    /**
     * Project one endpoint under a frozen serial-work model.
     *
     * <p>Already committed work forms the engine cursor. Collection deadlines
     * and that cursor overlap via {@code max(cursor, readyAt)}. The snapshot's
     * frozen delivery projection defines each group's completion shape.
     */
    RouteProjection.Candidate project(
            QueueSnapshot queue,
            WorkSnapshot committed,
            PrefillTimePredictor.Evaluator evaluator,
            RouteProjection.DeliveryProjection deliveryProjection) {
        long projectionAtMs = queue.capturedAtMs();
        if (evaluator == null) {
            return unavailable("PREDICTOR_MISSING");
        }
        if (incoming.expiresAtMs() <= 0L
                || projectionAtMs >= incoming.expiresAtMs()) {
            return unavailable("INCOMING_EXPIRED");
        }

        PredictionBoundary predictions = new PredictionBoundary(evaluator);
        try {
            return projectWithPredictions(
                    queue, committed, deliveryProjection,
                    projectionAtMs, predictions);
        } catch (PredictionFailure predictionFailure) {
            return unavailable(predictionFailure.detail("PREDICTION_FAILED"));
        }
    }

    private RouteProjection.Candidate projectWithPredictions(
            QueueSnapshot queue,
            WorkSnapshot committed,
            RouteProjection.DeliveryProjection deliveryProjection,
            long projectionAtMs,
            PredictionBoundary predictions) {
        if (containsCommittedRequest(committed, incoming.requestId())) {
            return unavailable("INCOMING_ALREADY_COMMITTED");
        }

        final long incomingPrefillMs;
        try {
            incomingPrefillMs = predictions.singleMs(
                    incoming.seqLen(), incoming.hitCache());
        } catch (PredictionFailure predictionFailure) {
            return unavailable(
                    predictionFailure.detail("SINGLE_PREDICTION_FAILED"));
        }

        if (committed.hasUnknownWork()) {
            if (containsActiveRequest(queue, incoming.requestId())) {
                return unavailable("INCOMING_ALREADY_ACTIVE");
            }
            if (queue.queueScheduling()) {
                GroupPlanner.Shape incomingShape =
                        GroupPlanner.Shape.empty().add(incoming.seqLen());
                if (!incomingShape.fitsKv(queue.constraints().batchKvCapacity())) {
                    return blocked(
                            incomingPrefillMs,
                            RouteProjection.Candidate.InitialHeadDisposition.NONE,
                            "PREFILL_KV_CAPACITY");
                }
            }
            return unmodeledEngineWork(incomingPrefillMs);
        }

        long committedMs = knownRemainingWorkMsAt(committed, projectionAtMs);

        if (!queue.queueScheduling()) {
            long completionMs = saturatedAdd(committedMs, incomingPrefillMs);
            return candidate(
                    RouteProjection.Candidate.State.MODELED,
                    OptionalLong.of(completionMs),
                    incoming.demand().drainRequired()
                            ? OptionalLong.of(completionMs)
                            : OptionalLong.empty(),
                    incomingPrefillMs,
                    RouteProjection.Candidate.InitialHeadDisposition.NONE,
                    "SERIAL_FROZEN_DIRECT");
        }
        GroupPlanner.Item probe = incoming.asItem();
        GroupPlanner.Item initialActiveHead =
                queue.activeItems().isEmpty() ? null : queue.activeItems().getFirst();
        RouteProjection.Candidate.InitialHeadDisposition initialHeadDisposition =
                initialActiveHead == null
                        ? RouteProjection.Candidate.InitialHeadDisposition.NONE
                        : null;
        List<GroupPlanner.Item> eligibleActive = new ArrayList<>(
                queue.activeItems().size());
        for (GroupPlanner.Item item : queue.activeItems()) {
            // Expired work is a terminal queue mutation, not work ahead of a
            // new request in this snapshot projection.
            if (expiredAt(item, projectionAtMs)) {
                if (item == initialActiveHead) {
                    initialHeadDisposition =
                            RouteProjection.Candidate.InitialHeadDisposition.TERMINAL_PRUNED;
                }
                continue;
            }
            if (item.requestId() == incoming.requestId()) {
                return unavailable("INCOMING_ALREADY_ACTIVE");
            }
            eligibleActive.add(item);
        }
        ProjectedQueue ordered = ProjectedQueue.create(
                eligibleActive,
                probe,
                initialActiveHead,
                queue.ordering());

        if (initialHeadDisposition == null) {
            initialHeadDisposition = ordered.initialHeadDisposition();
        }
        long cursorMs = committedMs;
        long probeCompletionMs = -1L;
        boolean probeSeen = false;
        boolean drainKnown = true;
        long decisionNowMs = projectionAtMs;
        String drainDetail = "SERIAL_FROZEN_QUEUE";

        while (!ordered.isEmpty()) {
            ExpirationPrune expiration = ordered.pruneExpired(decisionNowMs);
            if (expiration.initialHeadExpired()) {
                initialHeadDisposition =
                        RouteProjection.Candidate.InitialHeadDisposition.TERMINAL_PRUNED;
            }
            if (expiration.probeExpired()) {
                return unavailable("INCOMING_EXPIRED_BEFORE_DISPATCH");
            }
            if (ordered.isEmpty()) {
                break;
            }

            GroupPlanner.Item head = ordered.head();
            GroupPlanner.Shape headShape =
                    GroupPlanner.Shape.empty().add(head.seqLen());
            if (!headShape.fitsKv(queue.constraints().batchKvCapacity())) {
                if (probeSeen) {
                    drainDetail = "DRAIN_BLOCKED_PREFILL_KV_CAPACITY";
                    drainKnown = false;
                    break;
                }
                return blocked(
                        incomingPrefillMs,
                        initialHeadDisposition,
                        "PREFILL_KV_CAPACITY");
            }

            final GroupPlanner.Plan<GroupPlanner.Item> plan;
            try {
                RouteProjection.GroupPlanning planning =
                        deliveryProjection.planning(predictions);
                plan = GroupPlanner.plan(
                        ordered.items(),
                        GroupPlanner.itemAccess(),
                        queue.constraints(),
                        decisionNowMs,
                        items -> {
                            int projectedProbeIndex =
                                    identityIndexOf(items, probe);
                            int requiredThroughIndex = projectedProbeIndex >= 0
                                    ? projectedProbeIndex
                                    : items.size() - 1;
                            return planning.durationMs(
                                    items, requiredThroughIndex);
                        });
            } catch (PredictionFailure predictionFailure) {
                if (probeSeen) {
                    drainDetail = "DRAIN_PREDICTION_UNAVAILABLE";
                    drainKnown = false;
                    break;
                }
                return unavailable(
                        predictionFailure.detail("BATCH_PREDICTION_FAILED"));
            }
            if (plan.items().isEmpty()) {
                throw new IllegalStateException(
                        "non-empty projected queue produced an empty group");
            }

            if (!plan.ready()) {
                // The production worker wakes for the collection deadline or
                // the current head's absolute expiry, then validates every
                // candidate again. Advancing this explicit scheduling clock
                // and replanning reproduces that terminal cleanup without
                // pretending expired work consumes engine service.
                decisionNowMs = Math.min(
                        plan.collectionDeadlineMs(), head.expiresAtMs());
                continue;
            }

            long readyInMs = elapsedFromNow(projectionAtMs, decisionNowMs);
            long startMs = Math.max(cursorMs, readyInMs);

            int probeIndex = identityIndexOf(plan.items(), probe);
            boolean containsProbe = probeIndex >= 0;
            try {
                RouteProjection.GroupService service =
                        deliveryProjection.service(plan, predictions);
                if (containsProbe) {
                    probeCompletionMs = saturatedAdd(
                            startMs, service.completionOffsetMs(probeIndex));
                    // A suffix prediction failure affects drain only after
                    // this exact completion offset has been established.
                    probeSeen = true;
                }
                if (!containsProbe || incoming.demand().drainRequired()) {
                    cursorMs = saturatedAdd(
                            startMs, service.totalDurationMs());
                } else {
                    cursorMs = probeCompletionMs;
                }
            } catch (PredictionFailure predictionFailure) {
                if (probeSeen) {
                    drainDetail = "DRAIN_PREDICTION_UNAVAILABLE";
                    drainKnown = false;
                    break;
                }
                return unavailable(
                        predictionFailure.detail("SERVICE_PREDICTION_FAILED"));
            }

            if (containsProbe) {
                probeSeen = true;
            }
            ordered.removePlannedPrefix(plan.items().size());

            if (probeSeen && !incoming.demand().drainRequired()) {
                return modeledTtftOnly(
                        probeCompletionMs, incomingPrefillMs,
                        initialHeadDisposition);
            }
        }

        if (probeCompletionMs < 0L) {
            throw new IllegalStateException(
                    "projected queue exhausted before planning the probe");
        }
        return candidate(
                RouteProjection.Candidate.State.MODELED,
                OptionalLong.of(probeCompletionMs),
                drainKnown ? OptionalLong.of(cursorMs) : OptionalLong.empty(),
                incomingPrefillMs,
                initialHeadDisposition,
                drainDetail);
    }

    private RouteProjection.Candidate modeledTtftOnly(
            long probeCompletionMs,
            long incomingPrefillMs,
            RouteProjection.Candidate.InitialHeadDisposition initialHeadDisposition) {
        return candidate(
                RouteProjection.Candidate.State.MODELED,
                OptionalLong.of(probeCompletionMs),
                OptionalLong.empty(),
                incomingPrefillMs,
                initialHeadDisposition,
                "SERIAL_FROZEN_QUEUE_TTFT_ONLY");
    }

    private static boolean containsCommittedRequest(
            WorkSnapshot committed, long requestId) {
        for (WorkSnapshot.RequestWork request : committed.requests()) {
            if (request.requestId() == requestId) {
                return true;
            }
        }
        for (WorkSnapshot.BatchWork batch : committed.batches()) {
            if (batch.requestIds().contains(requestId)) {
                return true;
            }
        }
        return false;
    }

    private static boolean containsActiveRequest(
            QueueSnapshot queue, long requestId) {
        for (GroupPlanner.Item item : queue.activeItems()) {
            if (item.requestId() == requestId) {
                return true;
            }
        }
        return false;
    }

    /** Remaining committed work normalized to the projection's common clock. */
    private static long knownRemainingWorkMsAt(
            WorkSnapshot committed, long projectionAtMs) {
        long elapsedMs = elapsedFromNow(
                committed.capturedAtMs(), projectionAtMs);
        long totalMs = 0L;
        for (WorkSnapshot.RequestWork request : committed.requests()) {
            totalMs = saturatedAdd(totalMs, remainingAt(request.phase(),
                    request.remainingWorkMs(), elapsedMs));
        }
        for (WorkSnapshot.BatchWork batch : committed.batches()) {
            totalMs = saturatedAdd(totalMs, remainingAt(batch.phase(),
                    batch.remainingWorkMs().orElseThrow(), elapsedMs));
        }
        return totalMs;
    }

    private static long remainingAt(
            WorkSnapshot.Phase phase, long remainingMs, long elapsedMs) {
        if (phase != WorkSnapshot.Phase.ENGINE_RUNNING) {
            return remainingMs;
        }
        return elapsedMs >= remainingMs ? 0L : remainingMs - elapsedMs;
    }

    private static boolean expiredAt(
            GroupPlanner.Item item, long nowMs) {
        return item.expiresAtMs() <= 0L || nowMs >= item.expiresAtMs();
    }

    private record ExpirationPrune(
            boolean probeExpired,
            boolean initialHeadExpired) {
    }

    private static int identityIndexOf(
            List<GroupPlanner.Item> items,
            GroupPlanner.Item target) {
        for (int i = 0; i < items.size(); i++) {
            if (items.get(i) == target) {
                return i;
            }
        }
        return -1;
    }

    /**
     * Mutable projection-only view over one already ordered queue snapshot.
     *
     * <p>{@link LinkedHashSet} preserves scheduling order while supporting
     * constant-time arbitrary expiry and prefix removal. The expiry heap lets a
     * monotonically advancing decision clock visit each deadline once instead
     * of rescanning every remaining item for every decision group.
     */
    private static final class ProjectedQueue {

        private static final Comparator<ProjectionNode> EXPIRY_ORDER =
                Comparator.comparingLong(
                                (ProjectionNode node) -> node.item.expiresAtMs())
                        .thenComparingInt(node -> node.ordinal);

        private final LinkedHashSet<ProjectionNode> live;
        private final PriorityQueue<ProjectionNode> expiry;
        private final ProjectionNode probeNode;
        private final ProjectionNode initialHeadNode;

        private ProjectedQueue(
                List<ProjectionNode> orderedNodes,
                ProjectionNode probeNode,
                ProjectionNode initialHeadNode) {
            this.live = new LinkedHashSet<>(orderedNodes);
            this.expiry = new PriorityQueue<>(EXPIRY_ORDER);
            this.expiry.addAll(orderedNodes);
            this.probeNode = probeNode;
            this.initialHeadNode = initialHeadNode;
        }

        private static ProjectedQueue create(
                List<GroupPlanner.Item> eligibleActive,
                GroupPlanner.Item probe,
                GroupPlanner.Item initialActiveHead,
                Comparator<GroupPlanner.Item> order) {
            List<ProjectionNode> orderedNodes = new ArrayList<>(
                    eligibleActive.size() + 1);
            ProjectionNode probeNode = null;
            ProjectionNode initialHeadNode = null;
            int ordinal = 0;
            for (GroupPlanner.Item item : eligibleActive) {
                if (probeNode == null && order.compare(probe, item) < 0) {
                    probeNode = new ProjectionNode(probe, ordinal++);
                    orderedNodes.add(probeNode);
                }
                ProjectionNode node = new ProjectionNode(item, ordinal++);
                orderedNodes.add(node);
                if (item == initialActiveHead) {
                    initialHeadNode = node;
                }
            }
            if (probeNode == null) {
                probeNode = new ProjectionNode(probe, ordinal);
                orderedNodes.add(probeNode);
            }
            return new ProjectedQueue(
                    orderedNodes, probeNode, initialHeadNode);
        }

        private boolean isEmpty() {
            return live.isEmpty();
        }

        private RouteProjection.Candidate.InitialHeadDisposition
                initialHeadDisposition() {
            if (initialHeadNode == null) {
                return RouteProjection.Candidate.InitialHeadDisposition.TERMINAL_PRUNED;
            }
            return initialHeadNode.ordinal < probeNode.ordinal
                    ? RouteProjection.Candidate.InitialHeadDisposition.BEFORE_PROBE
                    : RouteProjection.Candidate.InitialHeadDisposition.AFTER_PROBE;
        }

        private GroupPlanner.Item head() {
            Iterator<ProjectionNode> iterator = live.iterator();
            if (!iterator.hasNext()) {
                throw new IllegalStateException("projected queue is empty");
            }
            return iterator.next().item;
        }

        /**
         * Lazy ordered view consumed directly by the shared planner. The
         * planner stops at the first mode/capacity/prediction boundary, so a
         * short feasible prefix never causes a long suffix to be copied and
         * rescanned.
         */
        private Iterable<GroupPlanner.Item> items() {
            return () -> {
                Iterator<ProjectionNode> nodes = live.iterator();
                return new Iterator<>() {
                    @Override
                    public boolean hasNext() {
                        return nodes.hasNext();
                    }

                    @Override
                    public GroupPlanner.Item next() {
                        return nodes.next().item;
                    }
                };
            };
        }

        private void removePlannedPrefix(int count) {
            if (count <= 0) {
                throw new IllegalArgumentException(
                        "planned prefix must contain at least one item");
            }
            Iterator<ProjectionNode> iterator = live.iterator();
            for (int removed = 0; removed < count; removed++) {
                if (!iterator.hasNext()) {
                    throw new IllegalStateException(
                            "planner selected beyond the projected queue prefix");
                }
                iterator.next();
                iterator.remove();
            }
        }

        private ExpirationPrune pruneExpired(long nowMs) {
            boolean probeExpired = false;
            boolean initialHeadExpired = false;
            while (!expiry.isEmpty()) {
                ProjectionNode next = expiry.peek();
                if (!expiredAt(next.item, nowMs)) {
                    break;
                }
                expiry.remove();
                if (!live.remove(next)) {
                    continue;
                }
                if (next == probeNode) {
                    probeExpired = true;
                }
                if (next == initialHeadNode) {
                    initialHeadExpired = true;
                }
            }
            return new ExpirationPrune(probeExpired, initialHeadExpired);
        }
    }

    /** Identity node: default Object equality is required by the live set. */
    private static final class ProjectionNode {
        private final GroupPlanner.Item item;
        private final int ordinal;

        private ProjectionNode(
                GroupPlanner.Item item,
                int ordinal) {
            this.item = item;
            this.ordinal = ordinal;
        }
    }

    private static long elapsedFromNow(long nowMs, long deadlineMs) {
        return deadlineMs <= nowMs ? 0L : deadlineMs - nowMs;
    }

    private RouteProjection.Candidate unavailable(String detail) {
        return candidate(
                RouteProjection.Candidate.State.UNAVAILABLE,
                OptionalLong.empty(), OptionalLong.empty(), 0L,
                RouteProjection.Candidate.InitialHeadDisposition.NONE, detail);
    }

    private RouteProjection.Candidate unmodeledEngineWork(
            long incomingPrefillMs) {
        return candidate(
                RouteProjection.Candidate.State.UNMODELED_ENGINE_WORK,
                OptionalLong.empty(), OptionalLong.empty(), incomingPrefillMs,
                RouteProjection.Candidate.InitialHeadDisposition.NONE,
                "ENGINE_WORK_UNOBSERVABLE");
    }

    private RouteProjection.Candidate blocked(
            long incomingPrefillMs,
            RouteProjection.Candidate.InitialHeadDisposition initialHeadDisposition,
            String detail) {
        return candidate(
                RouteProjection.Candidate.State.BLOCKED,
                OptionalLong.empty(), OptionalLong.empty(), incomingPrefillMs,
                initialHeadDisposition, detail);
    }

    private RouteProjection.Candidate candidate(
            RouteProjection.Candidate.State state,
            OptionalLong projectedTtftMs,
            OptionalLong projectedDrainMs,
            long incomingPrefillMs,
            RouteProjection.Candidate.InitialHeadDisposition headDisposition,
            String detail) {
        boolean carriesPending = state == RouteProjection.Candidate.State.MODELED
                || state == RouteProjection.Candidate.State.UNMODELED_ENGINE_WORK;
        return new RouteProjection.Candidate(
                state, projectedTtftMs, projectedDrainMs, incomingPrefillMs,
                headDisposition, detail, null,
                incoming.hitCache(), incoming.routingCacheMatchTokens(),
                carriesPending
                        ? OptionalLong.of(pendingRequestCount)
                        : OptionalLong.empty());
    }

    private static long saturatedAdd(long left, long right) {
        return right > 0L && left > Long.MAX_VALUE - right
                ? Long.MAX_VALUE : left + right;
    }

    /**
     * The sole boundary between projection math and an external predictor.
     * Predictor failures remain distinguishable from invalid returned values,
     * while every numeric output is validated before it reaches scheduling.
     */
    private static final class PredictionBoundary
            implements RouteProjection.Predictions {

        private final PrefillTimePredictor.Evaluator evaluator;

        private PredictionBoundary(PrefillTimePredictor.Evaluator evaluator) {
            this.evaluator = evaluator;
        }

        private long singleMs(long seqLen, long hitCache) {
            try {
                return PrefillPredictionBoundary.predictSingleRequestMs(
                        evaluator, seqLen, hitCache);
            } catch (InvalidPrefillPredictionException invalidPrediction) {
                throw PredictionFailure.invalid(invalidPrediction);
            } catch (RuntimeException predictionFailure) {
                throw PredictionFailure.execution(predictionFailure);
            }
        }

        @Override
        public long itemDurationMs(GroupPlanner.Item item) {
            return singleMs(item.seqLen(), item.hitCache());
        }

        @Override
        public double batchPlanningDurationMs(
                List<GroupPlanner.Item> items) {
            try {
                return PrefillPredictionBoundary.predictDecisionGroupMs(
                        evaluator, batchFeatures(items));
            } catch (InvalidPrefillPredictionException invalidPrediction) {
                throw PredictionFailure.invalid(invalidPrediction);
            } catch (RuntimeException predictionFailure) {
                throw PredictionFailure.execution(predictionFailure);
            }
        }

        @Override
        public long batchDurationMs(
                List<GroupPlanner.Item> items) {
            try {
                return PrefillPredictionBoundary.predictCommittedBatchMs(
                        evaluator, batchFeatures(items));
            } catch (InvalidPrefillPredictionException invalidPrediction) {
                throw PredictionFailure.invalid(invalidPrediction);
            } catch (RuntimeException predictionFailure) {
                throw PredictionFailure.execution(predictionFailure);
            }
        }

        @Override
        public long committedGroupDurationMs(double predictedMs) {
            try {
                return PrefillPredictionBoundary.committedDecisionGroupMs(predictedMs);
            } catch (InvalidPrefillPredictionException invalidPrediction) {
                throw PredictionFailure.invalid(invalidPrediction);
            }
        }

        private static PrefillBatchFeatures batchFeatures(
                List<GroupPlanner.Item> items) {
            return PrefillBatchFeatures.from(
                    items,
                    GroupPlanner.Item::seqLen,
                    GroupPlanner.Item::hitCache);
        }
    }

    /**
     * Unified predictor failure signal. Call sites decide whether the failed
     * prediction is still needed for probe TTFT or only for suffix drain.
     */
    private static final class PredictionFailure extends RuntimeException {
        private final boolean invalidValue;

        private PredictionFailure(
                RuntimeException cause,
                boolean invalidValue) {
            super(cause);
            this.invalidValue = invalidValue;
        }

        private static PredictionFailure execution(RuntimeException cause) {
            return new PredictionFailure(cause, false);
        }

        private static PredictionFailure invalid(RuntimeException cause) {
            return new PredictionFailure(cause, true);
        }

        private String detail(String executionDetail) {
            return invalidValue ? INVALID_PREDICTION_DETAIL : executionDetail;
        }
    }
}
