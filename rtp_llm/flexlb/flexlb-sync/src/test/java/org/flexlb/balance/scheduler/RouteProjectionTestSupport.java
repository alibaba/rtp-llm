package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.delivery.DeliveryMetrics;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.projection.QueueSnapshot;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.balance.projection.WorkSnapshot;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

import static org.mockito.Mockito.mock;

/** Frozen-value builders shared by canonical route-projection tests. */
final class RouteProjectionTestSupport {

    static final long NOW_MS = 10_000L;

    static final Comparator<GroupPlanner.Item> FIFO =
            Comparator.comparingLong(GroupPlanner.Item::enqueueSeq)
                    .thenComparingLong(GroupPlanner.Item::requestId);
    static final Comparator<GroupPlanner.Item> PRIORITY =
            Comparator.comparingInt(GroupPlanner.Item::priority)
                    .reversed()
                    .thenComparingLong(GroupPlanner.Item::enqueueSeq)
                    .thenComparingLong(GroupPlanner.Item::requestId);

    static final RouteProjection.DeliveryProjection ROUTE =
            new RouteDeliveryStrategy(
                    mock(RequestRegistry.class),
                    mock(DeliveryMetrics.class))
                    .projectionPolicy();
    static final RouteProjection.DeliveryProjection BATCH =
            new BatchDeliveryStrategy(
                    () -> CapacityBoundary.Attempt.rejected(
                            CapacityBoundary.OWNERSHIP_LOST),
                    () -> 1L,
                    mock(RequestRegistry.class),
                    mock(DeliveryMetrics.class))
                    .projectionPolicy();

    static final PrefillTimePredictor.Evaluator TOKEN_EVALUATOR =
            new PrefillTimePredictor.Evaluator() {
                @Override
                public long estimateMs(long totalTokens, long hitTokens) {
                    long sequence = Math.max(0L, totalTokens);
                    long hit = Math.max(0L, Math.min(hitTokens, sequence));
                    return (long) (sequence - hit + 0.3 * hit);
                }

                @Override
                public double predictBatchMs(
                        org.flexlb.balance.prediction.PrefillBatchFeatures
                                features) {
                    return features.items().stream()
                            .mapToLong(item -> estimateMs(
                                    item.seqLen(), item.hitCache()))
                            .sum();
                }
            };

    private RouteProjectionTestSupport() {
    }

    static GroupPlanner.Constraints constraints(
            int maxRequests,
            long collectionWindowMs) {
        return new GroupPlanner.Constraints(
                maxRequests,
                1_000_000L,
                1_000_000L,
                0L,
                collectionWindowMs);
    }

    static QueueSnapshot queue(
            boolean priority,
            GroupPlanner.Constraints constraints,
            List<GroupPlanner.Item> activeItems) {
        return queue(true, priority, constraints, activeItems, null);
    }

    static QueueSnapshot queue(
            boolean queueScheduling,
            boolean priority,
            GroupPlanner.Constraints constraints,
            List<GroupPlanner.Item> activeItems,
            QueueSnapshot.AdmissionBlock admissionBlock) {
        Comparator<GroupPlanner.Item> ordering = priority ? PRIORITY : FIFO;
        List<GroupPlanner.Item> ordered = new ArrayList<>(activeItems);
        ordered.sort(ordering);
        return new QueueSnapshot(
                NOW_MS,
                queueScheduling,
                ordering,
                constraints,
                ordered,
                admissionBlock);
    }

    static GroupPlanner.Item item(
            long requestId,
            int priority,
            long enqueueSequence,
            long serviceTokens) {
        return item(requestId, priority, enqueueSequence,
                serviceTokens, Long.MAX_VALUE);
    }

    static GroupPlanner.Item item(
            long requestId,
            int priority,
            long enqueueSequence,
            long serviceTokens,
            long expiresAtMs) {
        return new GroupPlanner.Item(
                requestId,
                priority,
                enqueueSequence,
                NOW_MS - 1L,
                expiresAtMs,
                serviceTokens,
                0L);
    }

    static RouteProjection.Probe probe(
            long requestId,
            int priority,
            long sequenceLength,
            long hitCache,
            RouteProjection.Demand demand) {
        return probe(requestId, priority, NOW_MS, Long.MAX_VALUE,
                sequenceLength, hitCache, demand);
    }

    static RouteProjection.Probe probe(
            long requestId,
            int priority,
            long enqueuedAtMs,
            long expiresAtMs,
            long sequenceLength,
            long hitCache,
            RouteProjection.Demand demand) {
        return new RouteProjection.Probe(
                requestId,
                priority,
                enqueuedAtMs,
                expiresAtMs,
                sequenceLength,
                hitCache,
                hitCache,
                demand);
    }

    static WorkSnapshot noCommittedWork() {
        return work(List.of(), List.of(), 0L);
    }

    static WorkSnapshot work(
            List<WorkSnapshot.RequestWork> requests,
            List<WorkSnapshot.BatchWork> batches,
            long unknownRequestCount) {
        return new WorkSnapshot(
                NOW_MS, requests, batches, unknownRequestCount);
    }

    static RouteProjection.Candidate project(
            QueueSnapshot queue,
            WorkSnapshot work,
            PrefillTimePredictor.Evaluator evaluator,
            RouteProjection.Probe probe,
            RouteProjection.DeliveryProjection deliveryProjection) {
        return candidate(
                queue, work, evaluator, probe, deliveryProjection, 0L);
    }

    static RouteProjection.Candidate candidate(
            QueueSnapshot queue,
            WorkSnapshot work,
            PrefillTimePredictor.Evaluator evaluator,
            RouteProjection.Probe probe,
            RouteProjection.DeliveryProjection deliveryProjection,
            long pendingRequestCount) {
        return RouteProjection.project(
                new RouteProjection.Inputs(
                        queue, work, pendingRequestCount),
                probe,
                evaluator,
                deliveryProjection);
    }
}
