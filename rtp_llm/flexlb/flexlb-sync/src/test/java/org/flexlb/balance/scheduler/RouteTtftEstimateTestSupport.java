package org.flexlb.balance.scheduler;

import org.flexlb.balance.estimate.BatchTtftEstimator;
import org.flexlb.balance.estimate.NonBatchTtftEstimator;
import org.flexlb.balance.estimate.PrefillTtftEstimate;
import org.flexlb.balance.estimate.QueueView;
import org.flexlb.balance.estimate.WorkView;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.prediction.PrefillTimePredictor;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/** Frozen-value builders shared by canonical route-estimate tests. */
final class RouteTtftEstimateTestSupport {

    static final long NOW_MS = 10_000L;

    static final Comparator<GroupPlanner.Item> FIFO =
            Comparator.comparingLong(GroupPlanner.Item::enqueueSeq)
                    .thenComparing(GroupPlanner.Item::requestId);
    static final Comparator<GroupPlanner.Item> PRIORITY =
            Comparator.comparingInt(GroupPlanner.Item::priority)
                    .reversed()
                    .thenComparingLong(GroupPlanner.Item::enqueueSeq)
                    .thenComparing(GroupPlanner.Item::requestId);

    static final PrefillTtftEstimate.TtftEstimationRule ROUTE =
            new NonBatchTtftEstimator().ttftEstimationRule();
    static final PrefillTtftEstimate.TtftEstimationRule BATCH =
            new BatchTtftEstimator().ttftEstimationRule();

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

    private RouteTtftEstimateTestSupport() {
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

    static QueueView queue(boolean priority,
                               GroupPlanner.Constraints constraints,
                               List<GroupPlanner.Item> activeItems) {
        return queue(priority, constraints, activeItems, null);
    }

    static QueueView queue(
            boolean priority,
            GroupPlanner.Constraints constraints,
            List<GroupPlanner.Item> activeItems,
            QueueView.AdmissionBlock admissionBlock) {
        Comparator<GroupPlanner.Item> ordering = priority ? PRIORITY : FIFO;
        List<GroupPlanner.Item> ordered = new ArrayList<>(activeItems);
        ordered.sort(ordering);
        return new QueueView(
                NOW_MS,
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
                Long.toString(requestId),
                priority,
                enqueueSequence,
                NOW_MS - 1L,
                expiresAtMs,
                serviceTokens,
                0L);
    }

    static PrefillTtftEstimate.Probe probe(
            long requestId,
            int priority,
            long sequenceLength,
            long hitCache) {
        return probe(requestId, priority, NOW_MS, Long.MAX_VALUE,
                sequenceLength, hitCache);
    }

    static PrefillTtftEstimate.Probe probe(
            long requestId,
            int priority,
            long enqueuedAtMs,
            long expiresAtMs,
            long sequenceLength,
            long hitCache) {
        return new PrefillTtftEstimate.Probe(
                Long.toString(requestId),
                priority,
                enqueuedAtMs,
                expiresAtMs,
                sequenceLength,
                hitCache,
                hitCache);
    }

    static WorkView noCommittedWork() {
        return work(List.of(), List.of(), 0L);
    }

    static WorkView work(
            List<WorkView.RequestWork> requests,
            List<WorkView.BatchWork> batches,
            long unknownRequestCount) {
        return new WorkView(
                NOW_MS, requests, batches, unknownRequestCount);
    }

    static PrefillTtftEstimate.Candidate estimate(
            QueueView queue,
            WorkView work,
            PrefillTimePredictor.Evaluator evaluator,
            PrefillTtftEstimate.Probe probe,
            PrefillTtftEstimate.TtftEstimationRule ttftEstimationRule) {
        return PrefillTtftEstimate.estimate(
                new PrefillTtftEstimate.LoadView(
                        queue, work),
                probe,
                evaluator,
                ttftEstimationRule);
    }
}
