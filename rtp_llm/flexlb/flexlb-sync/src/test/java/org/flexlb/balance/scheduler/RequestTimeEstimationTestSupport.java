package org.flexlb.balance.scheduler;

import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.projection.RequestTimeEstimation;
import org.flexlb.balance.projection.RequestTimeEstimation.EstimationSnapshot;
import org.flexlb.balance.projection.RequestTimeEstimation.WorkSnapshot;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

import static com.google.common.base.Preconditions.checkArgument;

/**
 * Frozen-value builders shared by request time estimation tests.
 */
public final class RequestTimeEstimationTestSupport {
    static final long NOW_MS = 10_000L;

    static final Comparator<GroupPlanner.Item> FIFO =
            Comparator.comparingLong(GroupPlanner.Item::enqueueSeq)
                    .thenComparing(GroupPlanner.Item::requestId);
    static final Comparator<GroupPlanner.Item> PRIORITY =
            Comparator.comparingInt(GroupPlanner.Item::priority)
                    .reversed()
                    .thenComparingLong(GroupPlanner.Item::enqueueSeq)
                    .thenComparing(GroupPlanner.Item::requestId);
    static final org.flexlb.config.DispatcherConfig.Type ROUTE =
            org.flexlb.config.DispatcherConfig.Type.NON_BATCH;
    static final org.flexlb.config.DispatcherConfig.Type BATCH =
            org.flexlb.config.DispatcherConfig.Type.BATCH;
    static final PrefillTimePredictor.Evaluator TOKEN_EVALUATOR =
            new PrefillTimePredictor.Evaluator() {
                @Override
                public double predictBatchMs(
                        org.flexlb.balance.prediction.PrefillTimePredictor.PrefillBatchFeatures
                                features) {
                    return features.items().stream()
                            .mapToLong(item -> estimateMs(item.seqLen(), item.hitCache()))
                            .sum();
                }

                @Override
                public long estimateMs(long totalTokens, long hitTokens) {
                    long sequence = Math.max(0L, totalTokens);
                    long hit = Math.max(0L, Math.min(hitTokens, sequence));
                    return (long) (sequence - hit + 0.3 * hit);
                }
            };

    private RequestTimeEstimationTestSupport() {}

    public static Candidate project(RequestTimeEstimation.Inputs inputs, Probe probe) {
        return project(inputs, probe, inputs.queue().capturedAtMs());
    }

    public static Candidate project(
            RequestTimeEstimation.Inputs inputs, Probe probe, long planningAtMs) {
        RequestTimeEstimation.CandidateView view =
                RequestTimeEstimation.current()
                        .projectCandidate(
                                inputs,
                                String.valueOf(probe.requestId()),
                                probe.priority(),
                                probe.enqueuedAtMs(),
                                probe.expiresAtMs(),
                                probe.seqLen(),
                                probe.hitCache(),
                                probe.routingCacheMatchTokens(),
                                planningAtMs);
        return immutable(view);
    }

    static GroupPlanner.Constraints constraints(int maxRequests, long collectionWindowMs) {
        return new GroupPlanner.Constraints(
                maxRequests, 1_000_000L, 1_000_000L, 0L, collectionWindowMs);
    }

    static EstimationSnapshot queue(
            boolean priority,
            GroupPlanner.Constraints constraints,
            List<GroupPlanner.Item> activeItems) {
        return queue(true, priority, constraints, activeItems);
    }

    static EstimationSnapshot queue(
            boolean queueScheduling,
            boolean priority,
            GroupPlanner.Constraints constraints,
            List<GroupPlanner.Item> activeItems) {
        Comparator<GroupPlanner.Item> ordering = priority ? PRIORITY : FIFO;
        List<GroupPlanner.Item> ordered = new ArrayList<>(activeItems);
        ordered.sort(ordering);
        return new EstimationSnapshot(
                NOW_MS,
                queueScheduling,
                org.flexlb.balance.planner.BuiltinGroupPlanner.FIXED_WINDOW,
                ordering,
                constraints,
                ordered);
    }

    static GroupPlanner.Item item(
            long requestId, int priority, long enqueueSequence, long serviceTokens) {
        return item(requestId, priority, enqueueSequence, serviceTokens, Long.MAX_VALUE);
    }

    static GroupPlanner.Item item(
            long requestId,
            int priority,
            long enqueueSequence,
            long serviceTokens,
            long expiresAtMs) {
        return new GroupPlanner.Item(
                String.valueOf(requestId),
                priority,
                enqueueSequence,
                NOW_MS - 1L,
                expiresAtMs,
                serviceTokens,
                0L);
    }

    static Probe probe(long requestId, int priority, long sequenceLength, long hitCache) {
        return probe(requestId, priority, NOW_MS, Long.MAX_VALUE, sequenceLength, hitCache);
    }

    static Probe probe(
            long requestId,
            int priority,
            long enqueuedAtMs,
            long expiresAtMs,
            long sequenceLength,
            long hitCache) {
        return new Probe(
                requestId, priority, enqueuedAtMs, expiresAtMs, sequenceLength, hitCache, hitCache);
    }

    static WorkSnapshot noCommittedWork() {
        return work(List.of(), List.of(), 0L);
    }

    static WorkSnapshot work(
            List<WorkSnapshot.RequestWork> requests,
            List<WorkSnapshot.BatchWork> batches,
            long unknownRequestCount) {
        return new WorkSnapshot(NOW_MS, requests, batches, unknownRequestCount);
    }

    static Candidate project(
            EstimationSnapshot queue,
            WorkSnapshot work,
            PrefillTimePredictor.Evaluator evaluator,
            Probe probe,
            org.flexlb.config.DispatcherConfig.Type deliveryMode) {
        return project(
                new RequestTimeEstimation.Inputs(queue, work, 0L, evaluator, deliveryMode),
                probe,
                queue.capturedAtMs());
    }

    private static Candidate immutable(RequestTimeEstimation.CandidateView source) {
        return new Candidate(
                source.state(),
                source.projectedTtftMsValue(),
                source.incomingPrefillMs(),
                source.initialHeadDisposition(),
                source.detail(),
                source.cacheHitTokens(),
                source.routingCacheMatchTokens());
    }

    /** Virtual request evaluated against one frozen endpoint snapshot. */
    public record Probe(
            long requestId,
            int priority,
            long enqueuedAtMs,
            long expiresAtMs,
            long seqLen,
            long hitCache,
            long routingCacheMatchTokens) {

        public Probe {
            checkArgument(seqLen >= 0L, "seqLen must be non-negative");
            checkArgument(hitCache >= 0L && hitCache <= seqLen, "hitCache must be in [0, seqLen]");
            checkArgument(
                    routingCacheMatchTokens >= 0L, "routingCacheMatchTokens must be non-negative");
        }
    }

    public record Candidate(
            RequestTimeEstimation.State state,
            long projectedTtftMsValue,
            long incomingPrefillMs,
            RequestTimeEstimation.InitialHeadDisposition initialHeadDisposition,
            String detail,
            long cacheHitTokens,
            long routingCacheMatchTokens)
            implements RequestTimeEstimation.CandidateView {}
}
