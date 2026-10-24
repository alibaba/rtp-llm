package org.flexlb.balance.projection.estimator;

import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.prediction.PrefillPredictionBoundary;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.projection.PrefillTtftEstimate;
import org.flexlb.balance.scheduler.ScheduledRequest;

import java.util.List;

/**
 * BATCH 模式的 Prefill 耗时预估：一批请求一起发给引擎，按整批的特征预测耗时。
 */
public final class BatchTtftEstimator implements PrefillTtftEstimator {

    private static final PrefillTtftEstimate.DeliveryProjection PROJECTION =
            new BatchProjection();

    @Override
    public GroupPlanner.PrefixPrediction<ScheduledRequest> newGroupPredictor(
            PrefillTimePredictor.Evaluator evaluator) {
        PrefillTimePredictor.BatchPrediction prediction = evaluator.newBatchPrediction();
        return (added, items) -> {
            return PrefillPredictionBoundary.requireValidDecisionGroupMs(
                    prediction.append(added.seqLen(), added.hitCache()));
        };
    }

    @Override
    public PrefillTtftEstimate.DeliveryProjection ttftEstimationRule() {
        return PROJECTION;
    }

    private static final class BatchProjection
            implements PrefillTtftEstimate.DeliveryProjection {

        private static final ThreadLocal<BatchPlanning> PLANNING =
                ThreadLocal.withInitial(BatchPlanning::new);
        private static final ThreadLocal<BatchService> SERVICE =
                ThreadLocal.withInitial(BatchService::new);

        @Override
        public long singletonCompletionOffsetMs(
                long seqLen,
                long hitCache,
                PrefillTtftEstimate.Predictions predictions) {
            // Committed prediction performs the same validated decision-group
            // evaluation before converting to lifecycle milliseconds. Running
            // the planning call first only evaluates the predictor twice for
            // every endpoint in the full-fleet singleton fast path.
            return predictions.singletonBatchDurationMs(seqLen, hitCache);
        }

        @Override
        public PrefillTtftEstimate.GroupPlanning planning(
                PrefillTtftEstimate.Predictions predictions) {
            PrefillTimePredictor.BatchPrediction incremental = predictions.newBatchPrediction();
            if (incremental != null) {
                return new AppendPlanning(incremental);
            }
            BatchPlanning planning = PLANNING.get();
            planning.reset(predictions);
            return planning;
        }

        @Override
        public PrefillTtftEstimate.GroupService service(
                GroupPlanner.Plan<GroupPlanner.Item> plan,
                PrefillTtftEstimate.Predictions predictions) {
            BatchService service = SERVICE.get();
            service.reset(plan, predictions, null);
            return service;
        }

        @Override
        public PrefillTtftEstimate.GroupService service(
                GroupPlanner.Plan<GroupPlanner.Item> plan, PrefillTtftEstimate.Predictions predictions,
                PrefillTtftEstimate.GroupPlanning planning) {
            BatchService service = SERVICE.get();
            service.reset(plan, predictions, planning);
            return service;
        }

        /** One planner invocation; prefixes only grow and the probe boundary never moves backwards. */
        private static final class AppendPlanning implements PrefillTtftEstimate.GroupPlanning {
            private final PrefillTimePredictor.BatchPrediction prediction;
            private int size;
            private double latestPredictionMs;
            private double previousPredictionMs;

            private AppendPlanning(PrefillTimePredictor.BatchPrediction prediction) {
                this.prediction = prediction;
            }

            @Override
            public double durationMs(List<GroupPlanner.Item> prefix, int through) {
                if (through < 0 || through >= prefix.size() || through + 1 < size) {
                    throw new IllegalArgumentException("Prediction requires a growing prefix");
                }
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

        private static final class BatchPlanning
                implements PrefillTtftEstimate.GroupPlanning {
            private GroupPlanner.Item[] previous = new GroupPlanner.Item[0];
            private double[] durations = new double[0];
            private boolean[] computed = new boolean[0];
            private int previousSize;
            private PrefillTtftEstimate.Predictions predictions;

            private void reset(PrefillTtftEstimate.Predictions exactPredictions) {
                predictions = exactPredictions;
                if (previousSize != 0) {
                    java.util.Arrays.fill(computed, 0, previousSize, false);
                }
                previousSize = 0;
            }

            @Override
            public double durationMs(
                    List<GroupPlanner.Item> candidatePrefix,
                    int requiredThroughIndex) {
                if (requiredThroughIndex < 0
                        || requiredThroughIndex >= candidatePrefix.size()) {
                    throw new IndexOutOfBoundsException(requiredThroughIndex);
                }
                if (!isIdentityPrefix(candidatePrefix)) {
                    ensureCapacity(candidatePrefix.size());
                    for (int index = 0; index < candidatePrefix.size(); index++) {
                        previous[index] = candidatePrefix.get(index);
                    }
                    java.util.Arrays.fill(
                            computed, 0, candidatePrefix.size(), false);
                    previousSize = candidatePrefix.size();
                } else if (previousSize < candidatePrefix.size()) {
                    ensureCapacity(candidatePrefix.size());
                    for (int index = previousSize;
                            index < candidatePrefix.size(); index++) {
                        previous[index] = candidatePrefix.get(index);
                        computed[index] = false;
                    }
                    previousSize = candidatePrefix.size();
                }
                if (!computed[requiredThroughIndex]) {
                    durations[requiredThroughIndex] =
                            predictions.batchPlanningDurationMs(
                                    candidatePrefix.subList(
                                            0, requiredThroughIndex + 1));
                    computed[requiredThroughIndex] = true;
                }
                return durations[requiredThroughIndex];
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
                durations = java.util.Arrays.copyOf(durations, size);
                computed = java.util.Arrays.copyOf(computed, size);
            }
        }

        private static final class BatchService
                implements PrefillTtftEstimate.GroupService {
            private GroupPlanner.Plan<GroupPlanner.Item> plan;
            private PrefillTtftEstimate.Predictions predictions;
            private PrefillTtftEstimate.GroupPlanning planning;
            private long[] completionOffsets = new long[0];
            private boolean[] computed = new boolean[0];

            private void reset(
                    GroupPlanner.Plan<GroupPlanner.Item> exactPlan,
                    PrefillTtftEstimate.Predictions exactPredictions,
                    PrefillTtftEstimate.GroupPlanning exactPlanning) {
                plan = exactPlan;
                predictions = exactPredictions;
                planning = exactPlanning;
                int size = exactPlan.items().size();
                if (completionOffsets.length < size) {
                    completionOffsets = java.util.Arrays.copyOf(
                            completionOffsets, size);
                    computed = java.util.Arrays.copyOf(computed, size);
                }
                java.util.Arrays.fill(computed, 0, size, false);
            }

            @Override
            public long completionOffsetMs(int memberIndex) {
                if (memberIndex < 0 || memberIndex >= plan.items().size()) {
                    throw new IndexOutOfBoundsException(memberIndex);
                }
                if (!computed[memberIndex] && planning != null) {
                    var cached = planning.predictedPrefixMs(memberIndex + 1);
                    if (cached.isPresent()) {
                        completionOffsets[memberIndex] = PrefillPredictionBoundary.committedDecisionGroupMs(
                                cached.getAsDouble());
                        computed[memberIndex] = true;
                    }
                }
                if (!computed[memberIndex]) {
                    completionOffsets[memberIndex] =
                            predictions.batchDurationMs(
                                    plan.items().subList(0, memberIndex + 1));
                    computed[memberIndex] = true;
                }
                return completionOffsets[memberIndex];
            }

            @Override
            public long totalDurationMs() {
                return completionOffsetMs(plan.items().size() - 1);
            }
        }
    }
}
