package org.flexlb.balance.estimate.estimator;

import org.flexlb.balance.estimate.PrefillTtftEstimate;
import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.prediction.PrefillPredictionBoundary;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.scheduler.ScheduledRequest;

import java.util.List;

/**
 * NON_BATCH 模式的 Prefill 耗时预估：每个请求单独发给引擎，前面请求的耗时逐个累加。
 */
public final class NonBatchTtftEstimator implements PrefillTtftEstimator {

    private static final PrefillTtftEstimate.TtftEstimationRule ESTIMATION_RULE =
            new NonBatchTtftEstimationRule();

    @Override
    public GroupPlanner.PrefixPrediction<ScheduledRequest> newGroupPredictor(
            PrefillTimePredictor.Evaluator evaluator) {
        return (added, items) -> estimateGroupDurationMs(items, evaluator);
    }

    private static double estimateGroupDurationMs(
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
    public PrefillTtftEstimate.TtftEstimationRule ttftEstimationRule() {
        return ESTIMATION_RULE;
    }

    private static final class NonBatchTtftEstimationRule
            implements PrefillTtftEstimate.TtftEstimationRule {

        private static final ThreadLocal<RoutePlanning> PLANNING =
                ThreadLocal.withInitial(RoutePlanning::new);
        private static final ThreadLocal<RouteService> SERVICE =
                ThreadLocal.withInitial(RouteService::new);

        @Override
        public long singletonCompletionOffsetMs(
                long seqLen,
                long hitCache,
                PrefillTtftEstimate.Predictions predictions) {
            return predictions.itemDurationMs(seqLen, hitCache);
        }

        @Override
        public PrefillTtftEstimate.GroupPlanning planning(
                PrefillTtftEstimate.Predictions predictions) {
            RoutePlanning planning = PLANNING.get();
            planning.reset(predictions);
            return planning;
        }

        @Override
        public PrefillTtftEstimate.GroupService service(
                GroupPlanner.Plan<GroupPlanner.Item> plan,
                PrefillTtftEstimate.Predictions predictions) {
            RouteService service = SERVICE.get();
            service.reset(plan, predictions);
            return service;
        }

        private static final class RoutePlanning
                implements PrefillTtftEstimate.GroupPlanning {
            private GroupPlanner.Item[] previous = new GroupPlanner.Item[0];
            private long[] offsets = new long[0];
            private int previousSize;
            private int computedThrough;
            private PrefillTtftEstimate.Predictions predictions;

            private void reset(PrefillTtftEstimate.Predictions exactPredictions) {
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
                implements PrefillTtftEstimate.GroupService {
            private GroupPlanner.Plan<GroupPlanner.Item> plan;
            private PrefillTtftEstimate.Predictions predictions;
            private long[] offsets = new long[0];
            private int computedThrough;

            private void reset(
                    GroupPlanner.Plan<GroupPlanner.Item> exactPlan,
                    PrefillTtftEstimate.Predictions exactPredictions) {
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
