package org.flexlb.balance.projection;

import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.prediction.PrefillPredictionBoundary;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.balance.scheduler.RequestRoute;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.dao.route.RoleType;

import java.util.List;
import java.util.Objects;
import java.util.OptionalDouble;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.math.LongMath.saturatedAdd;

/** Projects one incoming route against immutable, coherently captured inputs. */
public final class RouteProjection {

    private RouteProjection() {
    }

    /**
     * Pure prediction policy for the configured delivery mode.
     */
    public interface DeliveryProjection {

        /** Exact service completion for an otherwise empty endpoint queue. */
        long singletonCompletionOffsetMs(
                long seqLen,
                long hitCache,
                Predictions predictions);

        /** Invocation-local, prefix-aware planning cursor. */
        GroupPlanning planning(Predictions predictions);

        /** Compute the one required completion prefix, reusing only this exact planning cursor. */
        long completionOffsetMs(List<GroupPlanner.Item> items, int memberIndex,
                                Predictions predictions, GroupPlanning planning);

    }

    /**
     * Predict a candidate prefix only through the member whose timing is still
     * required, without evaluating members that complete after the probe.
     * Owned by one planner invocation: members are appended in order and the
     * required index never decreases. A changed snapshot gets a fresh cursor.
     */
    public interface GroupPlanning {
        double durationMs(
                List<GroupPlanner.Item> candidatePrefix,
                int requiredThroughIndex);

        /** Only predictions actually evaluated for this cursor's exact prefixes may be reused. */
        default OptionalDouble predictedPrefixMs(int size) {
            return OptionalDouble.empty();
        }
    }

    /** Prediction primitives evaluated against one frozen predictor snapshot. */
    public interface Predictions {

        long itemDurationMs(GroupPlanner.Item item);

        long itemDurationMs(long seqLen, long hitCache);

        /** Fresh append-only session; full-batch models adapt through Evaluator's default implementation. */
        PrefillTimePredictor.BatchPrediction newBatchPrediction();

        long batchDurationMs(List<GroupPlanner.Item> items);

        long singletonBatchDurationMs(long seqLen, long hitCache);

    }

    /** Effect of a captured blocked head after a probe has overtaken it. */
    public enum AfterProbeAdmission {
        BLOCKED,
        UNAVAILABLE
    }

    /** Delivery-independent meaning of one exact captured admission block. */
    public record AdmissionBlockSemantics(
            String blockedDetail,
            AfterProbeAdmission afterProbe,
            String afterProbeDetail,
            RoleType blockerRole) {
        public AdmissionBlockSemantics {
            blockerRole = java.util.Objects.requireNonNull(
                    blockerRole, "blockerRole");
        }
    }

    /** Read-only projection result. Invocation-scoped views must not be retained. */
    public interface CandidateView {
        Candidate.State state();

        long projectedTtftMsValue();

        long incomingPrefillMs();

        Candidate.InitialHeadDisposition initialHeadDisposition();

        String detail();

        RoleType blockerRole();

        long cacheHitTokens();

        long routingCacheMatchTokens();

        default boolean engineWorkUnmodeled() {
            return state() == Candidate.State.UNMODELED_ENGINE_WORK;
        }

        default boolean selectable() {
            return state() == Candidate.State.MODELED
                    && projectedTtftMsValue() != Candidate.UNKNOWN;
        }
    }

    /** Complete immutable projection used outside the full-fleet hot path. */
    public record Candidate(
            State state,
            long projectedTtftMsValue,
            long incomingPrefillMs,
            InitialHeadDisposition initialHeadDisposition,
            String detail,
            RoleType blockerRole,
            long cacheHitTokens,
            long routingCacheMatchTokens) implements CandidateView {

        public static final long UNKNOWN = -1L;

        public Candidate {
            checkArgument(projectedTtftMsValue >= UNKNOWN, "projectedTtftMs must be non-negative");
            checkArgument((state == State.MODELED) == (projectedTtftMsValue != UNKNOWN),
                    "only MODELED projections may carry a projected TTFT");
            checkArgument(incomingPrefillMs >= 0L, "incomingPrefillMs must be non-negative");
            checkArgument(blockerRole == null || state == State.BLOCKED || state == State.UNAVAILABLE,
                    "capacity block requires a blocked or unavailable result");
            detail = detail == null ? "" : detail;
            checkArgument(cacheHitTokens >= 0L && routingCacheMatchTokens >= 0L,
                    "cache token counts must be non-negative");
        }

        public enum InitialHeadDisposition {
            NONE,
            BEFORE_PROBE,
            AFTER_PROBE,
            TERMINAL_PRUNED
        }

        public enum State {
            MODELED,
            UNMODELED_ENGINE_WORK,
            BLOCKED,
            UNAVAILABLE
        }
    }

    /** Queue and committed work captured under one ownership lock. */
    public record Inputs(
            QueueSnapshot queue,
            WorkSnapshot work,
            long ownershipVersion) {

        public Inputs {
            // Committed work may reuse an older clock base while its ownership
            // is unchanged. The projector rebases running duration to queue time.
            checkArgument(queue.capturedAtMs() >= work.capturedAtMs(),
                    "work snapshot cannot be newer than its queue capture");
            checkArgument(ownershipVersion >= 0L, "ownershipVersion must be non-negative");
        }
    }

    /** Apply one observed worker admission wait without inventing a release duration. */
    static CandidateView applyAdmissionPolicy(
            QueueSnapshot queue,
            CandidateView candidate) {
        QueueSnapshot.AdmissionBlock observation = queue.admissionBlock();
        if (!queue.queueScheduling()
                || observation == null
                || observation.semantics() == null
                || (candidate.state() != Candidate.State.MODELED
                        && !candidate.engineWorkUnmodeled())) {
            return candidate;
        }
        AdmissionBlockSemantics semantics = observation.semantics();
        Candidate.State state = Candidate.State.BLOCKED;
        String detail = semantics.blockedDetail();
        if (!candidate.engineWorkUnmodeled()) {
            switch (candidate.initialHeadDisposition()) {
                case TERMINAL_PRUNED -> { return candidate; }
                case NONE -> throw new IllegalStateException(
                        "admission-blocked ACTIVE head was not projected");
                case BEFORE_PROBE -> { }
                case AFTER_PROBE -> {
                    state = switch (semantics.afterProbe()) {
                        case BLOCKED -> Candidate.State.BLOCKED;
                        case UNAVAILABLE -> Candidate.State.UNAVAILABLE;
                    };
                    detail = semantics.afterProbeDetail();
                }
            }
        }
        return new Candidate(
                state,
                Candidate.UNKNOWN,
                candidate.incomingPrefillMs(),
                candidate.initialHeadDisposition(),
                detail,
                semantics.afterProbe() == AfterProbeAdmission.UNAVAILABLE ? semantics.blockerRole() : null,
                candidate.cacheHitTokens(),
                candidate.routingCacheMatchTokens());
    }
    private static final DeliveryProjection BATCH = new BatchProjection();
    private static final DeliveryProjection ROUTE = new RouteProjectionPolicy();

    public static DeliveryProjection forDelivery(DispatcherConfig.Type mode) {
        return mode == DispatcherConfig.Type.BATCH ? BATCH : ROUTE;
    }

    /** One incremental predictor for one live queue selection. */
    public static GroupPlanner.PrefixPrediction<RequestRoute> newGroupPredictor(
            DispatcherConfig.Type mode, PrefillTimePredictor.Evaluator evaluator) {
        if (mode == DispatcherConfig.Type.BATCH) {
            var prediction = evaluator.newBatchPrediction();
            return (added, items) -> PrefillPredictionBoundary.requireValidDecisionGroupMs(
                    prediction.append(added.seqLen(), added.hitCache()));
        }
        return new GroupPlanner.PrefixPrediction<>() {
            private double totalMs;
            @Override public double append(RequestRoute added, List<RequestRoute> items) {
                totalMs += PrefillPredictionBoundary.predictSingleRequestMs(evaluator, added.seqLen(), added.hitCache());
                return PrefillPredictionBoundary.requireValidDecisionGroupMs(totalMs);
            }
        };
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
            return predictions.batchDurationMs(memberIndex + 1 == items.size()
                    ? items : items.subList(0, memberIndex + 1));
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
