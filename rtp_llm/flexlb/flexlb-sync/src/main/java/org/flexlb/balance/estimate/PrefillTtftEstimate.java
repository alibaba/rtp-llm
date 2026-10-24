package org.flexlb.balance.estimate;

import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.dao.route.RoleType;

import java.util.List;
import java.util.OptionalDouble;
import java.util.OptionalLong;

/**
 * 基于一致捕获的不可变快照，预估一条新请求排进某台 prefill 机器后的 TTFT。
 */
public final class PrefillTtftEstimate {

    private static final ThreadLocal<Session> SESSIONS =
            ThreadLocal.withInitial(Session::new);

    private PrefillTtftEstimate() {
    }

    /**
     * Pure immutable TTFT estimation rule SPI. Implementations live with the
     * delivery strategy, while this class owns the input and result vocabulary.
     */
    public interface TtftEstimationRule {

        /** Exact service completion for an otherwise empty endpoint queue. */
        long singletonCompletionOffsetMs(
                long seqLen,
                long hitCache,
                Predictions predictions);

        /** Invocation-local, prefix-aware planning cursor. */
        GroupPlanning planning(Predictions predictions);

        GroupService service(
                GroupPlanner.Plan<GroupPlanner.Item> plan,
                Predictions predictions);

        /** Service for the result of this exact planning cursor, with the same frozen model and inputs. */
        default GroupService service(GroupPlanner.Plan<GroupPlanner.Item> plan,
                                     Predictions predictions, GroupPlanning planning) {
            return service(plan, predictions);
        }
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

        default long itemDurationMs(long seqLen, long hitCache) {
            return itemDurationMs(new GroupPlanner.Item(
                    "", 0, 0L, 0L, Long.MAX_VALUE,
                    seqLen, hitCache));
        }

        double batchPlanningDurationMs(List<GroupPlanner.Item> items);

        /** Optional append-only implementation, scoped to one GroupPlanner invocation. */
        default PrefillTimePredictor.BatchPrediction newBatchPrediction() {
            return null;
        }

        long batchDurationMs(List<GroupPlanner.Item> items);

        default long singletonBatchDurationMs(
                long seqLen, long hitCache) {
            return batchDurationMs(List.of(new GroupPlanner.Item(
                    "", 0, 0L, 0L, Long.MAX_VALUE,
                    seqLen, hitCache)));
        }
    }

    /** Invocation-local lazy service cursor for one exact planned group. */
    public interface GroupService {

        /** Predict only through this member's completion. */
        long completionOffsetMs(int memberIndex);

        /** Predict a complete group that runs before the probe. */
        long totalDurationMs();
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
    }

    /** Virtual request evaluated against one frozen endpoint snapshot. */
    public record Probe(
            String requestId,
            int priority,
            long enqueuedAtMs,
            long expiresAtMs,
            long seqLen,
            long hitCache,
            long routingCacheMatchTokens) {

        public Probe {
            if (seqLen < 0L) {
                throw new IllegalArgumentException("seqLen must be non-negative");
            }
            if (hitCache < 0L || hitCache > seqLen) {
                throw new IllegalArgumentException("hitCache must be in [0, seqLen]");
            }
            if (routingCacheMatchTokens < 0L) {
                throw new IllegalArgumentException(
                        "routingCacheMatchTokens must be non-negative");
            }
        }
    }

    /** Read-only estimate result. Invocation-scoped views must not be retained. */
    public interface CandidateView {
        Candidate.State state();

        long estimatedTtftMsValue();

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
                    && estimatedTtftMsValue() != Candidate.UNKNOWN;
        }
    }

    /** Complete immutable estimate used outside the full-fleet hot path. */
    public record Candidate(
            State state,
            long estimatedTtftMsValue,
            long incomingPrefillMs,
            InitialHeadDisposition initialHeadDisposition,
            String detail,
            RoleType blockerRole,
            long cacheHitTokens,
            long routingCacheMatchTokens) implements CandidateView {

        public static final long UNKNOWN = -1L;

        public Candidate {
            requireKnownOrUnknown(estimatedTtftMsValue, "estimatedTtftMs");
            if ((state == State.MODELED)
                    != (estimatedTtftMsValue != UNKNOWN)) {
                throw new IllegalArgumentException(
                        "only MODELED estimates may carry a estimated TTFT");
            }
            if (incomingPrefillMs < 0L) {
                throw new IllegalArgumentException(
                        "incomingPrefillMs must be non-negative");
            }
            if (blockerRole != null
                    && state != State.BLOCKED
                    && state != State.UNAVAILABLE) {
                throw new IllegalArgumentException(
                        "capacity block requires a blocked or unavailable result");
            }
            detail = detail == null ? "" : detail;
            if (cacheHitTokens < 0L || routingCacheMatchTokens < 0L) {
                throw new IllegalArgumentException(
                        "cache token counts must be non-negative");
            }
        }

        public OptionalLong estimatedTtftMs() {
            return optional(estimatedTtftMsValue);
        }

        public long requiredEstimatedTtftMs() {
            if (estimatedTtftMsValue == UNKNOWN) {
                throw new IllegalStateException(
                        "candidate estimated TTFT is unknown");
            }
            return estimatedTtftMsValue;
        }

        public boolean engineWorkUnmodeled() {
            return state == State.UNMODELED_ENGINE_WORK;
        }

        public boolean selectable() {
            return state == State.MODELED
                    && estimatedTtftMsValue != UNKNOWN;
        }

        private static void requireKnownOrUnknown(long value, String name) {
            if (value < 0L && value != UNKNOWN) {
                throw new IllegalArgumentException(name + " must be non-negative");
            }
        }

        private static OptionalLong optional(long value) {
            return value == UNKNOWN
                    ? OptionalLong.empty() : OptionalLong.of(value);
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
    public record LoadView(
            QueueView queue,
            WorkView work,
            long ownershipVersion) {

        public LoadView(
                QueueView queue,
                WorkView work) {
            this(queue, work, 0L);
        }

        public LoadView {
            // Committed work may reuse an older clock base while its ownership
            // is unchanged. The timelineEstimator rebases running duration to queue time.
            if (queue.capturedAtMs() < work.capturedAtMs()) {
                throw new IllegalArgumentException(
                        "work snapshot cannot be newer than its queue capture");
            }
            if (ownershipVersion < 0L) {
                throw new IllegalArgumentException(
                        "ownershipVersion must be non-negative");
            }
        }
    }

    public static Candidate estimate(
            LoadView inputs,
            Probe probe,
            PrefillTimePredictor.Evaluator evaluator,
            TtftEstimationRule ttftEstimationRule) {
        return estimate(inputs, probe, evaluator, ttftEstimationRule,
                inputs.queue().capturedAtMs());
    }

    public static Candidate estimate(
            LoadView inputs,
            Probe probe,
            PrefillTimePredictor.Evaluator evaluator,
            TtftEstimationRule ttftEstimationRule,
            long planningAtMs) {
        CandidateView view = estimateView(
                inputs, probe.requestId(), probe.priority(),
                probe.enqueuedAtMs(), probe.expiresAtMs(), probe.seqLen(),
                probe.hitCache(), probe.routingCacheMatchTokens(),
                evaluator, ttftEstimationRule, planningAtMs);
        return immutable(view);
    }

    /**
     * Allocation-free full-fleet estimate. The returned view is overwritten
     * by the next estimate on the same thread and must be copied immediately.
     */
    public static CandidateView estimateView(
            LoadView inputs,
            String requestId,
            int priority,
            long enqueuedAtMs,
            long expiresAtMs,
            long seqLen,
            long hitCache,
            long routingCacheMatchTokens,
            PrefillTimePredictor.Evaluator evaluator,
            TtftEstimationRule ttftEstimationRule,
            long planningAtMs) {
        return session().estimateView(
                inputs, requestId, priority, enqueuedAtMs, expiresAtMs,
                seqLen, hitCache, routingCacheMatchTokens,
                evaluator, ttftEstimationRule, planningAtMs);
    }

    /** Reusable timelineEstimator owned by one planner thread for one fleet traversal. */
    public static Session session() {
        return SESSIONS.get();
    }

    public static final class Session {
        private final QueueTimelineEstimator timelineEstimator =
                new QueueTimelineEstimator();

        private Session() {
        }

        public CandidateView estimateView(LoadView inputs,
                                         String requestId,
                                         int priority,
                                         long enqueuedAtMs,
                                         long expiresAtMs,
                                         long seqLen,
                                         long hitCache,
                                         long routingCacheMatchTokens,
                                         PrefillTimePredictor.Evaluator evaluator,
                                         TtftEstimationRule ttftEstimationRule,
                                         long planningAtMs) {
            CandidateView candidate = estimateViewForSelection(
                    inputs, requestId, priority, enqueuedAtMs, expiresAtMs,
                    seqLen, hitCache, routingCacheMatchTokens,
                    evaluator, ttftEstimationRule, planningAtMs);
            return applyAdmissionPolicy(inputs.queue(), candidate);
        }

        /**
         * Estimates endpoint work while leaving observed capacity waits to the delivery step.
         */
        public CandidateView estimateViewForSelection(LoadView inputs,
                                                     String requestId,
                                                     int priority,
                                                     long enqueuedAtMs,
                                                     long expiresAtMs,
                                                     long seqLen,
                                                     long hitCache,
                                                     long routingCacheMatchTokens,
                                                     PrefillTimePredictor.Evaluator evaluator,
                                                     TtftEstimationRule ttftEstimationRule,
                                                     long planningAtMs) {
            timelineEstimator.reset(requestId, priority, enqueuedAtMs, expiresAtMs,
                    seqLen, hitCache, routingCacheMatchTokens);
            return timelineEstimator.estimate(
                    inputs.queue(), inputs.work(), evaluator,
                    ttftEstimationRule, planningAtMs);
        }
    }

    /** Apply one observed worker admission wait without inventing a release duration. */
    private static CandidateView applyAdmissionPolicy(QueueView queue, CandidateView candidate) {
        QueueView.AdmissionBlock observation = queue.admissionBlock();
        if (observation == null || observation.semantics() == null
                || (candidate.state() != Candidate.State.MODELED && !candidate.engineWorkUnmodeled())) {
            return candidate;
        }
        if (candidate.engineWorkUnmodeled()) {
            return copyAdmissionResult(
                    candidate,
                    Candidate.State.BLOCKED,
                    Candidate.UNKNOWN,
                    observation.semantics().blockedDetail(),
                    blockerRole(observation.semantics()));
        }

        return switch (candidate.initialHeadDisposition()) {
            case TERMINAL_PRUNED -> candidate;
            case BEFORE_PROBE -> copyAdmissionResult(
                    candidate,
                    Candidate.State.BLOCKED,
                    Candidate.UNKNOWN,
                    observation.semantics().blockedDetail(),
                    blockerRole(observation.semantics()));
            case NONE -> throw new IllegalStateException(
                    "admission-blocked ACTIVE head was not estimated");
            case AFTER_PROBE -> applyAfterProbeAdmission(
                    observation.semantics(), candidate);
        };
    }

    private static CandidateView applyAfterProbeAdmission(
            AdmissionBlockSemantics semantics,
            CandidateView candidate) {
        return switch (semantics.afterProbe()) {
            case BLOCKED -> copyAdmissionResult(
                    candidate,
                    Candidate.State.BLOCKED,
                    Candidate.UNKNOWN,
                    semantics.afterProbeDetail(),
                    null);
            case UNAVAILABLE -> copyAdmissionResult(
                    candidate,
                    Candidate.State.UNAVAILABLE,
                    Candidate.UNKNOWN,
                    semantics.afterProbeDetail(),
                    blockerRole(semantics));
        };
    }

    private static RoleType blockerRole(
            AdmissionBlockSemantics semantics) {
        return semantics.afterProbe() == AfterProbeAdmission.UNAVAILABLE
                ? semantics.blockerRole()
                : null;
    }

    private static Candidate copyAdmissionResult(
            CandidateView source,
            Candidate.State state,
            long estimatedTtftMs,
            String detail,
            RoleType blockerRole) {
        return new Candidate(
                state,
                estimatedTtftMs,
                source.incomingPrefillMs(),
                source.initialHeadDisposition(),
                detail,
                blockerRole,
                source.cacheHitTokens(),
                source.routingCacheMatchTokens());
    }

    private static Candidate immutable(CandidateView source) {
        return source instanceof Candidate candidate
                ? candidate
                : new Candidate(
                        source.state(),
                        source.estimatedTtftMsValue(),
                        source.incomingPrefillMs(),
                        source.initialHeadDisposition(),
                        source.detail(),
                        source.blockerRole(),
                        source.cacheHitTokens(),
                        source.routingCacheMatchTokens());
    }
}
