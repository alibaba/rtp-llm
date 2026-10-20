package org.flexlb.balance.projection;

import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.prediction.PrefillTimePredictor;

import java.util.List;
import java.util.OptionalDouble;
import java.util.OptionalLong;

/** Projects one incoming route against immutable, coherently captured inputs. */
public final class RouteProjection {

    private static final ThreadLocal<Session> SESSIONS =
            ThreadLocal.withInitial(Session::new);

    private RouteProjection() {
    }

    /**
     * Pure immutable delivery projection SPI. Implementations live with the
     * delivery strategy, while projection owns the input and result vocabulary.
     */
    public interface DeliveryProjection {

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
        INDEPENDENT
    }

    /** Delivery-independent meaning of one exact captured admission block. */
    public record AdmissionBlockSemantics(
            String blockedDetail,
            AfterProbeAdmission afterProbe,
            String afterProbeDetail) {
        public AdmissionBlockSemantics {
            java.util.Objects.requireNonNull(blockedDetail, "blockedDetail");
            java.util.Objects.requireNonNull(afterProbe, "afterProbe");
            if (afterProbe == AfterProbeAdmission.BLOCKED) {
                java.util.Objects.requireNonNull(afterProbeDetail, "afterProbeDetail");
            }
        }
    }

    /** Virtual request evaluated against one frozen endpoint snapshot. */
    public record Probe(
            String requestId,
            int priority,
            long enqueueSeq,
            long enqueuedAtMs,
            long expiresAtMs,
            long seqLen,
            long hitCache,
            long routingCacheMatchTokens) {

        /** New requests follow existing requests of the same priority. */
        public Probe(String requestId, int priority, long enqueuedAtMs,
                     long expiresAtMs, long seqLen, long hitCache, long routingCacheMatchTokens) {
            this(requestId, priority, Long.MAX_VALUE, enqueuedAtMs,
                    expiresAtMs, seqLen, hitCache, routingCacheMatchTokens);
        }

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

    /** Read-only projection result. Invocation-scoped views must not be retained. */
    public interface CandidateView {
        long projectedTtftMsValue();

        long incomingPrefillMs();

        Candidate.InitialHeadDisposition initialHeadDisposition();

        String detail();

        long cacheHitTokens();

        long routingCacheMatchTokens();

        default boolean hasPrediction() {
            return projectedTtftMsValue() != Candidate.UNKNOWN;
        }
    }

    /** Complete immutable projection used outside the full-fleet hot path. */
    public record Candidate(
            long projectedTtftMsValue,
            long incomingPrefillMs,
            InitialHeadDisposition initialHeadDisposition,
            String detail,
            long cacheHitTokens,
            long routingCacheMatchTokens) implements CandidateView {

        public static final long UNKNOWN = -1L;

        public Candidate {
            requireKnownOrUnknown(projectedTtftMsValue, "projectedTtftMs");
            if (incomingPrefillMs < 0L) {
                throw new IllegalArgumentException(
                        "incomingPrefillMs must be non-negative");
            }
            detail = detail == null ? "" : detail;
            if (cacheHitTokens < 0L || routingCacheMatchTokens < 0L) {
                throw new IllegalArgumentException(
                        "cache token counts must be non-negative");
            }
        }

        public OptionalLong projectedTtftMs() {
            return optional(projectedTtftMsValue);
        }

        public long requiredProjectedTtftMs() {
            if (projectedTtftMsValue == UNKNOWN) {
                throw new IllegalStateException(
                        "candidate projected TTFT is unknown");
            }
            return projectedTtftMsValue;
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
    }

    /** Queue and committed work captured under one ownership lock. */
    public record Inputs(
            QueueSnapshot queue,
            WorkSnapshot work,
            long ownershipVersion) {

        public Inputs(
                QueueSnapshot queue,
                WorkSnapshot work) {
            this(queue, work, 0L);
        }

        public Inputs {
            // Committed work may reuse an older clock base while its ownership
            // is unchanged. The projector rebases running duration to queue time.
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

    public static Candidate project(
            Inputs inputs,
            Probe probe,
            PrefillTimePredictor.Evaluator evaluator,
            DeliveryProjection deliveryProjection) {
        return project(inputs, probe, evaluator, deliveryProjection,
                inputs.queue().capturedAtMs());
    }

    public static Candidate project(
            Inputs inputs,
            Probe probe,
            PrefillTimePredictor.Evaluator evaluator,
            DeliveryProjection deliveryProjection,
            long planningAtMs) {
        CandidateView view = projectView(
                inputs, probe.requestId(), probe.priority(),
                probe.enqueueSeq(), probe.enqueuedAtMs(), probe.expiresAtMs(), probe.seqLen(),
                probe.hitCache(), probe.routingCacheMatchTokens(),
                evaluator, deliveryProjection, planningAtMs);
        return immutable(view);
    }

    /**
     * Allocation-free full-fleet projection. The returned view is overwritten
     * by the next projection on the same thread and must be copied immediately.
     */
    public static CandidateView projectView(
            Inputs inputs,
            String requestId,
            int priority,
            long enqueueSeq,
            long enqueuedAtMs,
            long expiresAtMs,
            long seqLen,
            long hitCache,
            long routingCacheMatchTokens,
            PrefillTimePredictor.Evaluator evaluator,
            DeliveryProjection deliveryProjection,
            long planningAtMs) {
        return session().projectView(
                inputs, requestId, priority, enqueueSeq, enqueuedAtMs, expiresAtMs,
                seqLen, hitCache, routingCacheMatchTokens,
                evaluator, deliveryProjection, planningAtMs);
    }

    /** Reusable projector owned by one planner thread for one fleet traversal. */
    public static Session session() {
        return SESSIONS.get();
    }

    public static final class Session {
        private final RouteTimelineProjector projector =
                new RouteTimelineProjector();

        private Session() {
        }

        /** Modeled queue work for ranking, excluding waits with no known release time. */
        public CandidateView projectView(
                Inputs inputs, String requestId, int priority, long enqueueSeq, long enqueuedAtMs,
                long expiresAtMs, long seqLen, long hitCache, long routingCacheMatchTokens,
                PrefillTimePredictor.Evaluator evaluator,
                DeliveryProjection deliveryProjection, long planningAtMs) {
            projector.reset(
                    requestId, priority, enqueueSeq, enqueuedAtMs, expiresAtMs,
                    seqLen, hitCache, routingCacheMatchTokens);
            return projector.project(inputs.queue(), inputs.work(), evaluator,
                    deliveryProjection, planningAtMs);
        }
    }

    /** The captured head's block reason for this probe, or null when it does not apply. */
    public static String admissionBlockReason(
            QueueSnapshot queue, CandidateView prediction, GroupPlanner.Item probe, long planningAtMs) {
        QueueSnapshot.AdmissionBlock observation = queue.admissionBlock();
        if (!queue.queueScheduling()
                || observation == null
                || observation.semantics() == null
                || !prediction.hasPrediction()) {
            return null;
        }
        AdmissionBlockSemantics semantics = observation.semantics();
        return switch (prediction.initialHeadDisposition()) {
            case TERMINAL_PRUNED -> null;
            case BEFORE_PROBE -> semantics.afterProbe() == AfterProbeAdmission.INDEPENDENT
                    && canBypassPredecessors(queue, probe, planningAtMs)
                    ? null : semantics.blockedDetail();
            case AFTER_PROBE -> semantics.afterProbe() == AfterProbeAdmission.BLOCKED
                    ? semantics.afterProbeDetail() : null;
            case NONE -> throw new IllegalStateException(
                    "admission-blocked ACTIVE head was not projected");
        };
    }

    private static boolean canBypassPredecessors(
            QueueSnapshot queue, GroupPlanner.Item probe, long planningAtMs) {
        long nowMs = Math.max(queue.capturedAtMs(), planningAtMs);
        for (GroupPlanner.Item item : queue.activeItems()) {
            if (queue.ordering().compare(probe, item) < 0) {
                break;
            }
            if (item.expiresAtMs() > nowMs && item.priority() > probe.priority()) {
                return false;
            }
        }
        return true;
    }

    private static Candidate immutable(CandidateView source) {
        return source instanceof Candidate candidate
                ? candidate
                : new Candidate(
                        source.projectedTtftMsValue(),
                        source.incomingPrefillMs(),
                        source.initialHeadDisposition(),
                        source.detail(),
                        source.cacheHitTokens(),
                        source.routingCacheMatchTokens());
    }
}
