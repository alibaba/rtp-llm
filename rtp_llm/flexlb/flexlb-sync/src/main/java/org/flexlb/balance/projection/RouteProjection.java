package org.flexlb.balance.projection;

import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.dao.route.RoleType;

import java.util.List;
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

    }

    /**
     * Predict a candidate prefix only through the member whose timing is still
     * required. A route projection can therefore establish probe TTFT without
     * evaluating a later suffix which is needed only for drain.
     */
    public interface GroupPlanning {
        double durationMs(
                List<GroupPlanner.Item> candidatePrefix,
                int requiredThroughIndex);
    }

    /** Prediction primitives evaluated against one frozen predictor snapshot. */
    public interface Predictions {

        long itemDurationMs(GroupPlanner.Item item);

        default long itemDurationMs(long seqLen, long hitCache) {
            return itemDurationMs(new GroupPlanner.Item(
                    0L, 0, 0L, 0L, Long.MAX_VALUE,
                    seqLen, hitCache));
        }

        double batchPlanningDurationMs(List<GroupPlanner.Item> items);

        default double singletonBatchPlanningDurationMs(
                long seqLen, long hitCache) {
            return batchPlanningDurationMs(List.of(new GroupPlanner.Item(
                    0L, 0, 0L, 0L, Long.MAX_VALUE,
                    seqLen, hitCache)));
        }

        long batchDurationMs(List<GroupPlanner.Item> items);

        default long singletonBatchDurationMs(
                long seqLen, long hitCache) {
            return batchDurationMs(List.of(new GroupPlanner.Item(
                    0L, 0, 0L, 0L, Long.MAX_VALUE,
                    seqLen, hitCache)));
        }

        long committedGroupDurationMs(double plannedDurationMs);
    }

    /** Invocation-local lazy service cursor for one exact planned group. */
    public interface GroupService {

        /** Predict only through this member's completion. */
        long completionOffsetMs(int memberIndex);

        /** Predict every remaining member, needed only for endpoint drain. */
        long totalDurationMs();
    }

    /** Effect of a captured blocked head after a probe has overtaken it. */
    public enum AfterProbeAdmission {
        BLOCKED,
        TTFT_KNOWN_DRAIN_UNKNOWN,
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

    /** Outputs requested by a routing policy from the frozen timeline. */
    public enum Demand {
        TTFT_ONLY(false),
        TTFT_AND_DRAIN(true);

        private final boolean drainRequired;

        Demand(boolean drainRequired) {
            this.drainRequired = drainRequired;
        }

        public boolean drainRequired() {
            return drainRequired;
        }
    }

    /** Virtual request evaluated against one frozen endpoint snapshot. */
    public record Probe(
            long requestId,
            int priority,
            long enqueuedAtMs,
            long expiresAtMs,
            long seqLen,
            long hitCache,
            long routingCacheMatchTokens,
            Demand demand) {

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

        GroupPlanner.Item asItem() {
            return new GroupPlanner.Item(
                    requestId, priority, Long.MAX_VALUE, enqueuedAtMs,
                    expiresAtMs, seqLen, hitCache);
        }
    }

    /** Read-only projection result. Invocation-scoped views must not be retained. */
    public interface CandidateView {
        Candidate.State state();

        long projectedTtftMsValue();

        long projectedDrainMsValue();

        long incomingPrefillMs();

        Candidate.InitialHeadDisposition initialHeadDisposition();

        String detail();

        RoleType blockerRole();

        long cacheHitTokens();

        long routingCacheMatchTokens();

        long pendingCountValue();

        default boolean hasProjectedDrain() {
            return projectedDrainMsValue() != Candidate.UNKNOWN;
        }

        default long requiredProjectedTtftMs() {
            if (projectedTtftMsValue() == Candidate.UNKNOWN) {
                throw new IllegalStateException(
                        "candidate projected TTFT is unknown");
            }
            return projectedTtftMsValue();
        }

        default long requiredProjectedDrainMs() {
            if (projectedDrainMsValue() == Candidate.UNKNOWN) {
                throw new IllegalStateException(
                        "candidate projected drain is unknown");
            }
            return projectedDrainMsValue();
        }

        default long requiredPendingCount() {
            if (pendingCountValue() == Candidate.UNKNOWN) {
                throw new IllegalStateException(
                        "candidate pending count is unknown");
            }
            return pendingCountValue();
        }

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
            long projectedDrainMsValue,
            long incomingPrefillMs,
            InitialHeadDisposition initialHeadDisposition,
            String detail,
            RoleType blockerRole,
            long cacheHitTokens,
            long routingCacheMatchTokens,
            long pendingCountValue) implements CandidateView {

        public static final long UNKNOWN = -1L;

        public Candidate {
            requireKnownOrUnknown(projectedTtftMsValue, "projectedTtftMs");
            requireKnownOrUnknown(projectedDrainMsValue, "projectedDrainMs");
            if ((state == State.MODELED)
                    != (projectedTtftMsValue != UNKNOWN)) {
                throw new IllegalArgumentException(
                        "only MODELED projections may carry a projected TTFT");
            }
            if (state != State.MODELED
                    && projectedDrainMsValue != UNKNOWN) {
                throw new IllegalArgumentException(
                        "only MODELED projections may carry a projected drain");
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
            boolean carriesPendingCount = state == State.MODELED
                    || state == State.UNMODELED_ENGINE_WORK;
            if (carriesPendingCount != (pendingCountValue != UNKNOWN)) {
                throw new IllegalArgumentException(
                        "only modeled or Engine-unmodeled projections may carry pending count");
            }
            requireKnownOrUnknown(pendingCountValue, "pendingCount");
        }

        public OptionalLong projectedTtftMs() {
            return optional(projectedTtftMsValue);
        }

        public OptionalLong projectedDrainMs() {
            return optional(projectedDrainMsValue);
        }

        public OptionalLong pendingCount() {
            return optional(pendingCountValue);
        }

        public boolean hasProjectedDrain() {
            return projectedDrainMsValue != UNKNOWN;
        }

        public long requiredProjectedTtftMs() {
            if (projectedTtftMsValue == UNKNOWN) {
                throw new IllegalStateException(
                        "candidate projected TTFT is unknown");
            }
            return projectedTtftMsValue;
        }

        public long requiredProjectedDrainMs() {
            if (projectedDrainMsValue == UNKNOWN) {
                throw new IllegalStateException(
                        "candidate projected drain is unknown");
            }
            return projectedDrainMsValue;
        }

        public long requiredPendingCount() {
            if (pendingCountValue == UNKNOWN) {
                throw new IllegalStateException(
                        "candidate pending count is unknown");
            }
            return pendingCountValue;
        }

        public boolean engineWorkUnmodeled() {
            return state == State.UNMODELED_ENGINE_WORK;
        }

        public boolean selectable() {
            return state == State.MODELED
                    && projectedTtftMsValue != UNKNOWN;
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

    /** Queue, committed work, and hard pending count from one ownership lock. */
    public record Inputs(
            QueueSnapshot queue,
            WorkSnapshot work,
            long pendingRequestCount,
            long ownershipVersion) {

        public Inputs(
                QueueSnapshot queue,
                WorkSnapshot work,
                long pendingRequestCount) {
            this(queue, work, pendingRequestCount, 0L);
        }

        public Inputs {
            if (queue.capturedAtMs() != work.capturedAtMs()) {
                throw new IllegalArgumentException(
                        "queue and work snapshots must share capturedAtMs");
            }
            if (pendingRequestCount < 0L) {
                throw new IllegalArgumentException(
                        "pendingRequestCount must be non-negative");
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
                probe.enqueuedAtMs(), probe.expiresAtMs(), probe.seqLen(),
                probe.hitCache(), probe.routingCacheMatchTokens(),
                probe.demand(), evaluator, deliveryProjection, planningAtMs);
        return immutable(view);
    }

    /** Allocation-free probe handoff for full-fleet selector hot paths. */
    public static Candidate project(
            Inputs inputs,
            long requestId,
            int priority,
            long enqueuedAtMs,
            long expiresAtMs,
            long seqLen,
            long hitCache,
            long routingCacheMatchTokens,
            Demand demand,
            PrefillTimePredictor.Evaluator evaluator,
            DeliveryProjection deliveryProjection,
            long planningAtMs) {
        return immutable(projectView(
                inputs, requestId, priority, enqueuedAtMs, expiresAtMs,
                seqLen, hitCache, routingCacheMatchTokens, demand,
                evaluator, deliveryProjection, planningAtMs));
    }

    /**
     * Allocation-free full-fleet projection. The returned view is overwritten
     * by the next projection on the same thread and must be copied immediately.
     */
    public static CandidateView projectView(
            Inputs inputs,
            long requestId,
            int priority,
            long enqueuedAtMs,
            long expiresAtMs,
            long seqLen,
            long hitCache,
            long routingCacheMatchTokens,
            Demand demand,
            PrefillTimePredictor.Evaluator evaluator,
            DeliveryProjection deliveryProjection,
            long planningAtMs) {
        return session().projectView(
                inputs, requestId, priority, enqueuedAtMs, expiresAtMs,
                seqLen, hitCache, routingCacheMatchTokens, demand,
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

        public CandidateView projectView(
                Inputs inputs,
                long requestId,
                int priority,
                long enqueuedAtMs,
                long expiresAtMs,
                long seqLen,
                long hitCache,
                long routingCacheMatchTokens,
                Demand demand,
                PrefillTimePredictor.Evaluator evaluator,
                DeliveryProjection deliveryProjection,
                long planningAtMs) {
            projector.reset(
                    requestId, priority, enqueuedAtMs, expiresAtMs,
                    seqLen, hitCache, routingCacheMatchTokens, demand,
                    inputs.pendingRequestCount());
            CandidateView candidate = projector.project(
                    inputs.queue(), inputs.work(), evaluator,
                    deliveryProjection, planningAtMs);
            return applyAdmissionPolicy(inputs.queue(), candidate);
        }
    }

    /** Apply one observed worker admission wait without inventing a release duration. */
    private static CandidateView applyAdmissionPolicy(
            QueueSnapshot queue,
            CandidateView candidate) {
        QueueSnapshot.AdmissionBlock observation = queue.admissionBlock();
        if (!queue.queueScheduling()
                || observation == null
                || (candidate.state() != Candidate.State.MODELED
                        && !candidate.engineWorkUnmodeled())) {
            return candidate;
        }
        if (candidate.engineWorkUnmodeled()) {
            return copyAdmissionResult(
                    candidate,
                    Candidate.State.BLOCKED,
                    Candidate.UNKNOWN,
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
                    Candidate.UNKNOWN,
                    observation.semantics().blockedDetail(),
                    blockerRole(observation.semantics()));
            case NONE -> throw new IllegalStateException(
                    "admission-blocked ACTIVE head was not projected");
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
                    Candidate.UNKNOWN,
                    semantics.afterProbeDetail(),
                    null);
            case TTFT_KNOWN_DRAIN_UNKNOWN -> copyAdmissionResult(
                    candidate,
                    Candidate.State.MODELED,
                    candidate.requiredProjectedTtftMs(),
                    Candidate.UNKNOWN,
                    semantics.afterProbeDetail(),
                    null);
            case UNAVAILABLE -> copyAdmissionResult(
                    candidate,
                    Candidate.State.UNAVAILABLE,
                    Candidate.UNKNOWN,
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
            long projectedTtftMs,
            long projectedDrainMs,
            String detail,
            RoleType blockerRole) {
        boolean carriesPending = state == Candidate.State.MODELED
                || state == Candidate.State.UNMODELED_ENGINE_WORK;
        return new Candidate(
                state,
                projectedTtftMs,
                projectedDrainMs,
                source.incomingPrefillMs(),
                source.initialHeadDisposition(),
                detail,
                blockerRole,
                source.cacheHitTokens(),
                source.routingCacheMatchTokens(),
                carriesPending
                        ? source.requiredPendingCount() : Candidate.UNKNOWN);
    }

    private static Candidate immutable(CandidateView source) {
        return source instanceof Candidate candidate
                ? candidate
                : new Candidate(
                        source.state(),
                        source.projectedTtftMsValue(),
                        source.projectedDrainMsValue(),
                        source.incomingPrefillMs(),
                        source.initialHeadDisposition(),
                        source.detail(),
                        source.blockerRole(),
                        source.cacheHitTokens(),
                        source.routingCacheMatchTokens(),
                        source.pendingCountValue());
    }
}
