package org.flexlb.balance.projection;

import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.dao.route.RoleType;

import java.util.List;
import java.util.OptionalLong;

/** Projects one incoming route against immutable, coherently captured inputs. */
public final class RouteProjection {

    private RouteProjection() {
    }

    /**
     * Pure immutable delivery projection SPI. Implementations live with the
     * delivery strategy, while projection owns the input and result vocabulary.
     */
    public interface DeliveryProjection {

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

        double batchPlanningDurationMs(List<GroupPlanner.Item> items);

        long batchDurationMs(List<GroupPlanner.Item> items);

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

    /** Complete immutable projection consumed by Prefill selection. */
    public record Candidate(
            State state,
            OptionalLong projectedTtftMs,
            OptionalLong projectedDrainMs,
            long incomingPrefillMs,
            InitialHeadDisposition initialHeadDisposition,
            String detail,
            RoleType blockerRole,
            long cacheHitTokens,
            long routingCacheMatchTokens,
            OptionalLong pendingCount) {

        public Candidate {
            projectedTtftMs = requireNonNegative(
                    projectedTtftMs, "projectedTtftMs");
            projectedDrainMs = requireNonNegative(
                    projectedDrainMs, "projectedDrainMs");
            if ((state == State.MODELED) != projectedTtftMs.isPresent()) {
                throw new IllegalArgumentException(
                        "only MODELED projections may carry a projected TTFT");
            }
            if (state != State.MODELED && projectedDrainMs.isPresent()) {
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
            pendingCount = java.util.Objects.requireNonNull(
                    pendingCount, "pendingCount");
            boolean carriesPendingCount = state == State.MODELED
                    || state == State.UNMODELED_ENGINE_WORK;
            if (carriesPendingCount != pendingCount.isPresent()) {
                throw new IllegalArgumentException(
                        "only modeled or Engine-unmodeled projections may carry pending count");
            }
            if (pendingCount.isPresent() && pendingCount.getAsLong() < 0L) {
                throw new IllegalArgumentException(
                        "pendingCount must be non-negative");
            }
        }

        public long requiredPendingCount() {
            return pendingCount.orElseThrow(
                    () -> new IllegalStateException(
                            "candidate pending count is unknown"));
        }

        public boolean engineWorkUnmodeled() {
            return state == State.UNMODELED_ENGINE_WORK;
        }

        public boolean selectable() {
            return state == State.MODELED && projectedTtftMs.isPresent();
        }

        private static OptionalLong requireNonNegative(
                OptionalLong value, String name) {
            if (value.isPresent() && value.getAsLong() < 0L) {
                throw new IllegalArgumentException(name + " must be non-negative");
            }
            return value;
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
            long pendingRequestCount) {

        public Inputs {
            if (queue.capturedAtMs() != work.capturedAtMs()) {
                throw new IllegalArgumentException(
                        "queue and work snapshots must share capturedAtMs");
            }
            if (pendingRequestCount < 0L) {
                throw new IllegalArgumentException(
                        "pendingRequestCount must be non-negative");
            }
        }
    }

    public static Candidate project(
            Inputs inputs,
            Probe probe,
            PrefillTimePredictor.Evaluator evaluator,
            DeliveryProjection deliveryProjection) {
        Candidate candidate = new RouteTimelineProjector(
                probe, inputs.pendingRequestCount()).project(
                        inputs.queue(), inputs.work(), evaluator,
                        deliveryProjection);
        return applyAdmissionPolicy(inputs.queue(), candidate);
    }

    /** Apply one observed worker admission wait without inventing a release duration. */
    private static Candidate applyAdmissionPolicy(
            QueueSnapshot queue,
            Candidate candidate) {
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
                    OptionalLong.empty(),
                    OptionalLong.empty(),
                    observation.semantics().blockedDetail(),
                    blockerRole(observation.semantics()));
        }

        return switch (candidate.initialHeadDisposition()) {
            case TERMINAL_PRUNED -> candidate;
            case BEFORE_PROBE -> copyAdmissionResult(
                    candidate,
                    Candidate.State.BLOCKED,
                    OptionalLong.empty(),
                    OptionalLong.empty(),
                    observation.semantics().blockedDetail(),
                    blockerRole(observation.semantics()));
            case NONE -> throw new IllegalStateException(
                    "admission-blocked ACTIVE head was not projected");
            case AFTER_PROBE -> applyAfterProbeAdmission(
                    observation.semantics(), candidate);
        };
    }

    private static Candidate applyAfterProbeAdmission(
            AdmissionBlockSemantics semantics,
            Candidate candidate) {
        return switch (semantics.afterProbe()) {
            case BLOCKED -> copyAdmissionResult(
                    candidate,
                    Candidate.State.BLOCKED,
                    OptionalLong.empty(),
                    OptionalLong.empty(),
                    semantics.afterProbeDetail(),
                    null);
            case TTFT_KNOWN_DRAIN_UNKNOWN -> copyAdmissionResult(
                    candidate,
                    Candidate.State.MODELED,
                    candidate.projectedTtftMs(),
                    OptionalLong.empty(),
                    semantics.afterProbeDetail(),
                    null);
            case UNAVAILABLE -> copyAdmissionResult(
                    candidate,
                    Candidate.State.UNAVAILABLE,
                    OptionalLong.empty(),
                    OptionalLong.empty(),
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
            Candidate source,
            Candidate.State state,
            OptionalLong projectedTtftMs,
            OptionalLong projectedDrainMs,
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
                carriesPending ? source.pendingCount() : OptionalLong.empty());
    }
}
