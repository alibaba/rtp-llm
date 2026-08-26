package org.flexlb.balance.projection;

import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.prediction.PrefillTimePredictor;

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
            String afterProbeDetail) {
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

    /** Structured outcome of placing a virtual request into a frozen snapshot. */
    public record Result(
            State state,
            OptionalLong projectedTtftMs,
            OptionalLong projectedDrainMs,
            long incomingPrefillMs,
            InitialHeadDisposition initialHeadDisposition,
            String detail) {

        public Result {
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
            detail = detail == null ? "" : detail;
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

        public boolean selectable() {
            return state == State.MODELED && projectedTtftMs.isPresent();
        }

        public boolean engineWorkUnmodeled() {
            return state == State.UNMODELED_ENGINE_WORK;
        }

        private static OptionalLong requireNonNegative(
                OptionalLong value, String name) {
            if (value.isPresent() && value.getAsLong() < 0L) {
                throw new IllegalArgumentException(name + " must be non-negative");
            }
            return value;
        }
    }

    /** Pure value produced for one route candidate. */
    public record Candidate(
            Result projection,
            long cacheHitTokens,
            long routingCacheMatchTokens,
            OptionalLong pendingCount) {

        public Candidate {
            if (cacheHitTokens < 0L || routingCacheMatchTokens < 0L) {
                throw new IllegalArgumentException(
                        "cache token counts must be non-negative");
            }
            boolean carriesPendingCount = projection.selectable()
                    || projection.engineWorkUnmodeled();
            if (carriesPendingCount != pendingCount.isPresent()) {
                throw new IllegalArgumentException(
                        "only modeled or Engine-unmodeled projections may carry pending count");
            }
            if (pendingCount.isPresent() && pendingCount.getAsLong() < 0L) {
                throw new IllegalArgumentException(
                        "pendingCount must be non-negative");
            }
        }

        public long projectedTtftMs() {
            return projection.projectedTtftMs().orElseThrow(
                    () -> new IllegalStateException("candidate TTFT is unknown"));
        }

        public OptionalLong projectedDrainMs() {
            return projection.projectedDrainMs();
        }

        public long incomingPrefillMs() {
            return projection.incomingPrefillMs();
        }

        public long requiredPendingCount() {
            return pendingCount.orElseThrow(
                    () -> new IllegalStateException(
                            "candidate pending count is unknown"));
        }

        public boolean engineWorkUnmodeled() {
            return projection.engineWorkUnmodeled();
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
        Result timeline = RouteTimelineProjector.project(
                inputs.queue(), inputs.work(), probe, evaluator,
                deliveryProjection);
        Result projection = RouteAdmissionPolicy.apply(inputs.queue(), timeline);
        return new Candidate(
                projection,
                probe.hitCache(),
                probe.routingCacheMatchTokens(),
                (projection.selectable() || projection.engineWorkUnmodeled())
                        ? OptionalLong.of(inputs.pendingRequestCount())
                        : OptionalLong.empty());
    }
}
