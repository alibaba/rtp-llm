package org.flexlb.balance.projection;

import org.flexlb.balance.planner.GroupPlanner;
import org.flexlb.balance.prediction.PrefillTimePredictor;
import org.flexlb.dao.route.RoleType;

import java.util.List;
import java.util.OptionalDouble;

import static com.google.common.base.Preconditions.checkArgument;

/** Projects one incoming route against immutable, coherently captured inputs. */
public final class RouteProjection {

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
}
