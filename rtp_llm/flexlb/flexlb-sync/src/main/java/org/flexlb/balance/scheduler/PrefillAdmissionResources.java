package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.balance.PlacementResult;
import org.flexlb.balance.strategy.SelectedRole;
import org.flexlb.dao.route.RoleType;
import org.flexlb.util.Logger;

import java.util.IdentityHashMap;
import java.util.List;
import java.util.Objects;

/**
 * Shared endpoint-capability mechanics for the two delivery transactions.
 *
 * <p>This class deliberately has no dispatcher selection logic. The active
 * transaction decides which Prefill reservation is prepared; this class owns the
 * exact per-request Decode permit, identity validation, and the canonical
 * post-commit cleanup state machine.</p>
 */
final class PrefillAdmissionResources {

    private static final RouteProjection.AdmissionBlockSemantics
            DECODE_BLOCK = new RouteProjection.AdmissionBlockSemantics(
                    "DELIVERY_CAPACITY_DECODE_ENGINE",
                    RouteProjection.AfterProbeAdmission.UNAVAILABLE,
                    "DECODE_CAPACITY_SCOPE_UNKNOWN",
                    RoleType.DECODE);
    private PrefillAdmissionResources() {
    }

    enum MemberOwnership {
        ADMISSION_OWNED,
        ENDPOINT_OWNED,
        OWNERSHIP_LOST
    }

    static final class Member {
        final ScheduledRequest item;
        final DecodeEndpoint.EngineDispatchPermit permit;
        MemberOwnership ownership = MemberOwnership.ADMISSION_OWNED;

        Member(
                ScheduledRequest item,
                DecodeEndpoint.EngineDispatchPermit permit) {
            this.item = Objects.requireNonNull(item, "item");
            this.permit = Objects.requireNonNull(permit, "permit");
        }
    }

    static CapacityBoundary.Attempt<Member> prepareMember(
            ScheduledRequest item) {
        ScheduledRequest.DecodeBinding binding = item.decodeBinding();
        DecodeEndpoint decode = binding.endpoint();
        if (decode == null || binding.reservation() == null) {
            return failed(missingEndpoint("Decode reservation", item));
        }
        DecodeEndpoint.EngineDispatchPermitAcquisition acquisition;
        try {
            acquisition = decode.acquireEngineDispatchPermit(
                    item.requestId(),
                    item.maxDecodeEngineRequests(),
                    item.maxDecodeKvUsagePercent());
        } catch (RuntimeException | Error failure) {
            return failed(failure);
        }
        return switch (acquisition.status()) {
            case ACQUIRED -> captureMember(
                    item, acquisition.permit());
            case CAPACITY_FULL -> prepareReplacementMember(
                    item,
                    binding,
                    decodePoolSequence(item));
            case NOT_OWNED, NOT_QUEUED -> rejected(
                    CapacityBoundary.OWNERSHIP_LOST);
            case ENDPOINT_RETIRED -> failed(retired("Decode", item));
            case ALREADY_ACQUIRED -> failed(new IllegalStateException(
                    "Decode dispatch permit already acquired: request_id="
                            + item.requestId()));
        };
    }

    /**
     * Preserve the Prefill queue head while moving only its queued Decode
     * capability. Selection is advisory; the replacement endpoint creates the
     * reservation and hard-gate permit in one admission-lock transaction.
     */
    private static CapacityBoundary.Attempt<Member> prepareReplacementMember(
            ScheduledRequest item,
            ScheduledRequest.DecodeBinding original,
            long observedPoolSequence) {
        PlacementResult<SelectedRole, RoleType> selection;
        try {
            selection = item.selectDecodeForDispatch();
        } catch (RuntimeException | Error failure) {
            return failed(failure);
        }
        if (selection.status() != PlacementResult.Status.SUCCESS) {
            return decodeCapacityFull(item, observedPoolSequence);
        }

        SelectedRole replacement = selection.value();
        WorkerEndpoint.GenerationPin pin = null;
        DecodeEndpoint candidate = null;
        DecodeEndpoint.EngineDispatchPermitAcquisition acquisition = null;
        try (replacement) {
            pin = replacement.takeGenerationPin();
            if (!(pin.endpoint() instanceof DecodeEndpoint exactCandidate)) {
                return failed(new IllegalStateException(
                        "Decode reselection returned another endpoint type"));
            }
            candidate = exactCandidate;
            if (candidate == original.endpoint()) {
                return decodeCapacityFull(item, observedPoolSequence);
            }

            long hardKv = Math.max(0L, item.seqLen());
            long expectedKv = item.ctx().getConfig()
                    .decodeKvReservationTokens(
                            hardKv,
                            item.ctx().getRequest().getMaxNewTokens(),
                            replacement.decodeTotalKv());
            acquisition = candidate
                    .tryAcquireQueuedEngineDispatchPermitPinned(
                            pin,
                            item.requestId(),
                            hardKv,
                            Math.max(hardKv, expectedKv),
                            item.priority(),
                            item.maxDecodeEngineRequests(),
                            item.maxDecodeKvUsagePercent());
            if (acquisition.status()
                    != DecodeEndpoint.EngineDispatchPermitAcquireStatus.ACQUIRED) {
                return switch (acquisition.status()) {
                    case CAPACITY_FULL, NOT_OWNED, NOT_QUEUED,
                            ALREADY_ACQUIRED -> decodeCapacityFull(
                                    item, observedPoolSequence);
                    case ENDPOINT_RETIRED -> failed(retired("Decode", item));
                    case ACQUIRED -> throw new IllegalStateException(
                            "acquired replacement has no ownership");
                };
            }

            ScheduledRequest.DecodeBinding rebound = new ScheduledRequest.DecodeBinding(
                    RequestRegistry.copyOf(
                            replacement.serverStatus()),
                    candidate,
                    acquisition.reservation());
            if (!item.replaceDecodeBinding(original, rebound)) {
                Throwable cleanup = rollbackReplacement(
                        candidate, acquisition, null);
                if (cleanup != null) {
                    return failed(cleanup);
                }
                return rejected(CapacityBoundary.OWNERSHIP_LOST);
            }

            try {
                original.endpoint().releaseReservationExact(
                        original.reservation());
            } catch (Throwable oldReleaseFailure) {
                Throwable failure = oldReleaseFailure;
                if (!item.replaceDecodeBinding(rebound, original)) {
                    failure.addSuppressed(new IllegalStateException(
                            "Decode binding changed while migration rolled back"));
                }
                failure = rollbackReplacement(
                        candidate, acquisition, failure);
                return failed(failure);
            }
            return captureMember(item, acquisition.permit());
        } catch (RuntimeException | Error failure) {
            if (acquisition != null
                    && acquisition.status()
                            == DecodeEndpoint.EngineDispatchPermitAcquireStatus.ACQUIRED
                    && item.decodeBinding() == original) {
                return failed(rollbackReplacement(
                        candidate, acquisition, failure));
            }
            return failed(failure);
        } finally {
            if (pin != null) {
                pin.close();
            }
        }
    }

    private static long decodePoolSequence(ScheduledRequest item) {
        PlacementAvailability pool = item.decodePlacementAvailability();
        return pool == null ? 0L : pool.sequence();
    }

    private static CapacityBoundary.Attempt<Member> decodeCapacityFull(
            ScheduledRequest item,
            long observedPoolSequence) {
        return rejected(CapacityBoundary.unavailable(
                new DecodeAvailability(item, observedPoolSequence),
                DECODE_BLOCK));
    }

    private static Throwable rollbackReplacement(
            DecodeEndpoint endpoint,
            DecodeEndpoint.EngineDispatchPermitAcquisition acquisition,
            Throwable priorFailure) {
        Throwable failure = rollbackPermit(
                acquisition == null ? null : acquisition.permit(),
                priorFailure);
        if (endpoint == null || acquisition == null
                || acquisition.reservation() == null) {
            return failure;
        }
        try {
            endpoint.releaseReservationExact(acquisition.reservation());
        } catch (Throwable rollbackFailure) {
            failure = combine(failure, rollbackFailure);
        }
        return failure;
    }

    /**
     * Capture the acquired Decode permit into its first owning value. Neither
     * {@link Member} nor the accepted-result wrapper exists before acquisition,
     * so both allocation windows are guarded by the exact permit rollback.
     */
    private static CapacityBoundary.Attempt<Member> captureMember(
            ScheduledRequest item,
            DecodeEndpoint.EngineDispatchPermit permit) {
        Member member = null;
        try {
            member = new Member(item, permit);
            return accepted(member);
        } catch (Throwable captureFailure) {
            Throwable failure = member == null
                    ? rollbackPermit(permit, captureFailure)
                    : rollbackMember(member, captureFailure);
            return failed(failure);
        }
    }

    static <T> CapacityBoundary.Attempt<T> rejectedPrefill(
            ScheduledRequest item,
            PrefillState.CapacityStatus status,
            CapacityBoundary capacityFull) {
        return switch (status) {
            case CAPACITY_FULL -> rejected(capacityFull);
            case REQUEST_NOT_ACTIVE -> rejected(
                    CapacityBoundary.OWNERSHIP_LOST);
            case ENDPOINT_RETIRED -> failed(retired("Prefill", item));
            case REQUEST_ALREADY_RESERVED, BATCH_ID_ALREADY_RESERVED ->
                    failed(new IllegalStateException(
                            "Prefill admission owns another reservation: "
                                    + "request_id=" + item.requestId()
                                    + " status=" + status));
            case ACQUIRED -> throw new IllegalArgumentException(
                    "ACQUIRED must carry a reservation");
        };
    }

    static IllegalStateException missingEndpoint(
            String role,
            ScheduledRequest item) {
        return new IllegalStateException(
                role + " is unavailable: request_id=" + item.requestId());
    }

    static IllegalStateException retired(
            String role,
            ScheduledRequest item) {
        return new IllegalStateException(
                role + " endpoint generation retired: request_id="
                        + item.requestId());
    }

    static <T> CapacityBoundary.Attempt<T> accepted(T value) {
        return CapacityBoundary.Attempt.accepted(value);
    }

    static <T> CapacityBoundary.Attempt<T> rejected(
            CapacityBoundary boundary) {
        return CapacityBoundary.Attempt.rejected(boundary);
    }

    static <T> CapacityBoundary.Attempt<T> failed(Throwable cause) {
        return rejected(CapacityBoundary.failed(cause));
    }

    static Throwable rollbackReservation(
            PrefillState.Reservation reservation,
            Throwable priorFailure) {
        if (reservation == null) {
            return priorFailure;
        }
        try {
            reservation.close();
        } catch (Throwable failure) {
            return combine(priorFailure, failure);
        }
        return priorFailure;
    }

    static Throwable rollbackPermit(
            DecodeEndpoint.EngineDispatchPermit permit,
            Throwable priorFailure) {
        if (permit == null) {
            return priorFailure;
        }
        try {
            permit.release();
        } catch (Throwable failure) {
            return combine(priorFailure, failure);
        }
        return priorFailure;
    }

    static Throwable rollbackMember(
            Member member,
            Throwable priorFailure) {
        if (member == null
                || member.ownership != MemberOwnership.ADMISSION_OWNED) {
            return priorFailure;
        }
        Throwable failure = rollbackPermit(member.permit, priorFailure);
        member.ownership = MemberOwnership.OWNERSHIP_LOST;
        return failure;
    }

    static void preserveRejectedCause(
            Throwable rollbackFailure,
            CapacityBoundary boundary) {
        if (boundary.status() == CapacityBoundary.Status.FAILED
                && boundary.cause() != rollbackFailure) {
            rollbackFailure.addSuppressed(boundary.cause());
        }
    }

    static void throwRollbackFailure(Throwable failure) {
        if (failure instanceof RuntimeException runtime) {
            throw runtime;
        }
        if (failure instanceof Error error) {
            throw error;
        }
        throw new IllegalStateException(
                "admission rollback failed", failure);
    }

    static boolean sameIdentitySequence(
            List<Member> expected,
            List<ScheduledRequest> supplied) {
        if (expected.size() != supplied.size()) {
            return false;
        }
        for (int index = 0; index < expected.size(); index++) {
            if (expected.get(index).item != supplied.get(index)) {
                return false;
            }
        }
        return true;
    }

    static CommittedAdmissionOwner createCommittedOwner(
            List<Member> exactMembers,
            int committedHandoffCount) {
        return new CommittedAdmissionOwner(
                exactMembers, committedHandoffCount);
    }

    private static void releaseCommittedPermit(
            DecodeEndpoint.EngineDispatchPermit permit) {
        try {
            permit.release();
        } catch (Throwable failure) {
            Logger.error(
                    "Committed Decode permit cleanup isolated", failure);
        }
    }

    private static void releaseCommittedHandoff(
            PrefillState.CommittedHandoff handoff) {
        if (handoff == null) {
            return;
        }
        try {
            handoff.close();
        } catch (Throwable failure) {
            Logger.error(
                    "Committed Prefill handoff cleanup isolated", failure);
        }
    }

    private static Throwable combine(
            Throwable aggregate,
            Throwable next) {
        if (aggregate == null) {
            return next;
        }
        if (aggregate != next) {
            aggregate.addSuppressed(next);
        }
        return aggregate;
    }

    /**
     * Fully allocated before the ledger crosses its canonical commit point.
     * Binding a returned handoff writes only preallocated reference slots.
     */
    static final class CommittedAdmissionOwner implements AutoCloseable {
        private final IdentityHashMap<ScheduledRequest, Member> membersByIdentity;
        private final Member[] members;
        private final DecodeEndpoint.EngineDispatchPermit[] permits;
        private final PrefillState.CommittedHandoff[] handoffs;
        private boolean bound;
        private boolean closed;

        private CommittedAdmissionOwner(
                List<Member> exactMembers,
                int committedHandoffCount) {
            if (exactMembers.isEmpty() || committedHandoffCount <= 0) {
                throw new IllegalArgumentException(
                        "committed admission requires members and handoffs");
            }
            membersByIdentity = new IdentityHashMap<>(exactMembers.size());
            members = exactMembers.toArray(Member[]::new);
            permits = new DecodeEndpoint.EngineDispatchPermit[members.length];
            handoffs = new PrefillState.CommittedHandoff[
                    committedHandoffCount];
            for (int index = 0; index < members.length; index++) {
                Member member = members[index];
                if (member.ownership != MemberOwnership.ADMISSION_OWNED) {
                    throw new IllegalStateException(
                            "committed admission member is not admission-owned");
                }
                membersByIdentity.put(member.item, member);
                permits[index] = member.permit;
            }
        }

        /** No-throw bind after one batch reservation has committed. */
        synchronized void bindBatchHandoff(
                PrefillState.CommittedHandoff exactHandoff) {
            handoffs[0] = exactHandoff;
            bound = true;
        }

        /**
         * No-allocation bind after a route group has committed. The ledger
         * contract guarantees one materialized handoff per exact reservation.
         */
        synchronized void bindRouteHandoffs(
                List<PrefillState.CommittedHandoff> exactHandoffs) {
            for (int index = 0; index < handoffs.length; index++) {
                handoffs[index] = exactHandoffs.get(index);
            }
            bound = true;
        }

        public synchronized boolean transferToEndpoint(
                ScheduledRequest exactItem) {
            if (closed) {
                throw new IllegalStateException(
                        "committed admission is closed");
            }
            if (!bound) {
                throw new IllegalStateException(
                        "committed admission is not bound");
            }
            Member member = membersByIdentity.get(exactItem);
            if (member == null) {
                throw new IllegalArgumentException(
                        "item does not belong to committed admission");
            }
            if (member.ownership != MemberOwnership.ADMISSION_OWNED) {
                throw new IllegalStateException(
                        "committed item was already transferred");
            }
            DecodeEndpoint.EngineDispatchPermitTransferStatus transfer =
                    member.permit.transferToEngineLifecycle();
            return switch (transfer) {
                case TRANSFERRED -> {
                    member.ownership = MemberOwnership.ENDPOINT_OWNED;
                    yield true;
                }
                case OWNERSHIP_LOST -> {
                    member.ownership = MemberOwnership.OWNERSHIP_LOST;
                    yield false;
                }
                case ENDPOINT_RETIRED -> {
                    member.ownership = MemberOwnership.OWNERSHIP_LOST;
                    throw retired("Decode", member.item);
                }
            };
        }

        public void close() {
            synchronized (this) {
                if (closed) {
                    return;
                }
                closed = true;
            }
            for (int index = 0; index < members.length; index++) {
                Member member = members[index];
                if (member.ownership == MemberOwnership.ADMISSION_OWNED) {
                    releaseCommittedPermit(permits[index]);
                    member.ownership = MemberOwnership.OWNERSHIP_LOST;
                }
            }
            for (PrefillState.CommittedHandoff handoff
                    : handoffs) {
                releaseCommittedHandoff(handoff);
            }
        }
    }

    /** Decode is the exact event source for its request-scoped permit. */
    private static final class DecodeAvailability
            implements CapacityBoundary.Availability {
        private final ScheduledRequest item;
        private final DecodeEndpoint endpoint;
        private final PlacementAvailability pool;
        private final PlacementKey poolKey;
        private final long observedPoolSequence;
        private Runnable subscribedListener;
        private PlacementAvailability.Listener poolListener;

        private DecodeAvailability(
                ScheduledRequest item,
                long observedPoolSequence) {
            this.item = item;
            ScheduledRequest.DecodeBinding binding = item.decodeBinding();
            this.endpoint = binding.endpoint();
            this.pool = item.decodePlacementAvailability();
            String group = binding.status() == null
                    ? null : binding.status().getGroup();
            this.poolKey = new PlacementKey(
                    org.flexlb.dao.route.RoleType.DECODE, group);
            this.observedPoolSequence = observedPoolSequence;
        }

        @Override
        public boolean isAvailable() {
            return endpoint.isEngineDispatchPermitAvailable(
                    item.requestId(),
                    item.maxDecodeEngineRequests(),
                    item.maxDecodeKvUsagePercent())
                    || pool != null && pool.lastChangedSequence(poolKey)
                            > observedPoolSequence;
        }

        @Override
        public synchronized void addListener(Runnable listener) {
            if (subscribedListener != null && subscribedListener != listener) {
                throw new IllegalStateException(
                        "Decode availability already has a listener");
            }
            if (subscribedListener == listener) {
                return;
            }
            subscribedListener = listener;
            endpoint.addEngineDispatchCapacityListener(listener);
            if (pool != null) {
                poolListener = (key, ignoredSequence) -> {
                    if (poolKey.equals(key)) {
                        listener.run();
                    }
                };
                pool.addListener(poolListener);
            }
        }

        @Override
        public synchronized void removeListener(Runnable listener) {
            if (subscribedListener != listener) {
                return;
            }
            endpoint.removeEngineDispatchCapacityListener(listener);
            if (pool != null && poolListener != null) {
                pool.removeListener(poolListener);
            }
            poolListener = null;
            subscribedListener = null;
        }
    }

}
