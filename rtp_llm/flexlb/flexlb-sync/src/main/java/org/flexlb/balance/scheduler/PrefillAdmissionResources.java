package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.DeliveryBlockReason;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.projection.PrefillTtftEstimate;
import org.flexlb.balance.scheduler.decode.DecodeDecideInfo;
import org.flexlb.balance.scheduler.decode.DecodeRequirement;
import org.flexlb.dao.route.RoleType;
import org.flexlb.util.Logger;

import java.util.IdentityHashMap;
import java.util.List;

/**
 * Shared endpoint-capability mechanics for the two delivery transactions.
 *
 * <p>This class deliberately has no dispatcher selection logic. The active
 * transaction decides which Prefill reservation is prepared; this class owns the
 * optional per-request Decode permit, identity validation, and the canonical
 * post-commit cleanup state machine.</p>
 */
final class PrefillAdmissionResources {

    private static final PrefillTtftEstimate.AdmissionBlockSemantics
            DECODE_BLOCK = new PrefillTtftEstimate.AdmissionBlockSemantics(
                    "DELIVERY_CAPACITY_DECODE_ENGINE",
                    PrefillTtftEstimate.AfterProbeAdmission.UNAVAILABLE,
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
        final DecodeEndpoint.EngineDispatchPermit decode;
        MemberOwnership ownership = MemberOwnership.ADMISSION_OWNED;

        Member(ScheduledRequest item, DecodeEndpoint.EngineDispatchPermit permit) {
            this.item = item;
            this.decode = permit;
        }
    }

    static DeliveryBlockReason.Attempt<Member> prepareMember(ScheduledRequest item) {
        DecodeDecideInfo decide = item.decodeDecideInfo();
        if (decide.isAbsent()) {
            // The committed topology has no independent Decode resource.
            // Its Prefill/PDFUSION reservation still follows the same handoff.
            return captureMember(item, null);
        }
        DecodeEndpoint decode = decide.endpoint();
        if (decode == null || decide.reservation() == null) {
            return failed(missingEndpoint("Decode reservation", item));
        }
        DecodeEndpoint.EngineDispatchPermitAcquisition acquisition;
        try {
            acquisition = decode.acquireDispatchPermit(decide.reservation(), decide.requirement().capacity());
        } catch (RuntimeException | Error failure) {
            return failed(failure);
        }
        return switch (acquisition.status()) {
            case ACQUIRED, ALREADY_ACCEPTED -> captureMember(item, acquisition.permit());
            case CAPACITY_FULL -> decodeCapacityFull(item);
            case NOT_OWNED, NOT_QUEUED -> rejected(
                    DeliveryBlockReason.OWNERSHIP_LOST);
            case ENDPOINT_RETIRED -> failed(retired("Decode", item));
            case ALREADY_ACQUIRED -> failed(new IllegalStateException(
                    "Decode dispatch permit already acquired: request_id="
                            + item.requestId()));
        };
    }

    private static DeliveryBlockReason.Attempt<Member> decodeCapacityFull(
            ScheduledRequest item) {
        return rejected(DeliveryBlockReason.unavailable(
                new DecodeAvailability(item),
                DECODE_BLOCK));
    }

    /**
     * Capture the acquired Decode permit into its first owning value. Neither
     * {@link Member} nor the accepted-result wrapper exists before acquisition,
     * so both allocation windows are guarded by the exact permit rollback.
     */
    private static DeliveryBlockReason.Attempt<Member> captureMember(
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

    static <T> DeliveryBlockReason.Attempt<T> rejectedPrefill(
            ScheduledRequest item,
            PrefillState.CapacityStatus status,
            DeliveryBlockReason capacityFull) {
        return switch (status) {
            case CAPACITY_FULL -> rejected(capacityFull);
            case REQUEST_NOT_ACTIVE -> rejected(
                    DeliveryBlockReason.OWNERSHIP_LOST);
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
        return new EndpointRetiredException(role, item);
    }

    static <T> DeliveryBlockReason.Attempt<T> accepted(T value) {
        return DeliveryBlockReason.Attempt.accepted(value);
    }

    static <T> DeliveryBlockReason.Attempt<T> rejected(
            DeliveryBlockReason boundary) {
        return DeliveryBlockReason.Attempt.rejected(boundary);
    }

    static <T> DeliveryBlockReason.Attempt<T> failed(Throwable cause) {
        return rejected(DeliveryBlockReason.failed(cause));
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
        Throwable failure = rollbackPermit(member.decode, priorFailure);
        member.ownership = MemberOwnership.OWNERSHIP_LOST;
        return failure;
    }

    static void preserveRejectedCause(
            Throwable rollbackFailure,
            DeliveryBlockReason boundary) {
        if (boundary.status() == DeliveryBlockReason.Status.FAILED
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
            List<Member> exactMembers) {
        return new CommittedAdmissionOwner(exactMembers);
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
     * Binding the returned handoff only assigns its reference.
     */
    static final class CommittedAdmissionOwner implements AutoCloseable {
        private final IdentityHashMap<ScheduledRequest, Member> membersByIdentity;
        private final Member[] members;
        private PrefillState.CommittedHandoff handoff;
        private boolean bound;
        private boolean closed;

        private CommittedAdmissionOwner(
                List<Member> exactMembers) {
            if (exactMembers.isEmpty()) {
                throw new IllegalArgumentException(
                        "committed admission requires members");
            }
            membersByIdentity = new IdentityHashMap<>(exactMembers.size());
            members = exactMembers.toArray(Member[]::new);
            for (int index = 0; index < members.length; index++) {
                Member member = members[index];
                if (member.ownership != MemberOwnership.ADMISSION_OWNED) {
                    throw new IllegalStateException(
                            "committed admission member is not admission-owned");
                }
                membersByIdentity.put(member.item, member);
            }
        }

        /** No-throw bind after one Prefill group has committed. */
        synchronized void bindPrefillHandoff(
                PrefillState.CommittedHandoff exactHandoff) {
            handoff = exactHandoff;
            bound = true;
        }

        public synchronized boolean transferToEndpoint(ScheduledRequest exactItem) {
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
            if (member.decode == null) {
                member.ownership = MemberOwnership.ENDPOINT_OWNED;
                return true;
            }
            DecodeEndpoint.EngineDispatchPermitTransferStatus transfer =
                    member.decode.dispatch();
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
                    Throwable failure = rollbackMember(member, null);
                    if (failure != null) {
                        Logger.error("Committed admission cleanup isolated", failure);
                    }
                }
            }
            releaseCommittedHandoff(handoff);
        }
    }

    /** Decode is the exact event source for its request-scoped permit. */
    private static final class DecodeAvailability
            implements DeliveryBlockReason.Availability {
        private final DecodeDecideInfo decide;
        private Runnable subscribedListener;

        private DecodeAvailability(ScheduledRequest item) {
            this.decide = item.decodeDecideInfo();
        }

        @Override
        public boolean isAvailable() {
            DecodeRequirement requirement = decide.requirement();
            return decide.endpoint().shouldRetryDispatch(requirement.requestId(), requirement.capacity());
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
            decide.endpoint().addEngineDispatchCapacityListener(listener);
        }

        @Override
        public synchronized void removeListener(Runnable listener) {
            if (subscribedListener != listener) {
                return;
            }
            decide.endpoint().removeEngineDispatchCapacityListener(listener);
            subscribedListener = null;
        }
    }

}
