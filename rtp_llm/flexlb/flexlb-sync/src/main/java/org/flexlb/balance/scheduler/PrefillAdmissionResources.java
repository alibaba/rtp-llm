package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.delivery.DeliveryItem;
import org.flexlb.balance.delivery.PrefillAdmissionPort;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.EndpointGenerationRetiredException;
import org.flexlb.balance.endpoint.PrefillWorkLedger;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.util.Logger;

import java.util.IdentityHashMap;
import java.util.List;
import java.util.Objects;

/**
 * Shared endpoint-capability mechanics for the two dispatcher-specific
 * admission adapters.
 *
 * <p>This class deliberately has no dispatcher selection logic. The active
 * adapter decides which Prefill reservation is prepared; this class owns only the
 * exact per-request Decode permit, identity validation, and the canonical
 * post-commit cleanup state machine.</p>
 */
final class PrefillAdmissionResources {

    private static final RouteProjection.AdmissionBlockSemantics
            DECODE_BLOCK = new RouteProjection.AdmissionBlockSemantics(
                    "DELIVERY_CAPACITY_DECODE_ENGINE",
                    RouteProjection.AfterProbeAdmission.UNAVAILABLE,
                    "DECODE_CAPACITY_SCOPE_UNKNOWN");
    private PrefillAdmissionResources() {
    }

    enum PreparedState {
        PREPARED,
        COMMITTED,
        CLOSED
    }

    enum MemberState {
        ADMISSION_OWNED,
        ENDPOINT_OWNED,
        OWNERSHIP_LOST
    }

    static final class Member {
        final BatchItem item;
        final DecodeEndpoint.EngineDispatchPermit permit;
        MemberState state = MemberState.ADMISSION_OWNED;

        Member(
                BatchItem item,
                DecodeEndpoint.EngineDispatchPermit permit) {
            this.item = Objects.requireNonNull(item, "item");
            this.permit = Objects.requireNonNull(permit, "permit");
        }
    }

    static CapacityBoundary.Attempt<Member> prepareMember(
            BatchItem item) {
        DecodeEndpoint decode = item.decodeEp();
        if (decode == null || item.decodeReservation() == null) {
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
            case CAPACITY_FULL -> rejected(new CapacityBoundary.Unavailable(
                    new DecodeAvailability(item), DECODE_BLOCK));
            case NOT_OWNED, NOT_QUEUED -> rejected(
                    CapacityBoundary.OwnershipLost.INSTANCE);
            case ENDPOINT_RETIRED -> failed(retired("Decode", item));
            case ALREADY_ACQUIRED -> failed(new IllegalStateException(
                    "Decode dispatch permit already acquired: request_id="
                            + item.requestId()));
        };
    }

    /**
     * Capture the acquired Decode permit into its first owning value. Neither
     * {@link Member} nor the accepted-result wrapper exists before acquisition,
     * so both allocation windows are guarded by the exact permit rollback.
     */
    private static CapacityBoundary.Attempt<Member> captureMember(
            BatchItem item,
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
            BatchItem item,
            PrefillWorkLedger.CapacityStatus status,
            CapacityBoundary.Unavailable capacityFull) {
        return switch (status) {
            case CAPACITY_FULL -> rejected(capacityFull);
            case REQUEST_NOT_ACTIVE -> rejected(
                    CapacityBoundary.OwnershipLost.INSTANCE);
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
            BatchItem item) {
        return new IllegalStateException(
                role + " is unavailable: request_id=" + item.requestId());
    }

    static EndpointGenerationRetiredException retired(
            String role,
            BatchItem item) {
        return new EndpointGenerationRetiredException(
                role + " endpoint generation retired: request_id="
                        + item.requestId());
    }

    static <T> CapacityBoundary.Attempt<T> accepted(T value) {
        return new CapacityBoundary.Attempt.Accepted<>(value);
    }

    static <T> CapacityBoundary.Attempt<T> rejected(
            CapacityBoundary boundary) {
        return new CapacityBoundary.Attempt.Rejected<>(boundary);
    }

    static <T> CapacityBoundary.Attempt<T> failed(Throwable cause) {
        return rejected(new CapacityBoundary.Failed(cause));
    }

    static Throwable rollbackReservation(
            PrefillWorkLedger.Reservation reservation,
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
        if (member == null || member.state != MemberState.ADMISSION_OWNED) {
            return priorFailure;
        }
        Throwable failure = rollbackPermit(member.permit, priorFailure);
        member.state = MemberState.OWNERSHIP_LOST;
        return failure;
    }

    /**
     * Restore a transaction list to the size it had before one append was
     * prepared. Every call site holds the transaction monitor, so only the
     * current append can own entries beyond {@code originalSize}.
     */
    static Throwable restorePreparedTail(
            List<?> values,
            int originalSize,
            Throwable priorFailure) {
        try {
            while (values.size() > originalSize) {
                values.remove(values.size() - 1);
            }
            assert values.size() == originalSize
                    : "prepared admission list shrank below its append boundary";
        } catch (Throwable failure) {
            return combine(priorFailure, failure);
        }
        return priorFailure;
    }

    static void preserveRejectedCause(
            Throwable rollbackFailure,
            CapacityBoundary boundary) {
        if (boundary instanceof CapacityBoundary.Failed failed
                && failed.cause() != rollbackFailure) {
            rollbackFailure.addSuppressed(failed.cause());
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
            List<DeliveryItem> supplied) {
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

    static CommittedAdmissionAdapter prepareCommitted(
            List<Member> exactMembers,
            int committedHandoffCount) {
        return new CommittedAdmissionAdapter(
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
            PrefillWorkLedger.CommittedHandoff handoff) {
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
    static final class CommittedAdmissionAdapter
            implements PrefillAdmissionPort.CommittedAdmission {
        private final IdentityHashMap<DeliveryItem, Member> membersByIdentity;
        private final Member[] members;
        private final DecodeEndpoint.EngineDispatchPermit[] permits;
        private final PrefillWorkLedger.CommittedHandoff[] handoffs;
        private boolean bound;
        private boolean closed;

        private CommittedAdmissionAdapter(
                List<Member> exactMembers,
                int committedHandoffCount) {
            assert !exactMembers.isEmpty() && committedHandoffCount > 0
                    : "committed admission requires members and handoffs";
            membersByIdentity = new IdentityHashMap<>(exactMembers.size());
            members = exactMembers.toArray(Member[]::new);
            permits = new DecodeEndpoint.EngineDispatchPermit[members.length];
            handoffs = new PrefillWorkLedger.CommittedHandoff[
                    committedHandoffCount];
            for (int index = 0; index < members.length; index++) {
                Member member = members[index];
                assert member.state == MemberState.ADMISSION_OWNED
                        : "committed admission member is not admission-owned";
                membersByIdentity.put(member.item, member);
                permits[index] = member.permit;
            }
        }

        /** No-throw bind after one batch reservation has committed. */
        synchronized void bind(
                PrefillWorkLedger.CommittedHandoff exactHandoff) {
            handoffs[0] = exactHandoff;
            bound = true;
        }

        /**
         * No-allocation bind after a route group has committed. The ledger
         * contract guarantees one materialized handoff per exact reservation.
         */
        synchronized void bind(
                List<PrefillWorkLedger.CommittedHandoff> exactHandoffs) {
            for (int index = 0; index < handoffs.length; index++) {
                handoffs[index] = exactHandoffs.get(index);
            }
            bound = true;
        }

        @Override
        public synchronized boolean transfer(
                DeliveryItem exactItem,
                Runnable pointOfNoReturn) {
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
            if (member.state != MemberState.ADMISSION_OWNED) {
                throw new IllegalStateException(
                        "committed item was already transferred");
            }
            Objects.requireNonNull(pointOfNoReturn, "pointOfNoReturn");
            DecodeEndpoint.EngineDispatchPermitTransferStatus transfer =
                    member.permit.transferToEngineLifecycle();
            return switch (transfer) {
                case TRANSFERRED -> {
                    member.state = MemberState.ENDPOINT_OWNED;
                    pointOfNoReturn.run();
                    yield true;
                }
                case OWNERSHIP_LOST -> {
                    member.state = MemberState.OWNERSHIP_LOST;
                    yield false;
                }
                case ENDPOINT_RETIRED -> {
                    member.state = MemberState.OWNERSHIP_LOST;
                    throw retired("Decode", member.item);
                }
            };
        }

        @Override
        public void close() {
            synchronized (this) {
                if (closed) {
                    return;
                }
                closed = true;
            }
            for (int index = 0; index < members.length; index++) {
                Member member = members[index];
                if (member.state == MemberState.ADMISSION_OWNED) {
                    releaseCommittedPermit(permits[index]);
                    member.state = MemberState.OWNERSHIP_LOST;
                }
            }
            for (PrefillWorkLedger.CommittedHandoff handoff
                    : handoffs) {
                releaseCommittedHandoff(handoff);
            }
        }
    }

    /** Decode is the exact event source for its request-scoped permit. */
    private static final class DecodeAvailability
            implements CapacityBoundary.Availability {
        private final BatchItem item;

        private DecodeAvailability(BatchItem item) {
            this.item = item;
        }

        @Override
        public boolean isAvailable() {
            return item.decodeEp().isEngineDispatchPermitAvailable(
                    item.requestId(),
                    item.maxDecodeEngineRequests(),
                    item.maxDecodeKvUsagePercent());
        }

        @Override
        public void addListener(Runnable listener) {
            item.decodeEp().addEngineDispatchCapacityListener(listener);
        }

        @Override
        public void removeListener(Runnable listener) {
            item.decodeEp().removeEngineDispatchCapacityListener(listener);
        }
    }

}
