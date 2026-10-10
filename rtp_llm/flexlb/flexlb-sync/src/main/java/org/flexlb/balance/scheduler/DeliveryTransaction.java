package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.CapacityBoundary;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.projection.RouteProjection;
import org.flexlb.balance.scheduler.DefaultBatchDispatcher.PreparedSubmission;
import org.flexlb.dao.route.RoleType;
import org.flexlb.util.Failures;
import org.flexlb.util.Logger;

import java.util.AbstractList;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.function.LongSupplier;
import java.util.function.Supplier;

import static com.google.common.base.Preconditions.checkState;
import static org.flexlb.balance.delivery.CapacityBoundary.Attempt.accepted;
import static org.flexlb.balance.delivery.CapacityBoundary.Attempt.rejected;

/** Owns prepared members, then the committed generation handoff, until delivery resolves them. */
public final class DeliveryTransaction implements AutoCloseable {
    // SUBMITTED is used only by BATCH: abort/close must not reclaim executor-owned work.
    private enum Phase { PREPARING, PREPARED, COMMITTED, SUBMITTED, CLOSED }

    /** Resources that exist only for an executor-backed batch. */
    static final class BatchResources {
        long batchId;
        long predictedMs;
        PreparedSubmission submission;
        PrefillState.BatchReservation reservation;
    }

    final BatchResources batch;
    final long[] routePredictions;
    final ArrayList<Member> members;
    private final List<RequestRoute> items = new AbstractList<>() {
        @Override public RequestRoute get(int index) { return members.get(index).route(); }
        @Override public int size() { return members.size(); }
    };
    private PrefillEndpoint prefill;
    private PrefillState.CommittedHandoff committed;
    RequestRoute blockedItem;
    CapacityBoundary blockedResult;
    private volatile Phase phase = Phase.PREPARING;

    DeliveryTransaction(List<RequestRoute> candidates, boolean batchDelivery) {
        members = new ArrayList<>(candidates.size());
        batch = batchDelivery ? new BatchResources() : null;
        routePredictions = batch == null ? new long[candidates.size()] : null;
        if (batch == null) { prefill = candidates.getFirst().prefillEp(); }
    }

    void finishPreparation() { if (!members.isEmpty()) { phase = Phase.PREPARED; } }

    /** Called under the exact request monitor; null means the member is prepared. */
    synchronized CapacityBoundary append(RequestRoute exact, long prediction,
            Supplier<CapacityBoundary.Attempt<PreparedSubmission>> prepareSubmission, LongSupplier batchIds) {
        requirePhase(Phase.PREPARING, "append");
        Member acquired = null;
        try {
            if (batch != null && members.isEmpty()) {
                try {
                    var attempt = Objects.requireNonNull(prepareSubmission.get(), "submission attempt");
                    if (!attempt.accepted()) { return attempt.boundary(); }
                    batch.submission = attempt.value();
                    batch.batchId = batchIds.getAsLong();
                    checkState(batch.batchId > 0L, "batch id supplier returned a non-positive id");
                    prefill = exact.prefillEp();
                } catch (Throwable failure) {
                    return CapacityBoundary.failed(Failures.run(failure, this::closeSubmission));
                }
                var result = prefill.reserveBatch(exact, batch.batchId,
                        exact.requirements().maxInflightBatchesPerPrefillWorker());
                if (result.status() != PrefillState.CapacityStatus.ACQUIRED) {
                    return rejectedPrefill(exact, result.status(), CapacityBoundary.deliveryUnavailable(
                            prefill.batchAdmissionAvailability(exact.requirements().maxInflightBatchesPerPrefillWorker())));
                }
                batch.reservation = result.reservation();
            }
            var attempt = prepareMember(exact);
            if (!attempt.accepted()) { return attempt.boundary(); }
            acquired = attempt.value();
            if (routePredictions != null) { routePredictions[members.size()] = prediction; }
            members.add(acquired);
            return null;
        } catch (Throwable failure) {
            return CapacityBoundary.failed(Failures.run(failure, acquired == null ? null : acquired::close));
        }
    }

    public List<RequestRoute> items() { return items; }
    public RequestRoute blockedItem() { return blockedItem; }
    public CapacityBoundary blockedResult() { return blockedResult; }

    /** Caller holds Prefill's ownership lock. Null means selection changed without committing. */
    public synchronized PrefillState.CommittedHandoff commitSelectionLocked(long nowMs) {
        requirePhase(Phase.PREPARED, "commit");
        RequestRoute failed = blockedResult != null && blockedResult.status() == CapacityBoundary.Status.FAILED
                ? blockedItem : null;
        committed = batch != null
                ? batch.reservation.commitLocked(items, batch.predictedMs, failed, nowMs)
                : prefill.commitQueuedRoutesLocked(items, Arrays.copyOf(routePredictions, members.size()), failed, nowMs);
        if (committed != null) {
            if (batch != null) { batch.reservation = null; }
            prefill = null;
            phase = Phase.COMMITTED;
        }
        return committed;
    }

    synchronized PrefillState.CommittedHandoff takeCommitted() {
        requirePhase(Phase.COMMITTED, "deliver");
        phase = Phase.CLOSED;
        var handoff = committed;
        committed = null;
        return handoff;
    }

    synchronized PreparedSubmission transferToExecutor() {
        requirePhase(Phase.COMMITTED, "submit delivery");
        phase = Phase.SUBMITTED;
        return batch.submission;
    }

    void requireSubmitted() { requirePhase(Phase.SUBMITTED, "deliver"); }

    public boolean batchDelivery() { return batch != null; }

    /** Only the executor may reclaim SUBMITTED work; the caller can reclaim an unsubmitted commit. */
    public synchronized boolean tryReclaimUnsent(boolean executorOwner) {
        if (phase != (executorOwner ? Phase.SUBMITTED : Phase.COMMITTED)) { return false; }
        phase = Phase.CLOSED;
        return true;
    }

    @Override
    public void close() {
        synchronized (this) {
            if (phase != Phase.PREPARING && phase != Phase.PREPARED) { return; }
            phase = Phase.CLOSED;
            if (batch != null) {
                rollbackPrepared();
                return;
            }
        }
        rollbackPrepared();
    }

    private void rollbackPrepared() {
        Throwable failure = null;
        try {
            for (var member : members) { failure = Failures.run(failure, member::close); }
            if (batch != null && batch.reservation != null) {
                failure = Failures.run(failure, () -> prefill.rollbackReservation(batch.reservation));
                batch.reservation = null;
            }
        } finally {
            if (batch != null) { failure = Failures.run(failure, this::closeSubmission); }
        }
        Failures.rethrow(failure, batch == null ? "admission rollback failed" : "batch delivery failed");
    }

    public void closeSubmission() {
        if (batch == null) { return; }
        var submission = batch.submission;
        if (submission != null) {
            batch.submission = null;
            submission.close();
        }
    }

    /** Close temporary ownership before DispatchGate is allowed to process early callbacks. */
    public Throwable finishDelivery() {
        Throwable cleanup = Failures.run(null, () -> {
            var handoff = committed;
            committed = null;
            if (handoff != null) { closeCommitted(members, handoff); }
        });
        try {
            return Failures.run(cleanup, this::closeSubmission);
        } finally {
            phase = Phase.CLOSED;
        }
    }

    private void requirePhase(Phase expected, String operation) {
        if (batch == null) {
            checkState(phase == expected, "expected %s route admission, was %s", expected, phase);
        } else {
            checkState(phase == expected, "cannot %s batch transaction in %s", operation, phase);
        }
    }
    private static final RouteProjection.AdmissionBlockSemantics
            DECODE_BLOCK = new RouteProjection.AdmissionBlockSemantics(
                    "DELIVERY_CAPACITY_DECODE_ENGINE",
                    RouteProjection.AfterProbeAdmission.UNAVAILABLE,
                    "DECODE_CAPACITY_SCOPE_UNKNOWN",
                    RoleType.DECODE);

    /** Immutable association; the exact Decode permit owns its resolution state. */
    record Member(RequestRoute route, DecodeEndpoint.EngineDispatchPermit decode) implements AutoCloseable {
        Member {
            Objects.requireNonNull(route, "route");
        }

        @Override
        public void close() {
            if (decode != null) { decode.release(); }
        }
    }

    static CapacityBoundary.Attempt<Member> prepareMember(
            RequestRoute item) {
        RequestRequirements binding = item.requirements();
        DecodeEndpoint decode = item.decodeEp();
        if (decode == null && item.decodeReservation() == null) {
            // The committed topology has no independent Decode resource.
            // Its Prefill/PDFUSION reservation still follows the same handoff.
            return captureMember(item, null);
        }
        if (decode == null || item.decodeReservation() == null) {
            return failed(new IllegalStateException("Decode reservation is unavailable: request_id=" + item.requestId()));
        }
        DecodeEndpoint.EngineDispatchPermitAcquisition acquisition;
        try {
            acquisition = decode.acquireDispatchPermit(item.decodeReservation(), binding.capacity());
        } catch (RuntimeException | Error failure) {
            return failed(failure);
        }
        return switch (acquisition.status()) {
            case ACQUIRED, ALREADY_ACCEPTED -> captureMember(item, acquisition.permit());
            case CAPACITY_FULL -> rejected(CapacityBoundary.unavailable(
                    new DecodeAvailability(item), DECODE_BLOCK));
            case NOT_OWNED, NOT_QUEUED -> rejected(
                    CapacityBoundary.OWNERSHIP_LOST);
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
            RequestRoute item,
            DecodeEndpoint.EngineDispatchPermit permit) {
        try {
            return accepted(new Member(item, permit));
        } catch (Throwable captureFailure) {
            return failed(Failures.run(captureFailure, permit == null ? null : permit::release));
        }
    }

    private static CapacityBoundary rejectedPrefill(
            RequestRoute item,
            PrefillState.CapacityStatus status,
            CapacityBoundary capacityFull) {
        return switch (status) {
            case CAPACITY_FULL -> capacityFull;
            case REQUEST_NOT_ACTIVE -> CapacityBoundary.OWNERSHIP_LOST;
            case ENDPOINT_RETIRED -> CapacityBoundary.failed(retired("Prefill", item));
            case REQUEST_ALREADY_RESERVED, BATCH_ID_ALREADY_RESERVED ->
                    CapacityBoundary.failed(new IllegalStateException(
                            "Prefill admission owns another reservation: "
                                    + "request_id=" + item.requestId()
                                    + " status=" + status));
            case ACQUIRED -> throw new IllegalArgumentException(
                    "ACQUIRED must carry a reservation");
        };
    }

    private static IllegalStateException retired(
            String role,
            RequestRoute item) {
        return new IllegalStateException(
                role + " endpoint generation retired: request_id="
                        + item.requestId());
    }

    private static <T> CapacityBoundary.Attempt<T> failed(Throwable cause) {
        return rejected(CapacityBoundary.failed(cause));
    }

    /** Close only locally retained permits; committed capacity remains endpoint-owned. */
    static void closeCommitted(List<Member> members, PrefillState.CommittedHandoff handoff) {
        try {
            for (int index = 0; index < members.size(); index++) {
                Throwable failure = Failures.close(members.get(index));
                if (failure != null) {
                    Logger.error("Committed admission cleanup isolated", failure);
                }
            }
        } finally {
            if (handoff != null) {
                Throwable failure = Failures.close(handoff);
                if (failure != null) {
                    Logger.error("Committed Prefill handoff cleanup isolated", failure);
                }
            }
        }
    }

    /** Decode is the exact event source for its request-scoped permit. */
    private record DecodeAvailability(RequestRoute item) implements CapacityBoundary.Availability {
        @Override
        public boolean isAvailable() {
            return item.decodeEp().shouldRetryDispatch(item.requestId(), item.requirements().capacity());
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
