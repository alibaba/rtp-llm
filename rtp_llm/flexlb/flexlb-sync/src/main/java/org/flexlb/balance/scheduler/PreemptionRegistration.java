package org.flexlb.balance.scheduler;

import org.flexlb.balance.preemption.PreemptionClaim;
import org.flexlb.balance.preemption.PreemptionCancelPhase;
import org.flexlb.balance.preemption.VictimTerminal;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionStage;

/**
 * Exact ownership token for one priority-preemption attempt.
 *
 * <p>This class owns only the attempt-local protocol. RequestSlot remains the
 * aggregate root and decides when a transition is legal for the request as a
 * whole. Keeping this small state machine separate makes that boundary
 * explicit and prevents transport bookkeeping from obscuring request
 * lifecycle decisions.</p>
 */
final class PreemptionRegistration implements PreemptionClaim {
    private final long requestId;
    private final long attemptToken;
    private final String detail;
    private final CompletableFuture<VictimTerminal> terminal =
            new CompletableFuture<>();

    private PreemptionCancelPhase phase = PreemptionCancelPhase.CLAIMED;
    private boolean settled;
    private DeferredTerminal pendingTerminal;
    private boolean pendingDeliveryConfirmation;
    private long pendingConfirmationBatchId;
    private String postDeliveryFenceDetail;

    PreemptionRegistration(
            long requestId,
            long attemptToken,
            String detail) {
        this.requestId = requestId;
        this.attemptToken = attemptToken;
        this.detail = detail == null ? "priority preemption" : detail;
    }

    @Override
    public long requestId() {
        return requestId;
    }

    @Override
    public long attemptToken() {
        return attemptToken;
    }

    @Override
    public CompletionStage<VictimTerminal> terminalObservation() {
        return terminal;
    }

    boolean signalTerminal(VictimTerminal exactTerminal) {
        return terminal.complete(exactTerminal);
    }

    String detail() {
        return detail;
    }

    String postDeliveryFenceDetail() {
        return postDeliveryFenceDetail;
    }

    void requirePostDeliveryFence(String fenceDetail) {
        postDeliveryFenceDetail = fenceDetail;
    }

    DeferredTerminal pendingTerminal() {
        return pendingTerminal;
    }

    boolean hasPendingDeliveryConfirmation() {
        return pendingDeliveryConfirmation;
    }

    long pendingConfirmationBatchId() {
        return pendingConfirmationBatchId;
    }

    boolean beginCancel() {
        if (settled || !phase.canStartCancel()) {
            return false;
        }
        phase = PreemptionCancelPhase.CANCEL_IN_FLIGHT;
        return true;
    }

    boolean acceptCancel() {
        if (settled || !phase.canAcceptCancel()) {
            return false;
        }
        phase = PreemptionCancelPhase.CANCEL_REQUESTED;
        return true;
    }

    boolean markNotFound() {
        if (settled || !phase.canRecordNotFound()) {
            return false;
        }
        phase = PreemptionCancelPhase.NOT_FOUND_STALE;
        return true;
    }

    boolean markUnknown() {
        if (settled || !phase.canRecordUnknown()) {
            return false;
        }
        phase = PreemptionCancelPhase.CANCEL_UNKNOWN;
        return true;
    }

    boolean settle() {
        if (settled) {
            return false;
        }
        settled = true;
        return true;
    }

    boolean isReleasable() {
        return !settled && phase.isLocallyReleasable();
    }

    boolean isFenceTransferable() {
        return !settled && phase.isFenceTransferable();
    }

    boolean isNotFound() {
        return !settled && phase == PreemptionCancelPhase.NOT_FOUND_STALE;
    }

    boolean isUnknown() {
        return !settled && phase == PreemptionCancelPhase.CANCEL_UNKNOWN;
    }

    boolean isSettled() {
        return settled;
    }

    boolean canSettleTombstone() {
        return !settled && phase.acceptsTombstone();
    }

    void retainTerminal(DeferredTerminal candidate) {
        if (pendingTerminal == null
                || (!pendingTerminal.authoritativeWorker()
                    && candidate.authoritativeWorker())) {
            pendingTerminal = candidate;
        }
    }

    void recordDeliveryConfirmation(long batchId) {
        if (!pendingDeliveryConfirmation) {
            pendingDeliveryConfirmation = true;
            pendingConfirmationBatchId = batchId;
        }
    }
}
