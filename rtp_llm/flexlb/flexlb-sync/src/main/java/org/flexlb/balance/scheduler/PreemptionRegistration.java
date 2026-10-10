package org.flexlb.balance.scheduler;

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
public final class PreemptionRegistration {
    private final RequestSlot owner;
    private final long requestId;
    private final long attemptToken;
    private final String detail;
    private final CompletableFuture<VictimTerminal> terminal =
            new CompletableFuture<>();

    private PreemptionCancelPhase phase = PreemptionCancelPhase.CLAIMED;
    private boolean finished;
    private DeferredTerminal pendingTerminal;
    private boolean pendingDeliveryConfirmation;
    private long pendingConfirmationBatchId;

    PreemptionRegistration(
            RequestSlot owner, long requestId,
            long attemptToken,
            String detail) {
        this.owner = owner;
        this.requestId = requestId;
        this.attemptToken = attemptToken;
        this.detail = detail == null ? "priority preemption" : detail;
    }

    public boolean applyPhase(PreemptionCancelPhase phase) { return owner.updatePreemption(this, phase); }

    public boolean release() { return owner.releasePreemption(this); }

    public boolean completePreemption(String detail) { return owner.completePreemption(this, detail); }

    public long requestId() {
        return requestId;
    }

    public long attemptToken() {
        return attemptToken;
    }

    public CompletionStage<VictimTerminal> terminalObservation() {
        return terminal;
    }

    boolean signalTerminal(VictimTerminal exactTerminal) {
        return terminal.complete(exactTerminal);
    }

    String detail() {
        return detail;
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

    boolean advanceTo(PreemptionCancelPhase next) {
        if (finished || !phase.canTransitionTo(next)) {
            return false;
        }
        phase = next;
        return true;
    }

    /** Record protocol completion once; resource cleanup and terminal notification still belong to the slot. */
    boolean tryFinish() {
        if (finished) {
            return false;
        }
        finished = true;
        return true;
    }

    boolean isReleasable() {
        return !finished && phase.isLocallyReleasable();
    }

    boolean isNotFound() {
        return !finished && phase == PreemptionCancelPhase.NOT_FOUND_STALE;
    }

    boolean isUnknown() {
        return !finished && phase == PreemptionCancelPhase.CANCEL_UNKNOWN;
    }

    boolean isFinished() {
        return finished;
    }

    boolean canCompletePreemption() {
        return !finished && phase.acceptsRequestFenced();
    }

    void storeTerminal(DeferredTerminal selected) {
        pendingTerminal = selected;
    }

    void recordDeliveryConfirmation(long batchId) {
        if (!pendingDeliveryConfirmation) {
            pendingDeliveryConfirmation = true;
            pendingConfirmationBatchId = batchId;
        }
    }
}
