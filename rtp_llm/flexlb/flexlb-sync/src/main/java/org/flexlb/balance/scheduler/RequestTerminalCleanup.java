package org.flexlb.balance.scheduler;

import org.flexlb.balance.preemption.VictimTerminal;
import org.flexlb.util.Logger;

/** Executes an already-owned terminal action, then commits its terminal record and publishes. */
final class RequestTerminalCleanup {
    private final ExpirationTimer expirationTimer;

    RequestTerminalCleanup(ExpirationTimer timer) {
        this.expirationTimer = timer;
    }
    RequestSlot.PublicationPermit finishTerminal(
            TerminalAction action) {
        RequestSlot entry = action.slot();
        Throwable cleanupFailure = null;
        cleanupFailure = runTerminalLeaf(
                cleanupFailure,
                () -> action.terminalResources().release(expirationTimer));
        cleanupFailure = runTerminalLeaf(cleanupFailure, () -> entry.releaseTerminalEndpoints(action));
        cleanupFailure = runTerminalLeaf(
                cleanupFailure,
                action.preemption() == null
                        ? null : () -> action.preemption().signalTerminal(new VictimTerminal(entry.requestId())));

        TerminationResult result;
        synchronized (entry) {
            result = entry.finishTermination(action);
        }
        Throwable terminalFailure = result.transitionFailure() == null
                ? cleanupFailure
                : appendFailure(cleanupFailure, result.transitionFailure());
        if (terminalFailure != null) {
            Logger.error("Terminal cleanup isolated after canonical claim: request_id={}",
                    entry.requestId(), terminalFailure);
        }
        return result.terminal() == null
                ? null : result.publication();
    }

    void submitTerminal(TerminalAction action) {
        if (action == null) {
            return;
        }
        RequestSlot.PublicationPermit permit = finishTerminal(action);
        if (permit != null && action.response() != null) {
            action.slot().submitTerminalResponse(permit, action.response());
        }
    }

    static Throwable runTerminalLeaf(Throwable first, Runnable leaf) {
        if (leaf == null) {
            return first;
        }
        try {
            leaf.run();
            return first;
        } catch (Throwable failure) {
            return appendFailure(first, failure);
        }
    }

    static Throwable appendFailure(Throwable first, Throwable next) {
        if (first == null) {
            return next;
        }
        if (first != next) {
            first.addSuppressed(next);
        }
        return first;
    }
}
