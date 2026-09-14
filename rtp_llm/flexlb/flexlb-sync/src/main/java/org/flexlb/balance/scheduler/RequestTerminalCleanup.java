package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.preemption.VictimTerminal;
import org.flexlb.util.Logger;

/** Executes an already-owned terminal action, then commits its tombstone and publishes. */
final class RequestTerminalCleanup {
    private final ExpirationTimer expirationTimer;

    RequestTerminalCleanup(ExpirationTimer timer) {
        this.expirationTimer = timer;
    }
    RequestSlot.PublicationPermit finishTerminal(
            TerminalAction action) {
        RequestSlot entry = action.slot();
        ScheduledRequest item = action.item();
        Throwable cleanupFailure = null;
        cleanupFailure = runTerminalLeaf(
                cleanupFailure,
                action.terminalResources() == null
                        ? null : () -> expirationTimer.release(
                                action.terminalResources()));
        cleanupFailure = runTerminalLeaf(
                cleanupFailure,
                !action.removePrefillQueue()
                                || item == null || item.prefillEp() == null
                        ? null : () -> item.prefillEp().removeQueued(
                                item, action.queueReason()));
        cleanupFailure = runTerminalLeaf(
                cleanupFailure,
                action.releaseDecode() && item != null
                        ? () -> rollback(item) : null);
        cleanupFailure = runTerminalLeaf(
                cleanupFailure,
                action.releasePrefill() && item != null
                        ? () -> releasePrefillAccounting(item) : null);
        cleanupFailure = runTerminalLeaf(cleanupFailure, action.counterpartCleanup());
        cleanupFailure = runTerminalLeaf(
                cleanupFailure,
                action.preemption() == null
                        ? null : () -> action.preemption().signalTerminal(new VictimTerminal(entry.requestId())));

        TombstoneResult tombstone;
        synchronized (entry) {
            tombstone = entry.finishTombstone(action);
        }
        Throwable terminalFailure = tombstone.transitionFailure() == null
                ? cleanupFailure
                : appendFailure(cleanupFailure, tombstone.transitionFailure());
        if (terminalFailure != null) {
            Logger.error("Terminal cleanup isolated after canonical claim: request_id={}",
                    entry.requestId(), terminalFailure);
        }
        return tombstone.terminal() == null
                ? null : tombstone.publication();
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

    private void rollback(ScheduledRequest item) {
        DecodeEndpoint decodeEp = item.decodeEp();
        DecodeEndpoint.ReservationHandle reservation =
                item.decodeReservation();
        if (decodeEp != null && reservation != null) {
            decodeEp.releaseReservationExact(reservation);
        }
    }

    private void releasePrefillAccounting(ScheduledRequest item) {
        PrefillEndpoint prefillEp = item.prefillEp();
        if (prefillEp == null) {
            return;
        }
        if (prefillEp.releaseCommittedItem(item)) {
            Logger.debug("FlexLB release canonical Prefill accounting: request_id={} engine={}",
                    item.requestId(), prefillEp.getIp());
        }
    }

    static void expireEndpointAccounting(ScheduledRequest item) {
        Throwable failure = null;
        if (item.decodeEp() != null && item.decodeReservation() != null) {
            failure = runTerminalLeaf(failure,
                    () -> item.decodeEp().expireReservationExact(item.decodeReservation()));
        }
        if (item.prefillEp() != null) {
            failure = runTerminalLeaf(failure, () -> item.prefillEp().expireCommittedItem(item));
        }
        if (failure != null) {
            throw new IllegalStateException("request expiration cleanup failed: " + item.requestId(), failure);
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
