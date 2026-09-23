package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.preemption.VictimTerminal;
import org.flexlb.util.Logger;

/** Executes an already-owned terminal action, then commits its terminal record and publishes. */
final class RequestTerminalCleanup {
    private final ExpirationTimer expirationTimer;

    RequestTerminalCleanup(ExpirationTimer timer) {
        this.expirationTimer = timer;
    }

    // ── 共同收尾：清理、提交记录、发布结果 ──

    void submitTerminal(TerminalAction action) {
        if (action == null) {
            return;
        }
        RequestCompletionPublisher.PublicationPermit permit = finishTerminal(action);
        if (permit != null && action.response() != null) {
            action.slot().submitTerminalResponse(permit, action.response());
        }
    }

    RequestCompletionPublisher.PublicationPermit finishTerminal(
            TerminalAction action) {
        RequestSlot entry = action.slot();
        entry.requireCleanupOwner(action);
        Throwable cleanupFailure = null;
        cleanupFailure = runStep(
                cleanupFailure,
                () -> action.terminalResources().release(expirationTimer));
        cleanupFailure = runStep(cleanupFailure, () -> releaseEndpoints(action));
        cleanupFailure = runStep(
                cleanupFailure,
                action.preemption() == null
                        ? null : () -> action.preemption().signalTerminal(new VictimTerminal(entry.requestId())));

        TerminationResult result = entry.commitTerminalRecord(action);
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

    private void releaseEndpoints(TerminalAction action) {
        if (action.endpointsSettled()) { return; }
        DeliveryClaimKind delivery = action.deliveryKind();
        ScheduledRequest exact = action.item();
        if (exact == null) { return; }
        DeferredTerminal event = action.event();
        Throwable failure = null;
        if (delivery == DeliveryClaimKind.NONE && exact.prefillEp() != null) {
            failure = runStep(failure,
                    () -> exact.prefillEp().removeQueued(exact, "TERMINAL_RELEASE"));
        }
        if (event != null && event.kind() == DeferredTerminal.Kind.INACTIVITY_EXPIRED) {
            // Expiry ends exact local tracking even when delivery is uncertain or Engine-owned.
            failure = runStep(failure,
                    exact.decodeEp() == null ? null
                            : () -> exact.decodeEp().release(exact.decodeReservation(), DecodeEndpoint.ReleaseReason.EXPIRED));
            failure = runStep(failure,
                    exact.prefillEp() == null ? null : () -> exact.prefillEp().expireCommittedItem(exact));
        } else {
            // A projection may lag endpoint ownership. Never roll back an Engine/protocol owner.
            failure = runStep(failure,
                    exact.decodeEp() == null ? null
                            : () -> exact.decodeEp().release(
                                    exact.decodeReservation(),
                                    DecodeEndpoint.ReleaseReason.COUNTERPART_FINISHED));
            boolean releasePrefill = event != null && switch (event.kind()) {
                case DECODE_GENERATION_RETIRED, PRIORITY -> true;
                case WORKER -> event.workerSource() != WorkerTerminalSource.PREFILL_ENDPOINT
                        && delivery != DeliveryClaimKind.BATCH_ENQUEUE;
                default -> delivery != DeliveryClaimKind.BATCH_ENQUEUE;
            };
            if (event == null) { releasePrefill = delivery != DeliveryClaimKind.BATCH_ENQUEUE; }
            if (releasePrefill && exact.prefillEp() != null) {
                failure = runStep(failure,
                        () -> exact.prefillEp().releaseCommittedItem(exact));
            }
        }
        rethrowCleanup(failure);
    }

    // ── 派发失败：分别结算 Prefill 与 Decode ──

    void settlePrefill(ScheduledRequest exact, boolean expired) {
        if (exact.prefillEp() == null) { return; }
        if (expired) {
            exact.prefillEp().removeQueued(exact, "REQUEST_INACTIVE");
            exact.prefillEp().expireCommittedItem(exact);
        } else {
            exact.prefillEp().settleFailedRequest(exact);
        }
    }

    boolean settleDecode(ScheduledRequest exact, DeliveryResult.Status source, boolean expired) {
        if (exact.decodeEp() == null || exact.decodeReservation() == null) { return true; }
        if (expired) {
            exact.decodeEp().release(exact.decodeReservation(), DecodeEndpoint.ReleaseReason.EXPIRED);
            return true;
        }
        return exact.decodeEp().settleFailedRequest(exact.decodeReservation(), source);
    }

    // ── 独立执行与异常汇总 ──

    static Throwable runStep(Throwable first, Runnable leaf) {
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

    static void rethrowCleanup(Throwable failure) {
        if (failure instanceof RuntimeException runtime) {
            throw runtime;
        }
        if (failure instanceof Error error) {
            throw error;
        }
        if (failure != null) {
            throw new IllegalStateException(
                    "request slot cleanup failed", failure);
        }
    }

}
