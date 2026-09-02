package org.flexlb.balance.scheduler;

import org.flexlb.balance.eviction.EngineCancelChannel;
import org.flexlb.balance.preemption.CancelTarget;
import org.flexlb.util.Logger;

import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.function.BiFunction;

/**
 * One-shot coordinator for an exact request slot and Engine fence.
 *
 * <p>The slot linearizes ownership before the Cancel invocation. Every local
 * or transport failure then leaves the exact fence waiting for authoritative
 * WorkerStatus or generation-retirement evidence; this coordinator never
 * retries and never owns a timer or a request registry.
 */
final class EngineFenceCoordinator {

    private final EngineCancelChannel cancelChannel;
    private final BiFunction<RequestSlot, RequestSlot.FenceTerminalProof,
            TerminalDisposition> terminalSink;

    EngineFenceCoordinator(
            EngineCancelChannel cancelChannel,
            BiFunction<RequestSlot, RequestSlot.FenceTerminalProof,
                    TerminalDisposition> terminalSink) {
        this.cancelChannel = Objects.requireNonNull(
                cancelChannel, "cancelChannel");
        this.terminalSink = Objects.requireNonNull(
                terminalSink, "terminalSink");
    }

    /**
     * Send exactly one Cancel after the exact fence crosses its invocation
     * boundary.
     *
     * @return true when this invocation acquired that boundary; false when
     *         the slot or fence no longer owned it
     */
    boolean start(
            RequestSlot slot,
            RequestSlot.FenceHandle exactFence,
            CancelTarget target,
            long timeoutMs) {
        synchronized (slot) {
            RequestSlot.FenceReduction reduction = slot.applyFenceUpdate(
                    exactFence, RequestSlot.FenceUpdate.CANCEL_STARTED);
            if (reduction.status()
                    == RequestSlot.FenceReduction.Status.STALE) {
                return false;
            }
            requireNoFenceEffect(reduction, "Cancel start");
        }

        CompletableFuture<EngineCancelChannel.CancelOutcome> outcome;
        try {
            outcome = cancelChannel.cancel(
                    target, slot.requestId(), timeoutMs);
            if (outcome == null) {
                awaitAuthoritativeTerminal(slot, exactFence);
                return true;
            }
            outcome.whenComplete((result, failure) -> {
                try {
                    complete(slot, exactFence, result, failure);
                } catch (Throwable invariantFailure) {
                    // The callback stage is intentionally not retained. Make
                    // a broken total-reducer contract operationally visible;
                    // never retry or reinterpret the authoritative outcome.
                    Logger.error(
                            "Engine fence completion invariant failed: request_id={}",
                            slot.requestId(), invariantFailure);
                }
            });
        } catch (RuntimeException | Error invocationFailure) {
            // The one-shot boundary was already crossed. Conservatively keep
            // exact ownership; a synchronous failure does not prove that the
            // cancel intent failed to reach the engine.
            awaitAuthoritativeTerminal(slot, exactFence);
        }
        return true;
    }

    private void complete(
            RequestSlot slot,
            RequestSlot.FenceHandle exactFence,
            EngineCancelChannel.CancelOutcome outcome,
            Throwable failure) {
        if (failure == null
                && outcome != null
                && outcome.ack()
                    == EngineCancelChannel.CancelAck.TOMBSTONED) {
            resumeTombstoned(slot, exactFence);
            return;
        }
        awaitAuthoritativeTerminal(slot, exactFence);
    }

    /**
     * Consume an exact TOMBSTONED proof, including one which arrived while an
     * admission mutation owned the slot. This is the sole sink boundary for
     * both the transport callback and admission-completion replay.
     */
    TerminalDisposition resumeTombstoned(
            RequestSlot slot,
            RequestSlot.FenceHandle exactFence) {
        RequestSlot.FenceReduction reduction;
        synchronized (slot) {
            reduction = slot.applyFenceUpdate(
                    exactFence, RequestSlot.FenceUpdate.TOMBSTONED);
        }
        if (reduction.status()
                == RequestSlot.FenceReduction.Status.DEFERRED) {
            return TerminalDisposition.DEFERRED;
        }
        if (reduction.status()
                == RequestSlot.FenceReduction.Status.STALE) {
            return TerminalDisposition.STALE;
        }
        if (reduction.status()
                != RequestSlot.FenceReduction.Status.TERMINAL_PROOF) {
            throw new IllegalStateException(
                    "Engine fence TOMBSTONED produced an invalid effect: "
                            + reduction.getClass().getSimpleName());
        }

        // The authoritative proof is already persisted in the exact slot.
        // The sink owns a total reduction: leaf cleanup failures are isolated
        // there and cannot veto the RequestSlot terminal edge.
        TerminalDisposition disposition =
                terminalSink.apply(slot, reduction.proof());
        if (disposition == null) {
            throw new IllegalStateException("missing Engine fence disposition");
        }
        validateDisposition(slot, exactFence, disposition);
        return disposition;
    }

    private static void validateDisposition(
            RequestSlot slot,
            RequestSlot.FenceHandle exactFence,
            TerminalDisposition disposition) {
        synchronized (slot) {
            RequestSlot.FenceReduction actual = slot.applyFenceUpdate(
                    exactFence, RequestSlot.FenceUpdate.TOMBSTONED);
            boolean valid = switch (disposition) {
                case TERMINALIZED -> slot.isTombstone();
                case DEFERRED -> actual.status()
                        == RequestSlot.FenceReduction.Status.DEFERRED;
                case STALE -> actual.status()
                        == RequestSlot.FenceReduction.Status.STALE;
            };
            if (!valid) {
                throw new IllegalStateException(
                        "Engine fence TOMBSTONED proof was not consumed: "
                                + "reported=" + disposition
                                + ", actual="
                                + actual.getClass().getSimpleName()
                                + ", request_id=" + slot.requestId());
            }
        }
    }

    private static void awaitAuthoritativeTerminal(
            RequestSlot slot,
            RequestSlot.FenceHandle exactFence) {
        synchronized (slot) {
            RequestSlot.FenceReduction reduction = slot.applyFenceUpdate(
                    exactFence, RequestSlot.FenceUpdate.AWAIT_TERMINAL);
            if (reduction.status()
                    != RequestSlot.FenceReduction.Status.STALE) {
                requireNoFenceEffect(reduction, "await terminal");
            }
        }
    }

    private static void requireNoFenceEffect(
            RequestSlot.FenceReduction reduction,
            String operation) {
        if (reduction.status()
                != RequestSlot.FenceReduction.Status.NONE) {
            throw new IllegalStateException(
                    operation + " produced an invalid Engine fence effect: "
                            + reduction.getClass().getSimpleName());
        }
    }

    enum TerminalDisposition {
        TERMINALIZED,
        DEFERRED,
        STALE
    }
}
