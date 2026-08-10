package org.flexlb.balance.scheduler.priority;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;

import java.util.concurrent.CompletableFuture;

/**
 * Abstraction over the engine-side cancel RPC. A cancel is
 * an <b>intent injection</b> only: even {@code ACCEPTED} does not mean the
 * resources are released — the sole confirmation source remains the periodic
 * WorkerStatus report ({@code resource_released=true}), consumed through
 * {@link ReleaseTracker}.
 *
 * <p>Contract highlights:
 * <ul>
 *   <li>the real engine cancel ALWAYS targets the victim's original
 *       <b>Prefill lifecycle owner</b> (looked up from the request's
 *       inflight entry via {@link InflightRegistrar#getDispatchTarget}),
 *       never the current Decode
 *       endpoint;</li>
 *   <li>the engine Cancel RPC always answers {@code ACCEPTED} (intent
 *       registration semantics) — no protocol branch carries release or
 *       terminal-state information, so the ack is diagnostics only;</li>
 *   <li>{@code request_id} is the idempotency key; a
 *       transport retry returns the same ACCEPTED.</li>
 * </ul>
 */
public interface EngineCancelChannel {

    /**
     * Whether victims on this endpoint may be planned for accepted-eviction.
     * Planning only considers accepted-layer victims on supported endpoints.
     */
    boolean isSupported(DecodeEndpoint endpoint);

    /**
     * Asynchronously ask the engine to cancel one request. Never throws
     * synchronously; transport-level failures surface either as a completed
     * {@code FAILED} outcome or as a failed future — callers treat both
     * identically (the intent may still have landed; release stays gated on
     * the WorkerStatus report).
     */
    CompletableFuture<CancelOutcome> cancel(CancelTarget target,
                                            long requestId,
                                            CancelReason reason);

    /** Why the cancel was issued — mirrors EngineCancelReasonPB. */
    enum CancelReason {
        USER_CANCELLED,
        PRIORITY_PREEMPTED,
        DEADLINE_EXCEEDED,
        ADMIN
    }

    /**
     * Cancel routing information.
     *
     * @param lifecycleOwner     original Prefill owner — the REAL engine cancel
     *                           destination; may be null when the owner
     *                           resolver has no record (cancel then cannot be
     *                           routed and resolves to {@code FAILED}, no
     *                           release assumed)
     * @param decodeEndpoint     current Decode endpoint — used only by the
     *                           TEST-ONLY mock control plane routing
     * @param batchId            diagnostics only, never used for fencing
     */
    record CancelTarget(PrefillEndpoint lifecycleOwner,
                        DecodeEndpoint decodeEndpoint,
                        long batchId) {

        public static CancelTarget of(PrefillEndpoint owner,
                                      DecodeEndpoint decodeEndpoint,
                                      long batchId) {
            return new CancelTarget(owner, decodeEndpoint, batchId);
        }
    }

    /**
     * Structured ack. The engine RPC only ever answers ACCEPTED (intent
     * registration); the two other values are local branches.
     */
    enum CancelAck {
        /** Intent registered engine-side (the only engine RPC answer). */
        ACCEPTED,
        /** Endpoint has no cancel path at all — planning-gate violation. */
        UNSUPPORTED,
        /** Transport-layer failure (RPC error/timeout, or unroutable cancel). */
        FAILED
    }

    /**
     * Engine response to a cancel intent. No release
     * future is provided by design — pair with {@link ReleaseTracker}.
     */
    record CancelOutcome(CancelAck ack) {

        public static CancelOutcome accepted() {
            return new CancelOutcome(CancelAck.ACCEPTED);
        }

        public static CancelOutcome unsupported() {
            return new CancelOutcome(CancelAck.UNSUPPORTED);
        }

        public static CancelOutcome failed() {
            return new CancelOutcome(CancelAck.FAILED);
        }
    }
}
