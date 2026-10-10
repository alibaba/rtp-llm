package org.flexlb.balance.eviction;

import org.flexlb.balance.preemption.CancelTarget;
import org.flexlb.balance.scheduler.CancelReason;

import java.util.concurrent.CompletableFuture;

/**
 * Transport for ordinary cleanup and priority-preemption Cancel RPCs.
 *
 * <p>Contract highlights:
 * <ul>
 *   <li>the cancel is sent to the request's original Prefill endpoint, which
 *       owns the P/D connection and propagates cancellation downstream;</li>
 *   <li>{@code ACCEPTED} only proves that Prefill installed the cancel intent;
 *       resource settlement requires separate, verified cleanup evidence;</li>
 *   <li>{@code request_id} identifies the request and cannot be reused remotely.</li>
 * </ul>
 */
public interface EngineCancelChannel {

    /**
     * Asynchronously ask the engine to cancel one request. Never throws
     * synchronously; transport-level failures surface either as a completed
     * {@code FAILED} outcome or as a failed future — callers treat both
     * identically (the intent may still have landed; release stays gated on
     * the WorkerStatus report).
     */
    CompletableFuture<CancelAck> cancel(CancelTarget target,
                                        long requestId,
                                        CancelReason reason,
                                        long timeoutMs);

    /**
     * Local delivery outcome. ACCEPTED and NOT_FOUND come from the engine
     * response; UNSUPPORTED and FAILED are local transport/capability branches.
     */
    enum CancelAck {
        /** The addressed Prefill accepted the cancel intent. */
        ACCEPTED,
        /** The addressed Prefill does not own or know the request. */
        NOT_FOUND,
        /**
         * Prefill atomically fenced this request id while it was absent. Any
         * racing later Enqueue is rejected before reaching the scheduler.
         */
        REQUEST_FENCED,
        /** Prefill fence plus explicit downstream cleanup proof; sender exit is still required. */
        REQUEST_CLEANED,
        /** Endpoint does not support the requested cancellation semantics. */
        UNSUPPORTED,
        /** Transport-layer failure (RPC error/timeout, or unroutable cancel). */
        FAILED
    }

}
