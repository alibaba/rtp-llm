package org.flexlb.balance.eviction;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.preemption.CancelTarget;

import java.util.concurrent.CompletableFuture;

/**
 * Abstraction over the engine-side priority-preemption Cancel RPC.
 *
 * <p>Contract highlights:
 * <ul>
 *   <li>the cancel is sent to the victim's original Prefill endpoint, which
 *       owns the P/D connection and propagates cancellation downstream;</li>
 *   <li>{@code ACCEPTED} only proves that Prefill installed the cancel intent;
 *       resource settlement requires typed WorkerStatus {@code CANCELED};</li>
 *   <li>{@code request_id} identifies the victim request.</li>
 * </ul>
 */
public interface EngineCancelChannel {

    /** Whether accepted eviction is enabled for victims held by this Decode endpoint. */
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
        TOMBSTONED,
        /** Endpoint has no cancel path at all — planning-gate violation. */
        UNSUPPORTED,
        /** Transport-layer failure (RPC error/timeout, or unroutable cancel). */
        FAILED
    }

    /**
     * Engine response to a cancel intent. No resource-release fact is carried.
     */
    record CancelOutcome(CancelAck ack) {

        public static CancelOutcome accepted() {
            return new CancelOutcome(CancelAck.ACCEPTED);
        }

        public static CancelOutcome notFound() {
            return new CancelOutcome(CancelAck.NOT_FOUND);
        }

        public static CancelOutcome tombstoned() {
            return new CancelOutcome(CancelAck.TOMBSTONED);
        }

        public static CancelOutcome unsupported() {
            return new CancelOutcome(CancelAck.UNSUPPORTED);
        }

        public static CancelOutcome failed() {
            return new CancelOutcome(CancelAck.FAILED);
        }
    }
}
