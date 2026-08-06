package org.flexlb.balance.scheduler.priority;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.enums.TaskPhase;

import java.util.concurrent.CompletableFuture;

/**
 * Abstraction over the engine-side cancel RPC (Phase 5, see
 * {@code docs/auto_tpm/engine_cancel_rpc_design.md}). A cancel is an
 * <b>intent injection</b> only: even a successful outcome does not mean the
 * resources are released — the sole confirmation source remains the next
 * WorkerStatus report (iron rule 4). Callers must therefore pair
 * {@link #cancel} with a bounded release-confirmation wait.
 *
 * <p>The real gRPC-backed implementation lands together with the engine-side
 * Cancel RPC; until then the Spring context wires
 * {@link UnsupportedEngineCancelChannel}, which keeps every endpoint
 * unsupported so accepted-eviction planning never activates.
 */
public interface EngineCancelChannel {

    /**
     * Whether the engine behind this endpoint supports the Cancel RPC.
     * Planning only considers accepted-layer victims on supported endpoints —
     * an unsupported endpoint must never receive a cancel intent.
     */
    boolean isSupported(DecodeEndpoint endpoint);

    /**
     * Asynchronously ask the engine to cancel one request. Never throws
     * synchronously; transport-level failures surface as a failed future,
     * protocol-level branches (not found / already finished / unsupported)
     * surface as a completed {@link CancelOutcome}.
     */
    CompletableFuture<CancelOutcome> cancel(DecodeEndpoint endpoint,
                                            long requestId,
                                            CancelReason reason);

    /** Why the cancel was issued — mirrors the design doc's CancelReasonPB. */
    enum CancelReason {
        USER_CANCELLED,
        PRIORITY_PREEMPTED,
        ADMIN
    }

    /**
     * Engine response to a cancel intent (three-branch contract):
     * <ul>
     *   <li>{@code found=false} — the engine does not know the request; the
     *       caller may treat its resources as already released,</li>
     *   <li>{@code alreadyFinished=true} — terminal before the cancel landed;
     *       resources already released,</li>
     *   <li>{@code found=true} — cancel intent accepted at {@code phase};
     *       release must still be confirmed via WorkerStatus.</li>
     * </ul>
     * {@code unsupported=true} means the endpoint has no Cancel RPC at all —
     * a planning-gate violation the scheduler must fail the plan on.
     */
    record CancelOutcome(boolean found,
                         TaskPhase phase,
                         boolean alreadyFinished,
                         boolean unsupported) {

        /** Intent accepted while the request was live at {@code phase}. */
        public static CancelOutcome accepted(TaskPhase phase) {
            return new CancelOutcome(true, phase, false, false);
        }

        /** The engine does not know the request. */
        public static CancelOutcome notFound() {
            return new CancelOutcome(false, null, false, false);
        }

        /** The request reached a terminal state before the cancel landed. */
        public static CancelOutcome finishedBeforeCancel() {
            return new CancelOutcome(true, null, true, false);
        }

        /** The endpoint has no Cancel RPC. */
        public static CancelOutcome unsupportedEndpoint() {
            return new CancelOutcome(false, null, false, true);
        }
    }
}
