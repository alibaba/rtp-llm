package org.flexlb.balance.scheduler.priority;

import org.flexlb.balance.scheduler.BatchItem;

import java.util.concurrent.CompletableFuture;

/**
 * Registers Auto-TPM admitted requests into the priority scheduler's inflight
 * tracking so that dispatch, completion, TTL cleanup and rollback treat them
 * exactly like legacy-path requests.
 *
 * <p>Implemented by {@code PriorityScheduler}; expressed as an interface to
 * avoid a circular bean dependency with {@code PriorityAdmissionScheduler}.
 */
public interface InflightRegistrar {

    /**
     * Register the item as inflight.
     *
     * @return false when the request id is already inflight or terminal
     *         (duplicate) — the item was NOT registered
     */
    boolean registerInflight(BatchItem item);

    /** Whether the exact public future has crossed the inflight handoff. */
    default boolean isInflightGeneration(
            String requestId, CompletableFuture<?> future) {
        return false;
    }

    /**
     * Request-scoped admission gate used by asynchronous cancel coordination.
     * Implementations may close it before the public future is published.
     */
    default boolean isAdmissionOpen(String requestId, CompletableFuture<?> future) {
        return !future.isDone();
    }

    /**
     * Claim an asynchronous admission mutation before it can affect another
     * request (for example, an Engine-side eviction). While claimed, external
     * cancellation is accepted as pending rather than falsely terminal.
     */
    default boolean claimAdmissionMutation(
            String requestId, CompletableFuture<?> future) {
        return isAdmissionOpen(requestId, future);
    }

    /** Release the matching mutation after its external side effects settle. */
    default void completeAdmissionMutation(
            String requestId, CompletableFuture<?> future) {
    }

    /**
     * Attach the admission lease to the exact inflight item registered by the
     * successful plan commit.  WorkerStatus is delivered to the registrar,
     * so the lease must live on the same generation-fenced entry instead of
     * only in the admission scheduler's local callback closure.
     *
     * @return {@code true} when the lease was attached to the live entry;
     *         {@code false} when the entry already reached a terminal state.
     *         A false result carries no Decode-acceptance implication.
     */
    boolean attachAdmissionLease(BatchItem item, AdmissionLease lease);

    /** Remove a previously registered item (offer failed, plan aborted). */
    void unregisterInflight(BatchItem item);

    /**
     * Transfer a successfully delivered request to the scheduler-owned
     * Engine-Cancel reconciliation path.
     *
     * <p>This operation never releases Prefill or Decode accounting by
     * itself.  The request remains charged until the engine provides an
     * authoritative terminal proof ({@code TOMBSTONED}, typed priority
     * {@code CANCELED}, or an ordinary Decode terminal).  The ownership
     * decision is serialized with priority preemption on the request entry,
     * so the two control protocols can never issue independently.</p>
     */
    PostDeliveryFenceResult fenceAfterDeliveryTimeout(BatchItem item, String detail);

    /**
     * Reduce a non-successful externally visible request future against the
     * scheduler's delivery ownership.
     *
     * <p>The reducer is the linearization point between caller cancellation,
     * delivery and priority preemption. A batch delivery claim, or a route
     * decision whose publication has already been claimed, must retain all
     * accounting behind Engine-Cancel reconciliation. A route decision which
     * has not crossed publication may be rolled back locally. The registrar
     * performs either action completely; it never returns while a live entry
     * has an ownerless cleanup marker.</p>
     *
     * @return {@code true} when the exact live entry consumed the terminal and
     *         now owns cleanup or reconciliation; {@code false} only when that
     *         entry generation is already absent, allowing the lease to run
     *         its idempotent fallback release
     */
    boolean reduceExternalFutureTerminal(BatchItem item, String detail);

    /**
     * Drive an evicted victim to its terminal state (design doc 9.5/17.3):
     * release its decode reservation, complete its future with
     * {@code PRIORITY_PREEMPTED} and tombstone the request id. Reserved for
     * victims the engine has already accepted (contract 5.3). Idempotent —
     * repeated calls (or races with other terminal paths) take effect once.
     */
    void finishPreempted(BatchItem victim, String detail);

    /**
     * {@link #finishPreempted} addressed by request id, for victims whose
     * {@code BatchItem} is not at hand (design doc 11.5). No-op when the id
     * is not inflight — the victim already reached a terminal state;
     * idempotent like {@code finishPreempted}.
     */
    void finishPreemptedById(String requestId, String detail);

    /**
     * Drive a yielded victim — one the engine never saw (prefill queue
     * eviction or decode reserved-only eviction, contract 5.3) — to its
     * terminal state: same idempotent release/tombstone chain as
     * {@link #finishPreempted}, but the client-visible terminal is the
     * retryable {@code NO_AVAILABLE_WORKER} with the yield reason.
     */
    void finishYielded(BatchItem victim, String detail);

    /**
     * {@link #finishYielded} addressed by request id, for decode
     * reserved-only victims whose {@code BatchItem} is not at hand. No-op
     * when the id is not inflight; idempotent like {@code finishYielded}.
     */
    void finishYieldedById(String requestId, String detail);

    /** Atomically attach one victim to a token before endpoint mutation. */
    boolean claimForPreemption(String requestId, long attemptToken, String detail);

    /** Roll back a claim when no Cancel RPC has been issued. */
    boolean releasePreemptionClaim(String requestId, long attemptToken);

    /** CLAIMED -> CANCEL_IN_FLIGHT; called for every victim before the first RPC. */
    boolean markPreemptionCancelInFlight(String requestId, long attemptToken);

    /** Cancel ACCEPTED; does not complete the victim or release resources. */
    boolean markPreemptionCancelAccepted(String requestId, long attemptToken);

    /** Explicit negative acknowledgement; keeps a stale reconciliation fence. */
    boolean markPreemptionNotFound(String requestId, long attemptToken);

    /** Transport result unknown; preserves attribution and accounting. */
    boolean markPreemptionUnknown(String requestId, long attemptToken);

    /**
     * Signal completed only by original-Prefill WorkerStatus carrying
     * priority_preemption_progress=CANCELED and exact code 8429.
     */
    CompletableFuture<PriorityCanceledObservation> priorityCanceledSignal(
            String requestId, long attemptToken);

    /** Token-fenced terminal settlement after the endpoint accounting CAS wins. */
    boolean finishPreemptedById(String requestId, long attemptToken, String detail);

    /**
     * Token-fenced settlement for an engine {@code TOMBSTONED} acknowledgement.
     * Unlike {@link #finishPreemptedById(long, long, String)}, this proof does
     * not wait for WorkerStatus: the engine atomically proved absence and
     * fenced every racing late enqueue for the same request id.
     */
    boolean finishTombstonedById(String requestId, long attemptToken, String detail);

    /** Fresh active observation reopens a NOT_FOUND_STALE victim. */
    boolean reconcilePreemptionActive(String requestId);

    /**
     * Resolve the original Prefill route from the authoritative inflight
     * entry. Returning {@code null} means the request is no longer inflight or
     * its Prefill route is unavailable.
     */
    EngineCancelChannel.CancelTarget resolveCancelTarget(String requestId);

    record PriorityCanceledObservation(String requestId, long errorCode) {
    }

    enum PostDeliveryFenceResult {
        /** Decode ownership was already authoritative; no cancel was issued. */
        ENGINE_OWNED,
        /** This call installed and started the request-scoped Engine fence. */
        STARTED,
        /** An Engine fence was already reconciling this request. */
        ALREADY_FENCED,
        /** Priority preemption owns reconciliation; the timeout joined it. */
        JOINED_PREEMPTION,
        /** The exact inflight generation had already reached a terminal state. */
        ALREADY_TERMINAL
    }

}
