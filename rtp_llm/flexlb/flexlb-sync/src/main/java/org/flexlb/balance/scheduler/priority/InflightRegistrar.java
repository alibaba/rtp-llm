package org.flexlb.balance.scheduler.priority;

import org.flexlb.balance.scheduler.BatchItem;
import org.flexlb.dao.loadbalance.Response;

import java.util.concurrent.CompletableFuture;

/**
 * Registers Auto-TPM admitted requests into the batch scheduler's inflight
 * tracking so that dispatch, completion, TTL cleanup and rollback treat them
 * exactly like legacy-path requests.
 *
 * <p>Implemented by {@code FlexlbBatchScheduler}; expressed as an interface to
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
     * Keep the request-id admission fence alive while an asynchronous
     * pre-inflight transaction owns provisional resources.
     */
    boolean retainPendingAdmission(long requestId,
                                   CompletableFuture<Response> expectedFuture);

    /** Release a hold acquired by {@link #retainPendingAdmission}. */
    void releasePendingAdmission(long requestId,
                                 CompletableFuture<Response> expectedFuture);

    /**
     * Start the authoritative Engine reconciliation for an EnqueueBatch that
     * was acknowledged but has not appeared in Decode WorkerStatus before
     * the post-handover deadline.
     *
     * <p>The registrar remains the sole owner of inflight, Decode reservation
     * and terminal settlement. The caller must not release those resources
     * before the reconciliation reaches a terminal fence.
     *
     * @return {@code true} when a live scheduler lifecycle owns the request
     *         (including an already-running reconciliation/preemption);
     *         {@code false} when the request is already gone
     */
    boolean requestPostHandoverReconciliation(BatchItem item, String detail);

    /**
     * Atomically decide whether an AdmissionLease cleanup or a priority
     * preemption claim owns the request.
     *
     * <p>{@code true} means the registrar already owns cleanup: normally a
     * preemption claim won and this cleanup was deferred behind it, or another
     * terminal path acquired cleanup first. The caller must not release Decode
     * accounting or unregister the item. {@code false} means this caller won
     * cleanup ownership and may perform its normal idempotent
     * release/unregister sequence. A later {@link #claimForPreemption} for the
     * same inflight entry must then fail.
     */
    boolean registrarOwnsAdmissionCleanup(BatchItem item, String detail);

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
    void finishPreemptedById(long requestId, String detail);

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
    void finishYieldedById(long requestId, String detail);

    /** Atomically attach one victim to a token before endpoint mutation. */
    boolean claimForPreemption(long requestId, long attemptToken, String detail);

    /** Roll back a claim when no Cancel RPC has been issued. */
    boolean releasePreemptionClaim(long requestId, long attemptToken);

    /** CLAIMED -> CANCEL_IN_FLIGHT; called for every victim before the first RPC. */
    boolean markPreemptionCancelInFlight(long requestId, long attemptToken);

    /** Cancel ACCEPTED; does not complete the victim or release resources. */
    boolean markPreemptionCancelAccepted(long requestId, long attemptToken);

    /**
     * The typed Prefill CANCELED completion budget elapsed after an ACCEPTED
     * Cancel. The accepted first-cause remains authoritative; an exact Decode
     * terminal may now prove resource release without reclassifying the victim
     * as an ordinary completion.
     */
    boolean markPreemptionCompletionTimedOut(long requestId, long attemptToken);

    /** Explicit negative acknowledgement; keeps a stale reconciliation fence. */
    boolean markPreemptionNotFound(long requestId, long attemptToken);

    /** Transport result unknown; preserves attribution and accounting. */
    boolean markPreemptionUnknown(long requestId, long attemptToken);

    /**
     * Signal completed only by original-Prefill WorkerStatus carrying
     * priority_preemption_progress=CANCELED and exact code 8429.
     */
    CompletableFuture<PriorityCanceledObservation> priorityCanceledSignal(
            long requestId, long attemptToken);

    /** Token-fenced terminal settlement after the endpoint accounting CAS wins. */
    boolean finishPreemptedById(long requestId, long attemptToken, String detail);

    /**
     * Resolve the original Prefill route from the authoritative inflight
     * entry. Returning {@code null} means the request is no longer inflight or
     * its Prefill route is unavailable.
     */
    EngineCancelChannel.CancelTarget resolveCancelTarget(long requestId);

    record PriorityCanceledObservation(long requestId, long errorCode) {
    }

}
