package org.flexlb.balance.scheduler.priority;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.scheduler.BatchItem;
import org.flexlb.balance.scheduler.PrefillQueueManager;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.util.Logger;

import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * AutoCloseable admission lease — the single ownership boundary between the
 * Auto-TPM admission scheduler and the delivery/completion pipeline.
 *
 * <p><b>Four-state CAS</b>: the original
 * two-state {@code settled} flag sealed the lease on prefill success, making
 * {@code close()} a permanent no-op and leaking KV cache blocks forever. The
 * state machine records the one ownership decision that matters: an
 * authoritative Decode acceptance transfers ownership even if the Enqueue
 * ACK is lost, delayed, or negative.
 *
 * <ul>
 *   <li>{@link #markDeliverySucceeded()} — the <b>success</b> path:
 *       a Prefill-only route closes directly as engine-owned because there is
 *       no later Decode observation; a route with Decode transitions
 *       {@code PENDING→DELIVERY_WAIT}. In batch mode this follows the Enqueue
 *       acknowledgement; in route-decision mode it follows publication to
 *       the frontend and may precede the frontend's engine request. A
 *       <em>soft timeout</em> reconciles the pending Decode ownership.</li>
 *   <li>{@link #close()} — the <b>failure</b> path
 *       ({@code PENDING→CLOSED} only):
 *       timeout, dispatch error, SLO expiry, eviction or external future
 *       cancellation. The registrar reduces this terminal against delivery
 *       ownership: locally reversible requests are released, while a batch
 *       claim or already-published route remains behind Engine fencing.
 *       Post-delivery cleanup routes through
 *       {@link #reconcileAfterDeliveryTimeout()} or
 *       {@link #markDecodeAccepted()} only.</li>
 *   <li>{@link #reconcileAfterDeliveryTimeout()} — the <b>soft-timeout</b> path
 *       ({@code DELIVERY_WAIT→CLOSED}): transfers cleanup to the
 *       scheduler's request-scoped Engine-Cancel fence. It never releases
 *       Prefill or Decode accounting locally; those ledgers remain charged
 *       until the engine proves a terminal outcome. Decode acceptance and
 *       the first Cancel invocation are linearized by the scheduler on the
 *       same inflight entry.</li>
 *   <li>{@link #markDecodeAccepted()} — the <b>decode-accepted</b> path
 *       ({@code PENDING/DELIVERY_WAIT→CLOSED_ENGINE_OWNED}). It never releases
 *       resources: the Decode engine has authoritative ownership.</li>
 * </ul>
 *
 * <p>Each resource-release step is idempotent, so concurrent terminal paths
 * (dispatch pipeline, calibrate, soft timeout) are harmless.
 *
 * <p><b>Legacy path</b> ({@code budget == null}): never constructs a lease;
 * the legacy dispatch lifecycle is unchanged byte-for-byte.
 */
public final class AdmissionLease implements AutoCloseable {

    // ==================== Four-state CAS ====================

    private static final int STATE_PENDING = 0;
    private static final int STATE_DELIVERY_WAIT = 1;
    private static final int STATE_CLOSED = 2;
    private static final int STATE_CLOSED_ENGINE_OWNED = 3;

    @FunctionalInterface
    public interface SoftTimeoutScheduler {
        ScheduledFuture<?> schedule(AdmissionLease lease,
                                    Runnable task,
                                    long delay,
                                    TimeUnit unit);

        /**
         * Remove a lease from the scheduler's lifecycle registry after any
         * terminal transition. Implementations without lifecycle tracking may
         * keep the default no-op.
         */
        default void onLeaseTerminated(AdmissionLease lease) {
        }
    }

    private final AtomicInteger leaseState = new AtomicInteger(STATE_PENDING);
    private final BatchItem item;
    private final DecodeEndpoint decodeEp;
    private final PrefillQueueManager prefillQueue;
    private final InflightRegistrar registrar;
    private final long softTimeoutMs;
    private final Runnable onCloseCallback;
    private final SoftTimeoutScheduler softTimeoutScheduler;
    private volatile ScheduledFuture<?> softTimeoutFuture;

    /**
     * Full constructor with soft-timeout and backpressure callback.
     *
     * @param item           the committed batch item (inflight-registered, queued)
     * @param decodeEp       the decode endpoint holding the reservation
     *                       ({@code null} when the plan has no decode endpoint)
     * @param prefillQueue   the prefill queue manager (for tryRemove on failure)
     * @param registrar      the inflight registrar (for unregisterInflight on failure)
     * @param softTimeoutMs  post-delivery soft timeout in ms; {@code <= 0} disables
     * @param onCloseCallback called exactly once when the lease transitions to CLOSED
     *                        (may be {@code null})
     * @param softTimeoutScheduler scheduler owned by the admission-scheduler bean;
     *                             required when {@code softTimeoutMs > 0}
     */
    public AdmissionLease(BatchItem item,
                          DecodeEndpoint decodeEp,
                          PrefillQueueManager prefillQueue,
                          InflightRegistrar registrar,
                          long softTimeoutMs,
                          Runnable onCloseCallback,
                          SoftTimeoutScheduler softTimeoutScheduler) {
        this.item = item;
        this.decodeEp = decodeEp;
        this.prefillQueue = prefillQueue;
        this.registrar = registrar;
        this.softTimeoutMs = softTimeoutMs;
        this.onCloseCallback = onCloseCallback;
        this.softTimeoutScheduler = softTimeoutMs > 0
                ? Objects.requireNonNull(softTimeoutScheduler,
                        "softTimeoutScheduler is required when soft timeout is enabled")
                : softTimeoutScheduler;
    }

    /**
     * Backward-compatible constructor (soft timeout disabled, no close callback).
     */
    public AdmissionLease(BatchItem item,
                          DecodeEndpoint decodeEp,
                          PrefillQueueManager prefillQueue,
                          InflightRegistrar registrar) {
        this(item, decodeEp, prefillQueue, registrar, 0, null, null);
    }

    // ==================== Terminal operations ====================

    /**
     * Failure / cleanup path: CAS PENDING→CLOSED only. Post-delivery
     * cleanup (from DELIVERY_WAIT) is not allowed here — it routes through
     * {@link #reconcileAfterDeliveryTimeout()} or {@link #markDecodeAccepted()}.
     * Each step is idempotent so a concurrent dispatch-pipeline terminal
     * path (or a second close) is harmless.
     */
    @Override
    public void close() {
        // Only PENDING→CLOSED (failure path). A Decode acceptance that
        // won first is authoritative even when Enqueue later reports failure;
        // post-delivery cleanup routes through reconcileAfterDeliveryTimeout().
        if (!leaseState.compareAndSet(STATE_PENDING, STATE_CLOSED)) {
            return;
        }
        // try-finally: ensure the lifecycle callback runs even if cancelSoftTimeout()
        // or releaseResources() throws. Without this, a thrown exception would
        // leak the active-admission backpressure counter.
        try {
            cancelSoftTimeout();
            settleExternalFutureTerminal("admission_future_terminal");
        } catch (Exception e) {
            Logger.error("[auto-tpm] admission lease close error: request_id={} error={}",
                    item.requestId(), e.getMessage(), e);
        } finally {
            notifyTermination();
        }
        Logger.debug("[auto-tpm] admission lease closed: request_id={}",
                item.requestId());
    }

    /**
     * Record a successful delivery. Unless Decode acceptance already
     * closed the lease as engine-owned, wait for authoritative ownership and
     * arm the post-delivery soft timeout.
     */
    public void markDeliverySucceeded() {
        // A Prefill-only route has no later Decode-acceptance observation. Its
        // successful delivery is therefore the ownership terminal itself; do
        // not strand the admission permit in DELIVERY_WAIT forever.
        if (decodeEp == null) {
            if (leaseState.compareAndSet(STATE_PENDING, STATE_CLOSED_ENGINE_OWNED)) {
                finishEngineOwned();
            }
            return;
        }
        if (leaseState.compareAndSet(STATE_PENDING, STATE_DELIVERY_WAIT)) {
            Logger.debug("[auto-tpm] admission lease awaiting decode acceptance: request_id={}",
                    item.requestId());
            scheduleSoftTimeout();
        }
    }

    /**
     * Soft-timeout path after a successful delivery.
     *
     * <p>The lease closes its admission-backpressure slot, but deliberately
     * does not touch queue, Prefill, Decode, or inflight ledgers. The
     * registrar owns the exact entry generation and reconciles an actual
     * Engine Cancel until an authoritative terminal proof is observed. This
     * avoids both the unregister-before-cancel hole and the more serious
     * early-capacity-release window while a frontend enqueue is still racing.</p>
     */
    public void reconcileAfterDeliveryTimeout() {
        if (!leaseState.compareAndSet(STATE_DELIVERY_WAIT, STATE_CLOSED)) {
            return;
        }
        try {
            cancelSoftTimeout();
            InflightRegistrar.PostDeliveryFenceResult result =
                    registrar.fenceAfterDeliveryTimeout(item, "post_delivery_soft_timeout");
            if (result == InflightRegistrar.PostDeliveryFenceResult.ENGINE_OWNED) {
                // Diagnostic state only: the callback is still notified by
                // this method exactly once. Resource ownership stays Engine-side.
                leaseState.compareAndSet(
                        STATE_CLOSED, STATE_CLOSED_ENGINE_OWNED);
            }
            Logger.debug("[auto-tpm] admission lease post-delivery fence: "
                            + "request_id={} result={}",
                    item.requestId(), result);
        } catch (Exception e) {
            Logger.error("[auto-tpm] admission lease delivery reconciliation error: "
                            + "request_id={} lease_state={} error={}",
                    item.requestId(), leaseState.get(), e.getMessage(), e);
        } finally {
            notifyTermination();
        }
    }

    /**
     * Record authoritative Decode ownership. It closes engine-owned whether
     * Enqueue is unresolved or already acknowledged. A later Enqueue failure
     * cannot change this state into cleanup ownership.
     */
    public void markDecodeAccepted() {
        while (true) {
            int state = leaseState.get();
            if (state != STATE_PENDING && state != STATE_DELIVERY_WAIT) {
                return;
            }
            if (leaseState.compareAndSet(state, STATE_CLOSED_ENGINE_OWNED)) {
                finishEngineOwned();
                return;
            }
        }
    }

    /**
     * Record authoritative request settlement by the owning scheduler.
     *
     * <p>This operation owns only the lease lifecycle: it cancels the soft
     * timeout and releases admission backpressure. Endpoint, queue and
     * inflight cleanup remain with the scheduler terminal reducer, so calling
     * it after a successful public response cannot release engine resources
     * twice.</p>
     */
    public void markRequestSettled() {
        while (true) {
            int state = leaseState.get();
            if (state != STATE_PENDING && state != STATE_DELIVERY_WAIT) {
                return;
            }
            if (leaseState.compareAndSet(state, STATE_CLOSED)) {
                try {
                    cancelSoftTimeout();
                } finally {
                    notifyTermination();
                }
                return;
            }
        }
    }

    private void finishEngineOwned() {
        // try-finally: ensure the lifecycle callback runs even if cancelSoftTimeout()
        // throws. Without this, the counter leaks.
        try {
            cancelSoftTimeout();
            Logger.debug("[auto-tpm] admission lease marked engine-owned: request_id={}",
                    item.requestId());
        } catch (Exception e) {
            Logger.error("[auto-tpm] admission lease markDecodeAccepted error: "
                            + "request_id={} error={}",
                    item.requestId(), e.getMessage(), e);
        } finally {
            notifyTermination();
        }
    }

    /**
     * Bind the lease to the request future: on success →
     * {@link #markDeliverySucceeded()} (seal + schedule soft timeout); on any
     * failure/timeout → {@link #close()} (release only while ownership is
     * still pending). The CAS on
     * {@link #leaseState} guarantees that exactly one terminal path executes
     * the resource release, even if the future completes while the dispatch
     * pipeline is mid-cleanup.
     */
    public void bindTo(CompletableFuture<Response> future) {
        future.whenComplete((resp, err) -> {
            if (err == null && resp != null && resp.isSuccess()) {
                markDeliverySucceeded();
            } else {
                close();
            }
        });
    }

    // ==================== Internal ====================

    /**
     * Release all resources held by the admission. Each step is idempotent.
     */
    private void releaseResources() {
        // 1. Remove from prefill queue (no-op if already dispatched/removed).
        if (prefillQueue != null) {
            prefillQueue.tryRemove(item.requestId(), "LEASE_RELEASE");
        }
        // 2. Release decode reservation (no-op if already released).
        if (decodeEp != null && item.decode() != null) {
            decodeEp.release(item.decode().getRequestId());
        }
        // 3. Unregister from inflight (no-op if already removed/tombstoned).
        registrar.unregisterInflight(item);
    }

    /**
     * Reduce the external future terminal before touching any resource. The
     * registrar serializes this decision with delivery and preemption. It
     * either completes a safe local rollback itself or retains the request
     * behind the appropriate reconciliation owner.
     *
     */
    private void settleExternalFutureTerminal(String detail) {
        if (registrar.reduceExternalFutureTerminal(item, detail)) {
            Logger.debug("[auto-tpm] admission future terminal reduced by scheduler: "
                            + "request_id={} reason={}",
                    item.requestId(), detail);
            return;
        }
        releaseResources();
    }

    /**
     * Schedule the post-delivery soft timeout. The scheduler, rather than this
     * timer thread, decides whether Decode acceptance or Engine fencing owns
     * the request. That single request-entry lock removes the former
     * check-then-CAS race.
     */
    private void scheduleSoftTimeout() {
        if (softTimeoutMs <= 0) {
            return;
        }
        if (decodeEp == null) {
            // No decode endpoint — nothing to soft-timeout.
            return;
        }
        ScheduledFuture<?> scheduled;
        try {
            scheduled = softTimeoutScheduler.schedule(
                    this,
                    this::reconcileAfterDeliveryTimeout,
                    softTimeoutMs, TimeUnit.MILLISECONDS);
        } catch (RejectedExecutionException shutdown) {
            // The owning admission scheduler is already shutting down. Do not
            // invoke the registrar/endpoint reconciliation path after its bean
            // lifecycle has closed; only release admission backpressure.
            if (leaseState.compareAndSet(STATE_DELIVERY_WAIT, STATE_CLOSED)) {
                notifyTermination();
            }
            Logger.debug("[auto-tpm] skip soft timeout after scheduler shutdown: request_id={}",
                    item.requestId());
            return;
        } catch (RuntimeException scheduleFailure) {
            // A scheduler implementation must not be able to strand a
            // successful admission in DELIVERY_WAIT. Delivery ownership has
            // already crossed its boundary, so release only admission
            // backpressure and leave Engine/resource reconciliation untouched.
            if (leaseState.compareAndSet(STATE_DELIVERY_WAIT, STATE_CLOSED)) {
                notifyTermination();
            }
            Logger.error("[auto-tpm] soft timeout scheduling failed: request_id={} error={}",
                    item.requestId(), scheduleFailure.getMessage(), scheduleFailure);
            return;
        }
        softTimeoutFuture = scheduled;
        // Decode acceptance may close the lease after schedule() enqueues the
        // task but before the volatile handle above is published. Its cancel
        // then legitimately observes null. Rechecking after publication makes
        // that race self-healing and avoids retaining this lease to the 30s
        // fallback deadline.
        if (leaseState.get() != STATE_DELIVERY_WAIT) {
            scheduled.cancel(false);
            if (softTimeoutFuture == scheduled) {
                softTimeoutFuture = null;
            }
            softTimeoutScheduler.onLeaseTerminated(this);
        }
    }

    private void cancelSoftTimeout() {
        ScheduledFuture<?> f = softTimeoutFuture;
        if (f != null) {
            f.cancel(false);  // don't interrupt a running task
            softTimeoutFuture = null;
        }
    }

    /**
     * Lifecycle-only terminal used by the owning scheduler during bean
     * shutdown. It deliberately does not call registrar or endpoint cleanup:
     * those components have their own ordered shutdown, while this lease owns
     * only its timer and admission permit at this boundary.
     */
    void terminateForSchedulerShutdown() {
        while (true) {
            int state = leaseState.get();
            if (state != STATE_PENDING && state != STATE_DELIVERY_WAIT) {
                return;
            }
            if (leaseState.compareAndSet(state, STATE_CLOSED)) {
                try {
                    cancelSoftTimeout();
                } finally {
                    notifyTermination();
                }
                return;
            }
        }
    }

    private void notifyTermination() {
        try {
            if (softTimeoutScheduler != null) {
                softTimeoutScheduler.onLeaseTerminated(this);
            }
        } finally {
            if (onCloseCallback != null) {
                onCloseCallback.run();
            }
        }
    }

    // ==================== Test-visible state ====================

    /**
     * Returns the current lease state (for testing/diagnostics).
     *
     * @return 0=PENDING, 1=DELIVERY_WAIT, 2=CLOSED,
     *         3=CLOSED_ENGINE_OWNED
     */
    int leaseState() {
        return leaseState.get();
    }

}
