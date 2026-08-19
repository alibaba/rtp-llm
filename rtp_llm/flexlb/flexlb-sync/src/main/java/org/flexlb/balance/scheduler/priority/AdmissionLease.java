package org.flexlb.balance.scheduler.priority;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.scheduler.BatchItem;
import org.flexlb.balance.scheduler.PrefillQueueManager;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.util.Logger;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Tracks the admission ownership handoff from the local queue to the engines.
 * Before EnqueueBatch succeeds, this lease may roll back queue/Decode/inflight
 * resources. After success, ownership is ambiguous until Decode WorkerStatus
 * confirms it; a timeout must therefore enter the scheduler's authoritative
 * Engine reconciliation instead of releasing local accounting optimistically.
 *
 * <p><b>Legacy path</b> ({@code budget == null}): never constructs a lease;
 * the legacy dispatch lifecycle is unchanged byte-for-byte.
 */
public final class AdmissionLease implements AutoCloseable {

    // ==================== Ownership state CAS ====================

    private static final int STATE_PENDING = 0;
    private static final int STATE_HANDOVER_WAIT = 1;
    private static final int STATE_CLOSED_CLEANUP = 2;
    private static final int STATE_CLOSED_ENGINE_OWNED = 3;
    private static final int STATE_RECONCILING = 4;

    /**
     * Daemon single-thread scheduler for post-success soft timeouts. A daemon
     * thread never prevents JVM shutdown.
     */
    private static final ScheduledExecutorService SOFT_TIMEOUT_EXECUTOR =
            Executors.newSingleThreadScheduledExecutor(r -> {
                Thread t = new Thread(r, "admission-lease-soft-timeout");
                t.setDaemon(true);
                return t;
            });

    private final AtomicInteger leaseState = new AtomicInteger(STATE_PENDING);
    private final BatchItem item;
    private final DecodeEndpoint decodeEp;
    private final PrefillQueueManager prefillQueue;
    private final InflightRegistrar registrar;
    private final long softTimeoutMs;
    private final Runnable onCloseCallback;
    private volatile ScheduledFuture<?> softTimeoutFuture;

    /**
     * Full constructor with soft-timeout and backpressure callback.
     *
     * @param item           the committed batch item (inflight-registered, queued)
     * @param decodeEp       the decode endpoint holding the reservation
     *                       ({@code null} when the plan has no decode endpoint)
     * @param prefillQueue   the prefill queue manager (for tryRemove on failure)
     * @param registrar      the inflight registrar (for unregisterInflight on failure)
     * @param softTimeoutMs  post-success soft timeout in ms; {@code <= 0} disables
     * @param onCloseCallback called exactly once when the lease transitions to CLOSED
     *                       (may be {@code null})
     */
    public AdmissionLease(BatchItem item,
                          DecodeEndpoint decodeEp,
                          PrefillQueueManager prefillQueue,
                          InflightRegistrar registrar,
                          long softTimeoutMs,
                          Runnable onCloseCallback) {
        this.item = item;
        this.decodeEp = decodeEp;
        this.prefillQueue = prefillQueue;
        this.registrar = registrar;
        this.softTimeoutMs = softTimeoutMs;
        this.onCloseCallback = onCloseCallback;
    }

    /**
     * Backward-compatible constructor (soft timeout disabled, no close callback).
     */
    public AdmissionLease(BatchItem item,
                          DecodeEndpoint decodeEp,
                          PrefillQueueManager prefillQueue,
                          InflightRegistrar registrar) {
        this(item, decodeEp, prefillQueue, registrar, 0, null);
    }

    // ==================== Terminal operations ====================

    /**
     * Failure / cleanup path: CAS PENDING→CLOSED_CLEANUP only. Post-handover
     * cleanup (from HANDOVER_WAIT) is not allowed here — it routes through
     * {@link #forceCloseAfterHandover()} or {@link #markDecodeAccepted()}.
     * Each step is idempotent so a concurrent dispatch-pipeline terminal
     * path (or a second close) is harmless.
     */
    @Override
    public void close() {
        // Only PENDING→CLOSED_CLEANUP (failure path). A Decode acceptance that
        // won first is authoritative even when Enqueue later reports failure;
        // post-handover cleanup routes through forceCloseAfterHandover().
        if (!leaseState.compareAndSet(STATE_PENDING, STATE_CLOSED_CLEANUP)) {
            return;
        }
        // try-finally: ensure notifyCloseCallback() runs even if cancelSoftTimeout()
        // or releaseResources() throws. Without this, a thrown exception would
        // leak the activeLeaseCount backpressure counter (Fix: counter leak).
        try {
            cancelSoftTimeout();
            releaseResourcesIfOwned("admission_future_terminal");
        } catch (Exception e) {
            Logger.error("[auto-tpm] admission lease close error: request_id={} error={}",
                    item.requestId(), e.getMessage(), e);
        } finally {
            notifyCloseCallback();
        }
    }

    /**
     * Record an Enqueue success. Unless Decode acceptance has already closed
     * the lease as engine-owned, wait for acceptance and arm the post-success
     * soft timeout.
     */
    public void handoverToEngine() {
        while (true) {
            int state = leaseState.get();
            if (state == STATE_PENDING) {
                if (!leaseState.compareAndSet(STATE_PENDING, STATE_HANDOVER_WAIT)) {
                    continue;
                }
                scheduleSoftTimeout();
                return;
            }
            return;
        }
    }

    /**
     * Start post-handover reconciliation. No local resource is released here:
     * EnqueueBatch has already succeeded, so only Decode WorkerStatus or the
     * scheduler's Engine Cancel/TOMBSTONED reducer may settle ownership.
     */
    public void forceCloseAfterHandover() {
        if (!leaseState.compareAndSet(STATE_HANDOVER_WAIT, STATE_RECONCILING)) {
            return;
        }
        try {
            cancelSoftTimeout();
            if (decodeEp != null && decodeEp.isConfirmedTracked(item.requestId())) {
                markDecodeAccepted();
                return;
            }
            if (!registrar.requestPostHandoverReconciliation(
                    item, "post_success_soft_timeout")) {
                completeSchedulerSettlement();
            }
        } catch (Exception e) {
            Logger.error("[auto-tpm] admission lease forceCloseAfterHandover error: "
                            + "request_id={} lease_state={} error={}",
                    item.requestId(), leaseState.get(), e.getMessage(), e);
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
            if (state != STATE_PENDING && state != STATE_HANDOVER_WAIT
                    && state != STATE_RECONCILING) {
                return;
            }
            if (leaseState.compareAndSet(state, STATE_CLOSED_ENGINE_OWNED)) {
                finishEngineOwned();
                return;
            }
        }
    }

    /**
     * Close lease bookkeeping after the scheduler has already settled request
     * ownership and resources. This method never releases queue/Decode/inflight
     * state itself; it only cancels the timer and decrements the admission
     * backpressure counter exactly once.
     */
    public void completeSchedulerSettlement() {
        while (true) {
            int state = leaseState.get();
            if (state == STATE_CLOSED_CLEANUP || state == STATE_CLOSED_ENGINE_OWNED) {
                return;
            }
            if (leaseState.compareAndSet(state, STATE_CLOSED_CLEANUP)) {
                try {
                    cancelSoftTimeout();
                } finally {
                    notifyCloseCallback();
                }
                return;
            }
        }
    }

    private void finishEngineOwned() {
        // try-finally: ensure notifyCloseCallback() runs even if cancelSoftTimeout()
        // throws. Without this, the counter leaks (Fix: counter leak).
        try {
            cancelSoftTimeout();
        } catch (Exception e) {
            Logger.error("[auto-tpm] admission lease markDecodeAccepted error: "
                            + "request_id={} error={}",
                    item.requestId(), e.getMessage(), e);
        } finally {
            notifyCloseCallback();
        }
    }

    /**
     * Detach optional lease tracking after the queue handoff without touching
     * queue membership or Decode accounting. The ordinary inflight lifecycle
     * remains authoritative.
     */
    void abandonWithoutCleanup() {
        if (leaseState.compareAndSet(STATE_PENDING, STATE_CLOSED_ENGINE_OWNED)) {
            finishEngineOwned();
        }
    }

    /**
     * Bind the lease to the request future: on success →
     * {@link #handoverToEngine()} (seal + schedule soft timeout); on any
     * failure/timeout → {@link #close()} (release only while ownership is
     * still pending). The CAS on
     * {@link #leaseState} guarantees that exactly one terminal path executes
     * the resource release, even if the future completes while the dispatch
     * pipeline is mid-cleanup.
     */
    public void bindTo(CompletableFuture<Response> future) {
        future.whenComplete((resp, err) -> {
            if (err == null && resp != null && resp.isSuccess()) {
                handoverToEngine();
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
     * Acquire ordinary-cleanup ownership before touching any resource. The
     * registrar serializes this decision with {@code claimForPreemption}; a
     * winning preemption claim retains Decode accounting and owns any later
     * replay of this cleanup.
     *
     * @return true when this lease owns and completed the cleanup
     */
    private boolean releaseResourcesIfOwned(String detail) {
        if (registrar.registrarOwnsAdmissionCleanup(item, detail)) {
            return false;
        }
        releaseResources();
        return true;
    }

    /**
     * Schedule the post-success soft timeout. When it fires, check whether
     * the decode endpoint has accepted the request:
     * <ul>
     *   <li>If accepted ({@code isConfirmedTracked} returns true) →
     *       {@link #markDecodeAccepted()}: decrement the backpressure counter
     *       only (no resource release — the engine owns the decode reservation).</li>
     *   <li>If not accepted → {@link #forceCloseAfterHandover()}: ask the
     *       scheduler to start its Engine reconciliation reducer.</li>
     * </ul>
     */
    private void scheduleSoftTimeout() {
        if (softTimeoutMs <= 0) {
            return;
        }
        if (decodeEp == null) {
            // No decode endpoint — nothing to soft-timeout.
            return;
        }
        long requestId = item.decode() != null
                ? item.decode().getRequestId() : item.requestId();
        softTimeoutFuture = SOFT_TIMEOUT_EXECUTOR.schedule(() -> {
            try {
                if (decodeEp != null && decodeEp.isConfirmedTracked(requestId)) {
                    // Decode accepted — mark lease closed (decrement counter only).
                    markDecodeAccepted();
                    return;
                }
                // Decode not accepted within the soft timeout window — force close.
                forceCloseAfterHandover();
            } catch (Exception e) {
                Logger.error("[auto-tpm] soft timeout task failed: request_id={}"
                                + " lease_state={} error={}",
                        requestId, leaseState.get(), e.getMessage(), e);
            }
        }, softTimeoutMs, TimeUnit.MILLISECONDS);
    }

    private void cancelSoftTimeout() {
        ScheduledFuture<?> f = softTimeoutFuture;
        if (f != null) {
            f.cancel(false);  // don't interrupt a running task
            softTimeoutFuture = null;
        }
    }

    private void notifyCloseCallback() {
        if (onCloseCallback != null) {
            onCloseCallback.run();
        }
    }

    // ==================== Test-visible state ====================

    /**
     * Returns the current lease state (for testing/diagnostics).
     *
     * @return 0=PENDING, 1=HANDOVER_WAIT, 2=CLOSED_CLEANUP,
     *         3=CLOSED_ENGINE_OWNED, 4=RECONCILING
     */
    int leaseState() {
        return leaseState.get();
    }
}
