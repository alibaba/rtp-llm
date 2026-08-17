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
 * AutoCloseable admission lease — the single ownership boundary between the
 * Auto-TPM admission scheduler and the dispatch/completion pipeline
 * (Luoli redesign §2.2).
 *
 * <p><b>Four-state CAS</b>: the original
 * two-state {@code settled} flag sealed the lease on prefill success, making
 * {@code close()} a permanent no-op and leaking KV cache blocks forever. The
 * state machine records the one ownership decision that matters: an
 * authoritative Decode acceptance transfers ownership even if the Enqueue
 * ACK is lost, delayed, or negative.
 *
 * <ul>
 *   <li>{@link #handoverToEngine()} — the <b>success</b> path
 *       ({@code PENDING→HANDOVER_WAIT}):
 *       prefill succeeded, the engine now owns the decode reservation. After
 *       sealing, a <em>soft timeout</em> is scheduled: if the decode endpoint
 *       hasn't accepted the request within {@code softTimeoutMs}, the lease is
 *       force-closed and a cancel signal is sent to the engine.</li>
 *   <li>{@link #close()} — the <b>failure</b> path
 *       ({@code PENDING→CLOSED_CLEANUP} only):
 *       timeout, dispatch error, SLO expiry, eviction. Releases all resources
 *       (tryRemove + release + unregisterInflight). Post-handover cleanup
 *       routes through {@link #forceCloseAfterHandover()} or
 *       {@link #markDecodeAccepted()} only.</li>
 *   <li>{@link #forceCloseAfterHandover()} — the <b>soft-timeout</b> path
 *       ({@code HANDOVER_WAIT→CLOSED_CLEANUP}): same resource release as
 *       {@code close()}, <em>plus</em>
 *       sends a cancel signal ({@code finishYieldedById}) to the prefill
 *       engine so the C++ side releases the {@code con_ref} (KV cache block)
 *       that would otherwise stay pinned. Includes a TOCTOU double-check:
 *       if decode accepted between the soft-timeout lambda's
 *       {@code isConfirmedTracked} check and this CAS, only the counter is
 *       decremented (no resource release, no cancel signal).</li>
 *   <li>{@link #markDecodeAccepted()} — the <b>decode-accepted</b> path
 *       ({@code PENDING/HANDOVER_WAIT→CLOSED_ENGINE_OWNED}). It never releases
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
    private static final int STATE_HANDOVER_WAIT = 1;
    private static final int STATE_CLOSED_CLEANUP = 2;
    private static final int STATE_CLOSED_ENGINE_OWNED = 3;

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
        Logger.debug("[auto-tpm] admission lease closed: request_id={}",
                item.requestId());
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
                Logger.debug("[auto-tpm] admission lease awaiting decode acceptance: request_id={}",
                        item.requestId());
                scheduleSoftTimeout();
                return;
            }
            return;
        }
    }

    /**
     * Soft-timeout force-close path: CAS HANDED_OVER→CLOSED. Releases all
     * resources <em>and</em> sends a cancel signal ({@code finishYieldedById})
     * to the prefill engine, triggering the C++ side to release the
     * {@code con_ref} (KV cache block) that was pinned by the successful
     * prefill but never consumed by decode.
     *
     * <p>TOCTOU fix: after the CAS succeeds, re-check
     * {@code isConfirmedTracked}. If decode accepted between the
     * soft-timeout lambda's check and this CAS, only decrement the counter
     * (no resource release, no cancel signal) — the engine owns the decode
     * reservation.
     */
    public void forceCloseAfterHandover() {
        if (!leaseState.compareAndSet(STATE_HANDOVER_WAIT, STATE_CLOSED_CLEANUP)) {
            return;
        }
        // try-finally: ensure notifyCloseCallback() runs even if any intermediate
        // step (cancelSoftTimeout, isConfirmedTracked, releaseResources,
        // finishYieldedById) throws. Without this, a thrown exception would
        // leak the activeLeaseCount backpressure counter (Fix: counter leak).
        try {
            cancelSoftTimeout();
            // TOCTOU fix: decode may have accepted between the isConfirmedTracked
            // check in the soft-timeout lambda and this CAS. Re-check here — if
            // decode has accepted, only decrement the counter (don't release
            // resources or send cancel signal).
            if (decodeEp != null && decodeEp.isConfirmedTracked(item.requestId())) {
                Logger.debug("[auto-tpm] admission lease force-closed after handover "
                                + "(decode accepted, TOCTOU): request_id={}",
                        item.requestId());
            } else {
                // Cleanup and priority Cancel share one registrar-owned
                // linearization point. If Cancel already owns the request,
                // neither the lease nor its follow-up terminal signal may
                // release Decode accounting ahead of typed CANCELED.
                if (releaseResourcesIfOwned("post_success_soft_timeout")) {
                    // Fix B: send cancel signal to engine so C++ releases con_ref.
                    // Only on the soft-timeout path — the normal close() failure path
                    // doesn't need this (the engine already knows the request failed).
                    registrar.finishYieldedById(item.requestId(), "post_success_soft_timeout");
                    Logger.debug("[auto-tpm] admission lease force-closed after handover: "
                                    + "request_id={} reason=post_success_soft_timeout",
                            item.requestId());
                }
            }
        } catch (Exception e) {
            Logger.error("[auto-tpm] admission lease forceCloseAfterHandover error: "
                            + "request_id={} lease_state={} error={}",
                    item.requestId(), leaseState.get(), e.getMessage(), e);
        } finally {
            notifyCloseCallback();
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
            if (state != STATE_PENDING && state != STATE_HANDOVER_WAIT) {
                return;
            }
            if (leaseState.compareAndSet(state, STATE_CLOSED_ENGINE_OWNED)) {
                finishEngineOwned();
                return;
            }
        }
    }

    private void finishEngineOwned() {
        // try-finally: ensure notifyCloseCallback() runs even if cancelSoftTimeout()
        // throws. Without this, the counter leaks (Fix: counter leak).
        try {
            cancelSoftTimeout();
            Logger.debug("[auto-tpm] admission lease marked decode-accepted: request_id={}",
                    item.requestId());
        } catch (Exception e) {
            Logger.error("[auto-tpm] admission lease markDecodeAccepted error: "
                            + "request_id={} error={}",
                    item.requestId(), e.getMessage(), e);
        } finally {
            notifyCloseCallback();
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
            Logger.debug("[auto-tpm] admission lease cleanup deferred to priority preemption: "
                            + "request_id={} reason={}",
                    item.requestId(), detail);
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
     *   <li>If not accepted → {@link #forceCloseAfterHandover()}: release
     *       resources + send cancel signal.</li>
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
            // try-catch: ScheduledExecutorService silently swallows exceptions
            // from failed tasks — without this catch, a thrown exception
            // (e.g. from isConfirmedTracked, releaseResources, or
            // finishYieldedById) would leave the lease stuck in HANDED_OVER
            // forever, leaking the activeLeaseCount counter. The forceClose
            // fallback ensures the CAS transitions to CLOSED even if the
            // first attempt threw mid-way (CAS is idempotent — a second call
            // after a successful CAS harmlessly returns).
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
                // Best-effort fallback: force-close to ensure counter decrement.
                // If the first call's CAS already succeeded (threw after CAS),
                // this call's CAS will fail harmlessly. If it failed before CAS,
                // this call will succeed and close the lease.
                try {
                    forceCloseAfterHandover();
                } catch (Exception fallback) {
                    Logger.error("[auto-tpm] soft timeout fallback force-close"
                                    + " failed: request_id={} error={}",
                            requestId, fallback.getMessage(), fallback);
                }
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
     *         3=CLOSED_ENGINE_OWNED
     */
    int leaseState() {
        return leaseState.get();
    }
}
