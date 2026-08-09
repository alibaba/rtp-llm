package org.flexlb.balance.scheduler.priority;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.scheduler.BatchItem;
import org.flexlb.balance.scheduler.PrefillQueueManager;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.util.Logger;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * AutoCloseable admission lease — the single ownership boundary between the
 * Auto-TPM admission scheduler and the dispatch/completion pipeline
 * (Luoli redesign §2.2).
 *
 * <p>Created at every plan-commit success point and bound to the request's
 * future. Two terminal operations, mutually exclusive via a CAS on
 * {@link #settled}:
 * <ul>
 *   <li>{@link #handoverToEngine()} — the <b>success</b> path: the future
 *       completed successfully, so the engine now owns the decode reservation
 *       and the prefill queue item will be consumed by the batcher's dispatch
 *       loop. The lease is sealed <em>without touching any resource</em> —
 *       releasing here would reopen the N2 oversell window (a new admission
 *       could grab the same decode slot while the engine is still using it).</li>
 *   <li>{@link #close()} — the <b>failure</b> path (timeout, dispatch error,
 *       SLO expiry, eviction): {@code tryRemove} the item from the prefill
 *       queue (idempotent — already dispatched/dispatched-and-removed items
 *       are a no-op), {@code release} the decode reservation (idempotent
 *       ConcurrentHashMap remove), and {@code unregisterInflight} (idempotent
 *       CAS remove). All three are safe to call on an already-settled item.</li>
 * </ul>
 *
 * <p>The dispatch pipeline's own terminal paths (onSuccess / onFailure /
 * onTimeout / onExpired / onOfferFailure) also clean up resources, each
 * guarded by its own CAS ({@code rollbackOnce}, {@code isDone} checks,
 * ConcurrentHashMap value-equality removes). The lease's CAS adds a second
 * exactly-once boundary so that an {@code orTimeout} firing before the
 * dispatch pipeline reaches the item still releases the stuck reservation —
 * and when the dispatch pipeline later settles the same item, every shared
 * cleanup step is an idempotent no-op (design §2.2: "已 dispatch 的请求撞上
 * 超时属于竞争窗口：close() 三步幂等无害").
 *
 * <p><b>Legacy path</b> ({@code budget == null}): never constructs a lease;
 * the legacy dispatch lifecycle is unchanged byte-for-byte.
 */
public final class AdmissionLease implements AutoCloseable {

    private final AtomicBoolean settled = new AtomicBoolean(false);
    private final BatchItem item;
    private final DecodeEndpoint decodeEp;
    private final PrefillQueueManager prefillQueue;
    private final InflightRegistrar registrar;

    /**
     * @param item         the committed batch item (inflight-registered, queued)
     * @param decodeEp     the decode endpoint holding the reservation
     *                     ({@code null} when the plan has no decode endpoint)
     * @param prefillQueue the prefill queue manager (for tryRemove on failure)
     * @param registrar    the inflight registrar (for unregisterInflight on failure)
     */
    public AdmissionLease(BatchItem item,
                           DecodeEndpoint decodeEp,
                           PrefillQueueManager prefillQueue,
                           InflightRegistrar registrar) {
        this.item = item;
        this.decodeEp = decodeEp;
        this.prefillQueue = prefillQueue;
        this.registrar = registrar;
    }

    /**
     * Failure / cleanup path: CAS exactly-once, then release every resource
     * the admission held. Each step is idempotent so a concurrent dispatch-
     * pipeline terminal path (or a second close) is harmless.
     */
    @Override
    public void close() {
        if (!settled.compareAndSet(false, true)) {
            return;
        }
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
        Logger.info("[auto-tpm] admission lease closed: request_id={}",
                item.requestId());
    }

    /**
     * Success path: seal the lease without touching any resource. The engine
     * now owns the decode reservation; the prefill queue item will be consumed
     * by the batcher's dispatch loop. Releasing here would reopen the N2
     * oversell window (design §2.2).
     */
    public void handoverToEngine() {
        if (!settled.compareAndSet(false, true)) {
            return;
        }
        Logger.info("[auto-tpm] admission lease handed over to engine: request_id={}",
                item.requestId());
    }

    /**
     * Bind the lease to the request future: on success →
     * {@link #handoverToEngine()} (seal, no resource release); on any
     * failure/timeout → {@link #close()} (release everything). The CAS on
     * {@link #settled} guarantees that exactly one of the two runs, even
     * if the future completes while the dispatch pipeline is mid-cleanup.
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
}
