package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.util.Prioritized;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicLong;

/**
 * A single inference request queued for a priority scheduling decision.
 *
 * <p>Extracted from {@link PriorityScheduler} to reduce coupling
 * with {@link WorkerBatcher}. A decision group may be delivered through a
 * batch RPC or returned as individual route decisions.
 *
 * <p>Carries direct {@link PrefillEndpoint} / {@link DecodeEndpoint} references
 * so downstream operations (commit, rollback, ack) avoid repeated
 * {@code EndpointRegistry} lookups by ip+port.
 *
 */
public final class BatchItem implements Prioritized {

    private static final AtomicLong ENQUEUE_SEQUENCE = new AtomicLong();

    private final BalanceContext ctx;
    private final CompletableFuture<Response> future;
    private final Response routeResponse;
    private final ServerStatus prefill;
    private final ServerStatus decode;
    private final PrefillEndpoint prefillEp;
    private final DecodeEndpoint decodeEp;
    private final long enqueuedAtMs;
    private final long enqueueSequence;
    private final DeliveryMode deliveryMode;

    /**
     * Non-null after the batching policy has declared this route decision
     * ready for delivery. Access is serialized by the owning worker batcher's
     * queue lock; keeping the marker on the existing item avoids allocating a
     * wrapper for every request held behind the request-mode inflight cap.
     */
    private String readyDeliveryReason;

    /**
     * Last batching-park diagnostics are owned by the request itself. Keeping the
     * lazily-created mutable holder here makes repeated parks allocation-free
     * while adding only one reference to requests that never park. Unlike a
     * scheduler-wide request-id map, it cannot retain externally removed
     * items. Access is serialized by the owning worker thread / queue lock.
     */
    private ParkTrace parkTrace;

    public BatchItem(BalanceContext ctx,
                     CompletableFuture<Response> future,
                     Response routeResponse,
                     ServerStatus prefill,
                     ServerStatus decode,
                     PrefillEndpoint prefillEp,
                     DecodeEndpoint decodeEp,
                     long enqueuedAtMs) {
        this.ctx = ctx;
        this.future = future;
        this.routeResponse = routeResponse;
        this.prefill = prefill;
        this.decode = decode;
        this.prefillEp = prefillEp;
        this.decodeEp = decodeEp;
        this.enqueuedAtMs = enqueuedAtMs;
        this.enqueueSequence = ENQUEUE_SEQUENCE.incrementAndGet();
        this.deliveryMode = DeliveryMode.from(ctx);
    }

    // -- accessors --

    public BalanceContext ctx() { return ctx; }
    public CompletableFuture<Response> future() { return future; }
    public Response routeResponse() { return routeResponse; }
    public ServerStatus prefill() { return prefill; }
    public ServerStatus decode() { return decode; }
    public PrefillEndpoint prefillEp() { return prefillEp; }
    public DecodeEndpoint decodeEp() { return decodeEp; }
    public long enqueuedAtMs() { return enqueuedAtMs; }
    DeliveryMode deliveryMode() { return deliveryMode; }

    String readyDeliveryReason() { return readyDeliveryReason; }

    void markRouteDecisionReady(String reason) {
        if (deliveryMode != DeliveryMode.ROUTE_DECISION) {
            throw new IllegalStateException(
                    "Only route decisions can enter the ready-delivery backlog");
        }
        readyDeliveryReason = reason == null || reason.isBlank()
                ? "route_decision_ready" : reason;
    }

    void clearRouteDecisionReady() { readyDeliveryReason = null; }

    void recordParkTrace(String reason, long budgetMs, long waitMs,
                         int queueSize, int inflightCount) {
        ParkTrace trace = parkTrace;
        if (trace == null) {
            trace = new ParkTrace();
            parkTrace = trace;
        }
        trace.update(reason, budgetMs, waitMs, queueSize, inflightCount);
    }

    ParkTrace consumeParkTrace() {
        ParkTrace trace = parkTrace;
        parkTrace = null;
        return trace == null ? ParkTrace.EMPTY : trace;
    }

    void clearParkTrace() {
        parkTrace = null;
    }

    boolean hasParkTrace() { return parkTrace != null; }

    static final class ParkTrace {
        private static final ParkTrace EMPTY =
                new ParkTrace("none", -1, -1, -1, -1);

        private String reason;
        private long budgetMs;
        private long waitMs;
        private int queueSize;
        private int inflightCount;

        private ParkTrace() {
        }

        private ParkTrace(String reason, long budgetMs, long waitMs,
                          int queueSize, int inflightCount) {
            update(reason, budgetMs, waitMs, queueSize, inflightCount);
        }

        private void update(String reason, long budgetMs, long waitMs,
                            int queueSize, int inflightCount) {
            this.reason = reason;
            this.budgetMs = budgetMs;
            this.waitMs = waitMs;
            this.queueSize = queueSize;
            this.inflightCount = inflightCount;
        }

        String reason() { return reason; }
        long budgetMs() { return budgetMs; }
        long waitMs() { return waitMs; }
        int queueSize() { return queueSize; }
        int inflightCount() { return inflightCount; }
    }

    /**
     * Normalized request priority. Satisfies {@link Prioritized#priority()}
     * for the per-worker batcher
     * queue's {@code PriorityBlockingQueue}.
     */
    @Override
    public int priority() {
        return ctx != null ? ctx.getPriority() : 0;
    }

    /**
     * Unique monotonic enqueue sequence used by FIFO and as the same-priority
     * tie-break in {@link org.flexlb.util.PriorityOrdering#STRICT}. A re-offer
     * keeps the original item and therefore the original queue position.
     */
    @Override
    public long enqueueSeq() {
        return enqueueSequence;
    }

    // -- derived accessors --

    public String requestId() {
        return ctx != null && ctx.getRequest() != null
                ? ctx.getRequest().getRequestId() : "";
    }

    /** Total sequence length of this request. */
    public long seqLen() {
        return ctx != null && ctx.getRequest() != null
                ? ctx.getRequest().getSeqLen() : 0;
    }

    /** Cache-hit tokens on the assigned prefill endpoint. */
    public long hitCache() {
        return hitCacheOf(prefill);
    }

    /** Extract cache-hit length from a {@link ServerStatus} debug info. */
    private static long hitCacheOf(ServerStatus ss) {
        return ss != null && ss.getDebugInfo() != null
                ? ss.getDebugInfo().getHitCacheLen() : 0;
    }

}
