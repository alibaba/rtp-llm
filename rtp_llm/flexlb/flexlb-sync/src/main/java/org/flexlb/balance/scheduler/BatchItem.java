package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * A single inference request queued for batch dispatch.
 *
 * <p>Extracted from {@link FlexlbBatchScheduler} to reduce coupling
 * with {@link WorkerBatcher}.
 *
 * <p>Carries direct {@link PrefillEndpoint} / {@link DecodeEndpoint} references
 * so downstream operations (commit, rollback, ack) avoid repeated
 * {@code EndpointRegistry} lookups by ip+port.
 *
 * <p>{@link #sortKey} is mutable — {@link FixedWindowBatcherAlgorithm} computes it
 * inside {@link WorkerBatcher#offer(BatchItem)} via {@link FixedWindowBatcherAlgorithm#computeSortKey}.
 */
public final class BatchItem {

    private final BalanceContext ctx;
    private final CompletableFuture<Response> future;
    private final Response routeResponse;
    private final ServerStatus prefill;
    private final ServerStatus decode;
    private final PrefillEndpoint prefillEp;
    private final DecodeEndpoint decodeEp;
    private final long enqueuedAtMs;

    /** Mutable sort key set by the batcher algorithm at offer time. */
    private volatile long sortKey;

    /**
     * Batch ID assigned in {@code FlexlbBatchScheduler#flushItems} just before
     * dispatch. Used for stale-ACK detection ({@code onSuccess} compares the
     * incoming batch ID against this field). 0 = not yet dispatched.
     */
    private volatile long assignedBatchId;

    /**
     * Wall-clock timestamp (ms) set just before {@code dispatcher.dispatch}.
     * Used for the dispatch-to-ACK latency metric. 0 = not yet dispatched.
     */
    private volatile long dispatchedAtMs;

    /**
     * CAS flag ensuring the decode reservation is rolled back at most once
     * ({@code FlexlbBatchScheduler#rollbackOnce}).
     */
    final AtomicBoolean rolledBack = new AtomicBoolean(false);

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

    /** Priority queue sort key. */
    public long sortKey() { return sortKey; }

    /** Set by {@link WorkerBatcher#offer} after {@link FixedWindowBatcherAlgorithm#computeSortKey}. */
    public void setSortKey(long sortKey) { this.sortKey = sortKey; }

    /** Batch ID assigned at dispatch time; 0 = not yet dispatched. */
    public long assignedBatchId() { return assignedBatchId; }

    /** Set by {@code FlexlbBatchScheduler#flushItems} before dispatch. */
    public void setAssignedBatchId(long assignedBatchId) { this.assignedBatchId = assignedBatchId; }

    /** Wall-clock dispatch timestamp (ms); 0 = not yet dispatched. */
    public long dispatchedAtMs() { return dispatchedAtMs; }

    /** Set by {@code FlexlbBatchScheduler#flushItems} just before {@code dispatcher.dispatch}. */
    public void setDispatchedAtMs(long dispatchedAtMs) { this.dispatchedAtMs = dispatchedAtMs; }

    // -- derived accessors --

    public long requestId() {
        return ctx != null && ctx.getRequest() != null
                ? ctx.getRequest().getRequestId() : 0;
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
