package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;

import java.util.concurrent.CompletableFuture;

/**
 * A single inference request queued for batch dispatch.
 *
 * <p>Contains only data needed while the request waits for prefill dispatch.
 * Decode ownership stays in the scheduler as an exact lease.
 */
public final class BatchItem {

    private final BalanceContext ctx;
    private final CompletableFuture<Response> future;
    private final Response routeResponse;
    private final long hitCache;
    private final PrefillEndpoint prefillEp;
    private final long enqueuedAtMs;

    /** Mutable sort key set by the batcher algorithm at offer time. */
    private volatile long sortKey;
    private volatile long batchId;

    public BatchItem(BalanceContext ctx,
                     CompletableFuture<Response> future,
                     Response routeResponse,
                     long hitCache,
                     PrefillEndpoint prefillEp,
                     long enqueuedAtMs) {
        this.ctx = ctx;
        this.future = future;
        this.routeResponse = routeResponse;
        this.hitCache = hitCache;
        this.prefillEp = prefillEp;
        this.enqueuedAtMs = enqueuedAtMs;
    }

    public BalanceContext ctx() { return ctx; }
    public CompletableFuture<Response> future() { return future; }
    public Response routeResponse() { return routeResponse; }
    public PrefillEndpoint prefillEp() { return prefillEp; }
    public long enqueuedAtMs() { return enqueuedAtMs; }

    public long sortKey() { return sortKey; }

    public void setSortKey(long sortKey) { this.sortKey = sortKey; }

    public long batchId() { return batchId; }

    void assignBatchId(long assignedBatchId) {
        if (assignedBatchId <= 0) {
            throw new IllegalArgumentException("batchId must be positive");
        }
        long current = batchId;
        if (current != 0 && current != assignedBatchId) {
            throw new IllegalStateException("item already belongs to batch " + current);
        }
        batchId = assignedBatchId;
    }

    public long requestId() {
        return ctx != null && ctx.getRequest() != null
                ? ctx.getRequest().getRequestId() : 0;
    }

    public long seqLen() {
        return ctx != null && ctx.getRequest() != null
                ? ctx.getRequest().getSeqLen() : 0;
    }

    public long hitCache() {
        return hitCache;
    }

}
