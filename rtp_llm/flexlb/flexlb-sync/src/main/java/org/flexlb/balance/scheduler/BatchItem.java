package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;

import java.util.Objects;
import java.util.concurrent.CompletableFuture;

/**
 * A single inference request queued for batch dispatch.
 *
 * <p>Extracted from {@link FlexlbBatchScheduler} to reduce coupling
 * with {@link WorkerBatcher}.
 *
 * <p>Carries direct {@link PrefillEndpoint} / {@link DecodeEndpoint} references
 * so downstream operations (commit, rollback, ack, cancel) avoid repeated
 * {@code EndpointRegistry} lookups by ip+port.
 *
 * <p>{@link #sortKey} is mutable — the {@link BatcherAlgorithm} computes it
 * inside {@link WorkerBatcher#offer(BatchItem)} via {@link BatcherAlgorithm#computeSortKey}.
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

    public BatchItem(BalanceContext ctx,
                     CompletableFuture<Response> future,
                     Response routeResponse,
                     ServerStatus prefill,
                     ServerStatus decode,
                     PrefillEndpoint prefillEp,
                     DecodeEndpoint decodeEp,
                     long sortKey,
                     long enqueuedAtMs) {
        this.ctx = ctx;
        this.future = future;
        this.routeResponse = routeResponse;
        this.prefill = prefill;
        this.decode = decode;
        this.prefillEp = prefillEp;
        this.decodeEp = decodeEp;
        this.sortKey = sortKey;
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

    /** Set by {@link WorkerBatcher#offer} after {@link BatcherAlgorithm#computeSortKey}. */
    public void setSortKey(long sortKey) { this.sortKey = sortKey; }

    /** @deprecated use {@link #sortKey()} instead; kept for SLO-budget references. */
    @Deprecated
    public long deadlineMs() { return sortKey; }

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
    public static long hitCacheOf(ServerStatus ss) {
        return ss != null && ss.getDebugInfo() != null
                ? ss.getDebugInfo().getHitCacheLen() : 0;
    }

    // -- Object --

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof BatchItem that)) return false;
        return sortKey == that.sortKey && enqueuedAtMs == that.enqueuedAtMs
                && Objects.equals(ctx, that.ctx) && Objects.equals(future, that.future)
                && Objects.equals(routeResponse, that.routeResponse)
                && Objects.equals(prefill, that.prefill)
                && Objects.equals(decode, that.decode)
                && Objects.equals(prefillEp, that.prefillEp)
                && Objects.equals(decodeEp, that.decodeEp);
    }

    @Override
    public int hashCode() {
        return Objects.hash(ctx, future, routeResponse, prefill, decode,
                prefillEp, decodeEp, sortKey, enqueuedAtMs);
    }

    @Override
    public String toString() {
        return "BatchItem{requestId=" + requestId() + ", seqLen=" + seqLen()
                + ", sortKey=" + sortKey + ", enqueuedAtMs=" + enqueuedAtMs + '}';
    }
}
