package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;

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

    /**
     * Absolute deadline (epoch ms) for end-to-end timeout propagation.
     * Computed as request_time_ms + generate_timeout from the Schedule request.
     * 0 means not set (fallback to per-stage deadline).
     */
    private final long absoluteDeadlineMs;

    /** Mutable sort key set by the batcher algorithm at offer time. */
    private volatile long sortKey;

    /**
     * Canonical constructor. The {@code sortKey} field defaults to 0 (via
     * {@code volatile long} zero-initialization) and is set later by
     * {@link BatcherAlgorithm#computeSortKey} via {@link #setSortKey}.
     */
    public BatchItem(BalanceContext ctx,
                     CompletableFuture<Response> future,
                     Response routeResponse,
                     ServerStatus prefill,
                     ServerStatus decode,
                     PrefillEndpoint prefillEp,
                     DecodeEndpoint decodeEp,
                     long enqueuedAtMs,
                     long absoluteDeadlineMs) {
        this.ctx = ctx;
        this.future = future;
        this.routeResponse = routeResponse;
        this.prefill = prefill;
        this.decode = decode;
        this.prefillEp = prefillEp;
        this.decodeEp = decodeEp;
        this.enqueuedAtMs = enqueuedAtMs;
        this.absoluteDeadlineMs = absoluteDeadlineMs;
    }

    /** Backward-compatible constructor (absoluteDeadlineMs defaults to 0 = not set). */
    public BatchItem(BalanceContext ctx,
                     CompletableFuture<Response> future,
                     Response routeResponse,
                     ServerStatus prefill,
                     ServerStatus decode,
                     PrefillEndpoint prefillEp,
                     DecodeEndpoint decodeEp,
                     long enqueuedAtMs) {
        this(ctx, future, routeResponse, prefill, decode, prefillEp, decodeEp,
                enqueuedAtMs, 0L);
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

    /** Absolute deadline (epoch ms); 0 means not set (fallback to per-stage deadline). */
    public long absoluteDeadlineMs() { return absoluteDeadlineMs; }

    /** Priority queue sort key. */
    public long sortKey() { return sortKey; }

    /** Set by {@link WorkerBatcher#offer} after {@link BatcherAlgorithm#computeSortKey}. */
    public void setSortKey(long sortKey) { this.sortKey = sortKey; }

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
