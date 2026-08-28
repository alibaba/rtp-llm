package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.DeliveryItem;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.RoutingConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.util.Prioritized;

import java.util.Objects;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicLong;

/**
 * A single inference request queued for a priority scheduling decision.
 *
 * <p>Extracted from {@link RequestScheduler} to reduce coupling
 * with {@link WorkerBatcher}. A decision group may be delivered through a
 * batch RPC or returned as individual route decisions.
 *
 * <p>Carries direct {@link PrefillEndpoint} / {@link DecodeEndpoint} references
 * so downstream operations (commit, rollback, ack) avoid repeated
 * {@code EndpointRegistry} lookups by ip+port.
 *
 */
public final class BatchItem implements DeliveryItem, Prioritized {

    private static final AtomicLong ENQUEUE_SEQUENCE = new AtomicLong();

    private final BalanceContext ctx;
    private final CompletableFuture<Response> future;
    private final Response routeResponse;
    private final ServerStatus prefill;
    private final ServerStatus decode;
    private final PrefillEndpoint prefillEp;
    private final DecodeEndpoint decodeEp;
    private final DecodeEndpoint.ReservationHandle decodeReservation;
    /**
     * Expected-demand KV frozen at Decode reservation time
     * (decodeKvReservationTokens estimate); 0 when unknown (eviction
     * hand-off path or absent Decode reservation). Mirrored by the slot's
     * pRow resource ledger (plan 3.1 item 2).
     */
    private final long decodeExpectedKvTokens;
    private final long enqueuedAtMs;
    private final long enqueueSequence;
    private final long requestId;
    private final int priority;
    private final long expiresAtMs;
    private final long seqLen;
    private final long hitCache;
    private final long maxDecodeEngineRequests;
    private final long maxDecodeKvUsagePercent;
    private final int maxInflightDeliveriesPerPrefillWorker;

    public BatchItem(BalanceContext ctx,
                     CompletableFuture<Response> future,
                     Response routeResponse,
                     ServerStatus prefill,
                     ServerStatus decode,
                     PrefillEndpoint prefillEp,
                     DecodeEndpoint decodeEp,
                     DecodeEndpoint.ReservationHandle decodeReservation,
                     long decodeExpectedKvTokens,
                     long enqueuedAtMs) {
        this.ctx = Objects.requireNonNull(ctx, "ctx");
        this.future = future;
        this.routeResponse = routeResponse;
        this.prefill = prefill;
        this.decode = decode;
        this.prefillEp = prefillEp;
        this.decodeEp = decodeEp;
        this.decodeReservation = decodeReservation;
        this.decodeExpectedKvTokens = decodeExpectedKvTokens;
        this.enqueuedAtMs = enqueuedAtMs;
        this.enqueueSequence = ENQUEUE_SEQUENCE.incrementAndGet();
        Request request = ctx.getRequest();
        this.requestId = request == null ? 0L : request.getRequestId();
        this.priority = request == null && ctx.schedulingMetadata() == null
                ? 0 : ctx.getPriority();
        this.expiresAtMs = ctx.getRequestExpiresAtMs();
        this.seqLen = request == null ? 0L : request.getSeqLen();
        this.hitCache = hitCacheOf(prefill);
        FlexlbConfig schedulingConfig = Objects.requireNonNull(
                ctx.getConfig(), "request scheduling config");
        RoutingConfig.DecodeAvailabilityConfig decodeAvailability =
                schedulingConfig.getRouter().getRoles()
                .getDecode().getAvailability();
        Long configuredDecodeLimit = decodeAvailability.getMaxEngineRequests();
        this.maxDecodeEngineRequests = configuredDecodeLimit == null
                ? 0L : configuredDecodeLimit;
        this.maxDecodeKvUsagePercent =
                decodeAvailability.getMaxKvUsagePercent();
        Integer configuredDeliveryLimit = schedulingConfig.getDispatcher()
                .maxInflightDeliveriesPerPrefillWorker();
        this.maxInflightDeliveriesPerPrefillWorker =
                configuredDeliveryLimit == null ? 0 : configuredDeliveryLimit;
    }

    // -- accessors --

    public BalanceContext ctx() { return ctx; }
    public CompletableFuture<Response> future() { return future; }
    public Response routeResponse() { return routeResponse; }
    public ServerStatus prefill() { return prefill; }
    public ServerStatus decode() { return decode; }
    public PrefillEndpoint prefillEp() { return prefillEp; }
    public DecodeEndpoint decodeEp() { return decodeEp; }
    public DecodeEndpoint.ReservationHandle decodeReservation() {
        return decodeReservation;
    }
    public long decodeExpectedKvTokens() {
        return decodeExpectedKvTokens;
    }
    @Override
    public long enqueuedAtMs() { return enqueuedAtMs; }
    public long expiresAtMs() { return expiresAtMs; }
    public boolean requestExpired(long nowMs) {
        return expiresAtMs <= 0L || nowMs >= expiresAtMs;
    }
    public long maxDecodeEngineRequests() { return maxDecodeEngineRequests; }
    public long maxDecodeKvUsagePercent() { return maxDecodeKvUsagePercent; }
    public int maxInflightDeliveriesPerPrefillWorker() {
        return maxInflightDeliveriesPerPrefillWorker;
    }

    /**
     * Normalized request priority. Satisfies {@link Prioritized#priority()}
     * for the per-worker batcher
     * queue's {@code PriorityBlockingQueue}.
     */
    @Override
    public int priority() {
        return priority;
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

    @Override
    public long requestId() {
        return requestId;
    }

    /** Total sequence length of this request. */
    @Override
    public long seqLen() {
        return seqLen;
    }

    /** Cache-hit tokens on the assigned prefill endpoint. */
    @Override
    public long hitCache() {
        return hitCache;
    }

    /** Extract cache-hit length from a {@link ServerStatus} debug info. */
    private static long hitCacheOf(ServerStatus ss) {
        return ss != null && ss.getDebugInfo() != null
                ? ss.getDebugInfo().getHitCacheLen() : 0;
    }

}
