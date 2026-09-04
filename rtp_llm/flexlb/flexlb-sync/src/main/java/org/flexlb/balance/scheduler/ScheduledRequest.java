package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.config.DispatcherConfig;
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
import java.util.concurrent.atomic.AtomicReference;

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
public final class ScheduledRequest implements Prioritized {

    private static final AtomicLong ENQUEUE_SEQUENCE = new AtomicLong();

    private final BalanceContext ctx;
    private final CompletableFuture<Response> future;
    private final Response routeResponse;
    private final ServerStatus prefill;
    private final PrefillEndpoint prefillEp;
    private final DecodeBinding decodeBinding;
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
    private final boolean routeDelivery;
    /**
     * Publish-time NON_BATCH credit. It has exactly one downstream owner:
     * this ACTIVE item until RouteDeliveryStrategy takes it, then the route
     * transaction. ACTIVE terminal paths take and close the same capability.
     */
    private final AtomicReference<PrefillState.RouteReservation>
            publishedRouteReservation = new AtomicReference<>();

    public ScheduledRequest(BalanceContext ctx,
                     CompletableFuture<Response> future,
                     Response routeResponse,
                     ServerStatus prefill,
                     ServerStatus decode,
                     PrefillEndpoint prefillEp,
                     DecodeEndpoint decodeEp,
                     DecodeEndpoint.ReservationHandle decodeReservation,
                     long enqueuedAtMs) {
        this.ctx = Objects.requireNonNull(ctx, "ctx");
        this.future = future;
        this.routeResponse = routeResponse;
        this.prefill = prefill;
        this.prefillEp = prefillEp;
        this.decodeBinding = new DecodeBinding(
                decode, decodeEp, decodeReservation);
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
        this.routeDelivery = schedulingConfig.getDispatcher().getType()
                == DispatcherConfig.Type.NON_BATCH;
    }

    // -- accessors --

    public BalanceContext ctx() { return ctx; }
    public CompletableFuture<Response> future() { return future; }
    public Response routeResponse() { return routeResponse; }
    public ServerStatus prefill() { return prefill; }
    public ServerStatus decode() { return decodeBinding.status(); }
    public PrefillEndpoint prefillEp() { return prefillEp; }
    public DecodeEndpoint decodeEp() { return decodeBinding.endpoint(); }
    public DecodeEndpoint.ReservationHandle decodeReservation() {
        return decodeBinding.reservation();
    }

    DecodeBinding decodeBinding() {
        return decodeBinding;
    }
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

    /** Whether ACTIVE publication must atomically own one request credit. */
    public boolean requiresRouteReservation() {
        return routeDelivery;
    }

    /** Bind the sole publish-time route credit before ACTIVE becomes visible. */
    boolean bindPublishedRouteReservation(
            PrefillState.RouteReservation reservation) {
        return publishedRouteReservation.compareAndSet(
                null, Objects.requireNonNull(reservation, "reservation"));
    }

    /** Transfer the sole uncommitted route credit to delivery or cleanup. */
    PrefillState.RouteReservation takePublishedRouteReservation() {
        return publishedRouteReservation.getAndSet(null);
    }

    /** Observe without transferring; optimistic preparation owns no credit. */
    PrefillState.RouteReservation publishedRouteReservation() {
        return publishedRouteReservation.get();
    }

    /** Transfer only the reservation validated during optimistic preparation. */
    boolean takePublishedRouteReservation(
            PrefillState.RouteReservation expected) {
        return publishedRouteReservation.compareAndSet(
                Objects.requireNonNull(expected, "expected"), null);
    }

    /** Restore ownership after a pre-commit operation made no state change. */
    boolean restorePublishedRouteReservation(
            PrefillState.RouteReservation reservation) {
        return publishedRouteReservation.compareAndSet(
                null, Objects.requireNonNull(reservation, "reservation"));
    }

    /**
     * Normalized request priority. Satisfies {@link Prioritized#priority()}
     * for the per-worker ordered active index.
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

    public long requestId() {
        return requestId;
    }

    /** Total sequence length of this request. */
    public long seqLen() {
        return seqLen;
    }

    /** Cache-hit tokens on the assigned prefill endpoint. */
    public long hitCache() {
        return hitCache;
    }

    /** Extract cache-hit length from a {@link ServerStatus} debug info. */
    private static long hitCacheOf(ServerStatus ss) {
        return ss != null && ss.getDebugInfo() != null
                ? ss.getDebugInfo().getHitCacheLen() : 0;
    }

    record DecodeBinding(
            ServerStatus status,
            DecodeEndpoint endpoint,
            DecodeEndpoint.ReservationHandle reservation) {
    }

}
