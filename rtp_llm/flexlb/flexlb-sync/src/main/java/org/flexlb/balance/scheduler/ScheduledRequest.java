package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
import org.flexlb.balance.scheduler.decode.DecodeDecideInfo;
import org.flexlb.balance.scheduler.decode.DecodeSelection;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
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
    private final DecodeDecideInfo decodeDecideInfo;
    private final long enqueuedAtMs;
    /**
     * Monotonic entry time for the current endpoint queue visit, not the first route attempt.
     */
    private long subQueueEnteredAtNanos;
    private final long enqueueSequence;
    private final long expiresAtMs;
    private final long hitCache;
    private final int maxInflightBatchesPerPrefillWorker;
    private final boolean routeDelivery;
    /**
     * Publish-time NON_BATCH ownership. It has exactly one downstream owner:
     * this ACTIVE item until NonBatchDeliveryStrategy takes it, then the route
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
        this(ctx, future, routeResponse, prefill, decode, prefillEp, decodeEp, decodeReservation, enqueuedAtMs,
                DecodeDecideInfo.capture(ctx));
    }

    public ScheduledRequest(BalanceContext ctx,
                            CompletableFuture<Response> future,
                            Response routeResponse,
                            ServerStatus prefill,
                            ServerStatus decode,
                            PrefillEndpoint prefillEp,
                            DecodeEndpoint decodeEp,
                            DecodeEndpoint.ReservationHandle decodeReservation,
                            long enqueuedAtMs,
                            DecodeDecideInfo frozenDecode) {
        this.ctx = ctx;
        this.future = future;
        this.routeResponse = routeResponse;
        this.prefill = prefill;
        this.prefillEp = prefillEp;
        this.decodeDecideInfo = frozenDecode.withSelection(DecodeSelection.of(decode, decodeEp, decodeReservation));
        if (ctx.getWorkerEnqueueSequence() == 0L) {
            ctx.setFirstWorkerEnqueueTime(enqueuedAtMs);
            ctx.setWorkerEnqueueSequence(ENQUEUE_SEQUENCE.incrementAndGet());
        }
        this.enqueuedAtMs = ctx.getFirstWorkerEnqueueTime();
        this.enqueueSequence = ctx.getWorkerEnqueueSequence();
        this.expiresAtMs = ctx.getRequestExpiresAtMs();
        this.hitCache = hitCacheOf(prefill);
        FlexlbConfig schedulingConfig = ctx.getConfig();
        this.routeDelivery = schedulingConfig.getDispatcher().getType() == DispatcherConfig.Type.NON_BATCH;
        this.maxInflightBatchesPerPrefillWorker = routeDelivery ? 0
                : schedulingConfig.getDispatcher().getMaxInflightPerPrefillWorker();
    }

    // -- accessors --

    public BalanceContext ctx() { return ctx; }
    public CompletableFuture<Response> future() { return future; }
    public Response routeResponse() { return routeResponse; }
    public ServerStatus prefill() { return prefill; }
    public ServerStatus decode() { return decodeDecideInfo.status(); }
    public PrefillEndpoint prefillEp() { return prefillEp; }
    public DecodeEndpoint decodeEp() { return decodeDecideInfo.endpoint(); }
    public DecodeEndpoint.ReservationHandle decodeReservation() {
        return decodeDecideInfo.reservation();
    }

    public DecodeDecideInfo decodeDecideInfo() {
        return decodeDecideInfo;
    }
    public long enqueuedAtMs() { return enqueuedAtMs; }

    void markSubQueueEntry() {
        subQueueEnteredAtNanos = System.nanoTime();
    }

    long subQueueEnteredAtNanos() {
        return subQueueEnteredAtNanos;
    }

    public long finishSubQueueVisitMs() {
        long enteredAt = subQueueEnteredAtNanos;
        if (enteredAt == 0L) {
            return -1L;
        }
        subQueueEnteredAtNanos = 0L;
        return java.util.concurrent.TimeUnit.NANOSECONDS.toMillis(
                Math.max(0L, System.nanoTime() - enteredAt));
    }
    public long expiresAtMs() { return expiresAtMs; }
    public boolean requestExpired(long nowMs) {
        return expiresAtMs <= 0L || nowMs >= expiresAtMs;
    }
    public int maxInflightBatchesPerPrefillWorker() {
        return maxInflightBatchesPerPrefillWorker;
    }

    /** Whether ACTIVE publication must atomically own one route reservation. */
    public boolean requiresRouteReservation() {
        return routeDelivery;
    }

    /** Bind the sole publish-time route reservation before ACTIVE becomes visible. */
    boolean bindPublishedRouteReservation(
            PrefillState.RouteReservation reservation) {
        return publishedRouteReservation.compareAndSet(
                null, Objects.requireNonNull(reservation, "reservation"));
    }

    /** Transfer the sole uncommitted route reservation to delivery or cleanup. */
    PrefillState.RouteReservation takePublishedRouteReservation() {
        return publishedRouteReservation.getAndSet(null);
    }

    /** Observe without transferring; optimistic preparation does not transfer ownership. */
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
        return decodeDecideInfo.requirement().priority();
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
        return decodeDecideInfo.requirement().requestId();
    }

    /** Total sequence length of this request. */
    public long seqLen() {
        return decodeDecideInfo.requirement().inputKvTokens();
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
}
