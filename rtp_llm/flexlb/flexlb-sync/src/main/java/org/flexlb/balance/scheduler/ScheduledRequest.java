package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.endpoint.PrefillState;
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
    private final DecodeBinding decodeBinding;
    private final long enqueuedAtMs;
    private final long enqueueSequence;
    private final long expiresAtMs;
    private final long hitCache;
    private final int maxInflightBatchesPerPrefillWorker;
    private final boolean routeDelivery;
    /**
     * Publish-time NON_BATCH ownership. It has exactly one downstream owner:
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
        this(ctx, future, routeResponse, prefill, decode, prefillEp, decodeEp, decodeReservation,
                enqueuedAtMs, DecodeBinding.capture(ctx));
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
                     DecodeBinding frozenDecode) {
        this.ctx = Objects.requireNonNull(ctx, "ctx");
        this.future = future;
        this.routeResponse = routeResponse;
        this.prefill = prefill;
        this.prefillEp = prefillEp;
        this.decodeBinding = frozenDecode.bind(decode, decodeEp, decodeReservation);
        this.enqueuedAtMs = enqueuedAtMs;
        this.enqueueSequence = ENQUEUE_SEQUENCE.incrementAndGet();
        this.expiresAtMs = ctx.getRequestExpiresAtMs();
        this.hitCache = hitCacheOf(prefill);
        FlexlbConfig schedulingConfig = Objects.requireNonNull(
                ctx.getConfig(), "request scheduling config");
        this.routeDelivery = schedulingConfig.getDispatcher().getType() == DispatcherConfig.Type.NON_BATCH;
        this.maxInflightBatchesPerPrefillWorker = routeDelivery ? 0
                : schedulingConfig.getDispatcher().getMaxInflightPerPrefillWorker();
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

    public DecodeBinding decodeBinding() {
        return decodeBinding;
    }
    public long enqueuedAtMs() { return enqueuedAtMs; }
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
        return decodeBinding.priority();
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
        return decodeBinding.requestId();
    }

    /** Total sequence length of this request. */
    public long seqLen() {
        return decodeBinding.hardKvTokens();
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

    public enum DecodeMode {
        IMMEDIATE,
        WAIT_AT_DISPATCH,
        PREEMPT_AT_PLACEMENT;

        public static DecodeMode from(FlexlbConfig config) {
            if (config.isDirect()) { return IMMEDIATE; }
            return config.queueScheduler().getOrdering().preemptionPolicy().isPresent()
                    ? PREEMPT_AT_PLACEMENT : WAIT_AT_DISPATCH;
        }
    }

    /** Request values are frozen before selection; selected ownership is attached once known. */
    public record DecodeBinding(
            ServerStatus status,
            DecodeEndpoint endpoint,
            DecodeEndpoint.ReservationHandle reservation,
            long requestId,
            int priority,
            long hardKvTokens,
            long expectedKvTokens,
            DecodeEndpoint.AdmissionCapacity capacity,
            DecodeMode mode) {
        public DecodeBinding {
            Objects.requireNonNull(capacity, "capacity");
            Objects.requireNonNull(mode, "mode");
            if (reservation != null && reservation.requestId() != requestId) {
                throw new IllegalArgumentException("Decode reservation belongs to another request");
            }
        }

        public static DecodeBinding capture(BalanceContext context) {
            var request = Objects.requireNonNull(context.getRequest(), "request");
            var config = Objects.requireNonNull(context.getConfig(), "request config");
            var availability = config.getRouter().getRoles().getDecode().getAvailability();
            long promptTokens = Math.max(0L, request.getSeqLen());
            long outputTokens = Math.max(0L, request.getMaxNewTokens());
            long expectedTokens = promptTokens > Long.MAX_VALUE - outputTokens
                    ? Long.MAX_VALUE : promptTokens + outputTokens;
            return new DecodeBinding(null, null, null, request.getRequestId(), context.getPriority(),
                    promptTokens, expectedTokens,
                    new DecodeEndpoint.AdmissionCapacity(
                            availability.getMaxEngineRequests() == null ? 0L : availability.getMaxEngineRequests(),
                            availability.getMaxKvUsagePercent()), DecodeMode.from(config));
        }

        DecodeBinding bind(ServerStatus selectedStatus, DecodeEndpoint selectedEndpoint,
                           DecodeEndpoint.ReservationHandle selectedReservation) {
            return new DecodeBinding(selectedStatus, selectedEndpoint, selectedReservation,
                    requestId, priority, hardKvTokens, expectedKvTokens, capacity, mode);
        }

        /** Absence is a topology choice; a partial binding is still an error. */
        boolean isAbsent() {
            return status == null && endpoint == null && reservation == null;
        }
    }

}
