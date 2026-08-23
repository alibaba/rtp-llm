package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.config.BatchDispatcherConfig;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.config.NonBatchDispatcherConfig;
import org.flexlb.config.RoutingConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.util.Prioritized;

import java.util.Objects;
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
    private final long maxDecodeEngineRequests;
    private final long maxDecodeKvUsagePercent;
    private final int maxPrefillRequestsPerWorker;
    private final int maxInflightBatchesPerPrefillWorker;

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
        FlexlbConfig schedulingConfig = Objects.requireNonNull(
                ctx.getConfig(), "request scheduling config");
        this.deliveryMode = DeliveryMode.from(schedulingConfig);
        RoutingConfig.DecodeAvailabilityConfig decodeAvailability =
                schedulingConfig.getRouter().getRoles()
                .getDecode().getAvailability();
        Long configuredDecodeLimit = decodeAvailability.getMaxEngineRequests();
        this.maxDecodeEngineRequests = configuredDecodeLimit == null
                ? 0L : configuredDecodeLimit;
        this.maxDecodeKvUsagePercent =
                decodeAvailability.getMaxKvUsagePercent();
        Integer configuredPrefillLimit = schedulingConfig.getDispatcher()
                instanceof NonBatchDispatcherConfig nonBatch
                ? nonBatch.getMaxInflightRequestsPerPrefillWorker()
                : null;
        this.maxPrefillRequestsPerWorker = configuredPrefillLimit == null
                ? 0 : configuredPrefillLimit;
        Integer configuredBatchLimit = schedulingConfig.getDispatcher()
                instanceof BatchDispatcherConfig batch
                ? batch.getMaxInflightBatchesPerPrefillWorker()
                : null;
        this.maxInflightBatchesPerPrefillWorker = configuredBatchLimit == null
                ? 0 : configuredBatchLimit;
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
    long maxDecodeEngineRequests() { return maxDecodeEngineRequests; }
    long maxDecodeKvUsagePercent() { return maxDecodeKvUsagePercent; }
    int maxPrefillRequestsPerWorker() { return maxPrefillRequestsPerWorker; }
    int maxInflightBatchesPerPrefillWorker() {
        return maxInflightBatchesPerPrefillWorker;
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
