package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.prediction.DecodeCostFormula;
import org.flexlb.config.DispatcherConfig;
import org.flexlb.config.FlexlbConfig;
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
    private final RequestExecution execution;

    public ScheduledRequest(
            BalanceContext ctx,
            CompletableFuture<Response> future,
            Response routeResponse,
            ServerStatus prefill,
            ServerStatus decode,
            PrefillEndpoint prefillEp,
            DecodeEndpoint decodeEp,
            DecodeEndpoint.ReservationHandle decodeReservation,
            long enqueuedAtMs) {
        this(
                ctx,
                future,
                routeResponse,
                prefill,
                decode,
                prefillEp,
                decodeEp,
                decodeReservation,
                enqueuedAtMs,
                DecodeBinding.capture(ctx));
    }

    public ScheduledRequest(
            BalanceContext ctx,
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
        if (ctx.getWorkerEnqueueSequence() == 0L) {
            ctx.setFirstWorkerEnqueueTime(enqueuedAtMs);
            ctx.setWorkerEnqueueSequence(ENQUEUE_SEQUENCE.incrementAndGet());
        }
        FlexlbConfig schedulingConfig =
                Objects.requireNonNull(ctx.getConfig(), "request scheduling config");
        boolean routeDelivery =
                schedulingConfig.getDispatcher().getType() == DispatcherConfig.Type.NON_BATCH;
        this.execution =
                new RequestExecution(
                        routeResponse,
                        prefill,
                        prefillEp,
                        frozenDecode.bind(decode, decodeEp, decodeReservation),
                        ctx.getFirstWorkerEnqueueTime(),
                        ctx.getWorkerEnqueueSequence(),
                        ctx.getRequestExpiresAtMs(),
                        hitCacheOf(prefill),
                        routeDelivery
                                ? 0
                                : schedulingConfig
                                        .getDispatcher()
                                        .getMaxInflightPerPrefillWorker());
    }

    // -- accessors --

    public BalanceContext ctx() {
        return ctx;
    }

    /** Stable identity retained by endpoint state and request lifecycle tracking. */
    public RequestExecution execution() {
        return execution;
    }

    /** O(1) wire-size bound; grouping must not parse or copy GenerateInput payloads. */
    public long batchPayloadSizeUpperBound() {
        var input = ctx().getGenerateInputPb();
        // Input/external-input/DP-slot envelopes, priority and config length growth.
        // Charge a separate DP envelope per item, which also bounds mixed-DP batches.
        return (input == null ? 0L : input.size())
                + 64L
                + roleAddressSizeUpperBound(prefill())
                + roleAddressSizeUpperBound(decode());
    }

    private static long roleAddressSizeUpperBound(ServerStatus status) {
        if (status == null) {
            return 0L;
        }
        // Tags, lengths, enum, role name and two int32 ports fit in 64 bytes.
        // UTF-8 requires at most three bytes per Java UTF-16 code unit.
        String ip = status.getServerIp();
        return 64L + (ip == null ? 0L : 3L * ip.length());
    }

    public CompletableFuture<Response> future() {
        return future;
    }

    public Response routeResponse() {
        return execution.routeResponse();
    }

    public ServerStatus prefill() {
        return execution.prefill();
    }

    public ServerStatus decode() {
        return execution.decode();
    }

    public PrefillEndpoint prefillEp() {
        return execution.prefillEp();
    }

    public DecodeEndpoint decodeEp() {
        return execution.decodeEp();
    }

    /** Advisory only: delivery still acquires the exact Decode permit. */
    public boolean decodeReadyForDispatch() {
        return execution.decodeReadyForDispatch();
    }

    public DecodeEndpoint.ReservationHandle decodeReservation() {
        return execution.decodeReservation();
    }

    public DecodeBinding decodeBinding() {
        return execution.decodeBinding();
    }

    public long enqueuedAtMs() {
        return execution.enqueuedAtMs();
    }

    public long expiresAtMs() {
        return execution.expiresAtMs();
    }

    public boolean requestExpired(long nowMs) {
        return execution.requestExpired(nowMs);
    }

    public int maxInflightBatchesPerPrefillWorker() {
        return execution.maxInflightBatchesPerPrefillWorker();
    }

    /**
     * Normalized request priority. Satisfies {@link Prioritized#priority()}
     * for the per-worker ordered active index.
     */
    @Override
    public int priority() {
        return execution.priority();
    }

    /**
     * Unique monotonic enqueue sequence used by FIFO and as the same-priority
     * tie-break in {@link org.flexlb.util.PriorityOrdering#STRICT}. A re-offer
     * keeps the original item and therefore the original queue position.
     */
    @Override
    public long enqueueSeq() {
        return execution.enqueueSeq();
    }

    // -- derived accessors --

    public String requestId() {
        return execution.requestId();
    }

    /** Total sequence length of this request. */
    public long seqLen() {
        return execution.seqLen();
    }

    /** Cache-hit tokens on the assigned prefill endpoint. */
    public long hitCache() {
        return execution.hitCache();
    }

    /** Extract cache-hit length from a {@link ServerStatus} debug info. */
    private static long hitCacheOf(ServerStatus ss) {
        return ss != null && ss.getDebugInfo() != null ? ss.getDebugInfo().getHitCacheLen() : 0;
    }

    public enum DecodeMode {
        IMMEDIATE,
        WAIT_AT_PLACEMENT,
        PREEMPT_AT_PLACEMENT;

        public static DecodeMode from(FlexlbConfig config) {
            if (config.isDirect()) {
                return IMMEDIATE;
            }
            return config.queueScheduler().getOrdering().preemptionPolicy().isPresent()
                    ? PREEMPT_AT_PLACEMENT
                    : WAIT_AT_PLACEMENT;
        }
    }

    /**
     * Request values frozen before selection.
     * inputKvTokens is max(0, seqLen). inputAndMaxOutputKvTokens is
     * inputKvTokens + max(0, maxNewTokens), saturated at Long.MAX_VALUE.
     * Neither value is an estimate of the actual generated output length.
     */
    public record DecodeBinding(
            ServerStatus status,
            DecodeEndpoint endpoint,
            DecodeEndpoint.ReservationHandle reservation,
            String requestId,
            int priority,
            long inputKvTokens,
            long inputAndMaxOutputKvTokens,
            DecodeEndpoint.AdmissionCapacity capacity,
            DecodeMode mode,
            DecodeCostFormula costFormula) {
        public DecodeBinding {
            Objects.requireNonNull(capacity, "capacity");
            Objects.requireNonNull(mode, "mode");
            Objects.requireNonNull(costFormula, "costFormula");
            if (reservation != null && !Objects.equals(reservation.requestId(), requestId)) {
                throw new IllegalArgumentException("Decode reservation belongs to another request");
            }
        }

        public static DecodeBinding capture(BalanceContext context) {
            var request = Objects.requireNonNull(context.getRequest(), "request");
            var config = Objects.requireNonNull(context.getConfig(), "request config");
            var decode = config.getRouter().getRoles().getDecode();
            var availability = decode.getAvailability();
            long promptTokens = Math.max(0L, request.getSeqLen());
            long maxOutputTokens = Math.max(0L, request.getMaxNewTokens());
            long inputAndMaxOutputTokens =
                    promptTokens > Long.MAX_VALUE - maxOutputTokens
                            ? Long.MAX_VALUE
                            : promptTokens + maxOutputTokens;
            return new DecodeBinding(
                    null,
                    null,
                    null,
                    request.getRequestId(),
                    context.getPriority(),
                    promptTokens,
                    inputAndMaxOutputTokens,
                    new DecodeEndpoint.AdmissionCapacity(
                            availability.getMaxEngineRequests() == null
                                    ? 0L
                                    : availability.getMaxEngineRequests(),
                            availability.getMaxKvUsagePercent()),
                    DecodeMode.from(config),
                    decode.getCostEstimator().compiledFormula());
        }

        DecodeBinding bind(
                ServerStatus selectedStatus,
                DecodeEndpoint selectedEndpoint,
                DecodeEndpoint.ReservationHandle selectedReservation) {
            return new DecodeBinding(
                    selectedStatus,
                    selectedEndpoint,
                    selectedReservation,
                    requestId,
                    priority,
                    inputKvTokens,
                    inputAndMaxOutputKvTokens,
                    capacity,
                    mode,
                    costFormula);
        }

        /** Absence is a topology choice; a partial binding is still an error. */
        boolean isAbsent() {
            return status == null && endpoint == null && reservation == null;
        }
    }
}
