package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.util.Prioritized;

import java.util.Objects;

/**
 * Frozen scheduling values and endpoint ownership retained during generation.
 *
 * <p>One instance identifies one selected route. It has no reference to the
 * scheduling input, its context, or its completion future.
 */
public final class RequestExecution implements Prioritized {

    private final Response routeResponse;
    private final ServerStatus prefill;
    private final PrefillEndpoint prefillEp;
    private final ScheduledRequest.DecodeBinding decodeBinding;
    private final long enqueuedAtMs;
    private final long enqueueSequence;
    private final long expiresAtMs;
    private final long hitCache;
    private final int maxInflightBatchesPerPrefillWorker;

    RequestExecution(
            Response routeResponse,
            ServerStatus prefill,
            PrefillEndpoint prefillEp,
            ScheduledRequest.DecodeBinding decodeBinding,
            long enqueuedAtMs,
            long enqueueSequence,
            long expiresAtMs,
            long hitCache,
            int maxInflightBatchesPerPrefillWorker) {
        this.routeResponse = routeResponse;
        this.prefill = prefill;
        this.prefillEp = prefillEp;
        this.decodeBinding = Objects.requireNonNull(decodeBinding, "decodeBinding");
        this.enqueuedAtMs = enqueuedAtMs;
        this.enqueueSequence = enqueueSequence;
        this.expiresAtMs = expiresAtMs;
        this.hitCache = hitCache;
        this.maxInflightBatchesPerPrefillWorker = maxInflightBatchesPerPrefillWorker;
    }

    public String requestId() {
        return decodeBinding.requestId();
    }

    @Override
    public int priority() {
        return decodeBinding.priority();
    }

    public long seqLen() {
        return decodeBinding.inputKvTokens();
    }

    public long hitCache() {
        return hitCache;
    }

    public long enqueuedAtMs() {
        return enqueuedAtMs;
    }

    @Override
    public long enqueueSeq() {
        return enqueueSequence;
    }

    public long expiresAtMs() {
        return expiresAtMs;
    }

    public boolean requestExpired(long nowMs) {
        return expiresAtMs <= 0L || nowMs >= expiresAtMs;
    }

    public Response routeResponse() {
        return routeResponse;
    }

    public ServerStatus prefill() {
        return prefill;
    }

    public PrefillEndpoint prefillEp() {
        return prefillEp;
    }

    public ServerStatus decode() {
        return decodeBinding.status();
    }

    public DecodeEndpoint decodeEp() {
        return decodeBinding.endpoint();
    }

    public DecodeEndpoint.ReservationHandle decodeReservation() {
        return decodeBinding.reservation();
    }

    public ScheduledRequest.DecodeBinding decodeBinding() {
        return decodeBinding;
    }

    public int maxInflightBatchesPerPrefillWorker() {
        return maxInflightBatchesPerPrefillWorker;
    }

    /** Advisory only: delivery still acquires the exact Decode permit. */
    public boolean decodeReadyForDispatch() {
        return decodeBinding.isAbsent()
                || decodeBinding.endpoint() == null
                || decodeBinding
                        .endpoint()
                        .shouldRetryDispatch(requestId(), decodeBinding.capacity());
    }
}
