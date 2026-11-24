package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.route.RoleType;
import org.flexlb.util.Prioritized;

import java.util.Objects;

/**
 * Frozen scheduling values and endpoint ownership retained during generation.
 *
 * <p>One instance identifies one selected route. It has no reference to the
 * scheduling input, its context, or its completion future.
 */
public final class RequestExecution implements Prioritized {

    // Set under the global queue lock; a fast completion can precede publication bookkeeping.
    boolean prefillEnded;
    private final Response routeResponse;
    private final ServerStatus prefill;
    private final PrefillEndpoint prefillEp;
    // The slot owns the only write, at the point the delivery permit transfers
    // to the endpoint. Tentative selections live on the delivery transaction.
    private volatile ScheduledRequest.DecodeBinding decodeBinding;
    private final long enqueuedAtMs;
    private final long enqueueSequence;
    private final long expiresAtMs;
    private final long hitCache;
    private final int maxInflightBatchesPerPrefillWorker;

    RequestExecution(Response routeResponse,
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

    public long requestId() { return decodeBinding.requestId(); }

    @Override
    public int priority() { return decodeBinding.priority(); }

    public long seqLen() { return decodeBinding.hardKvTokens(); }
    public long hitCache() { return hitCache; }
    public long enqueuedAtMs() { return enqueuedAtMs; }

    @Override
    public long enqueueSeq() { return enqueueSequence; }

    public long expiresAtMs() { return expiresAtMs; }
    public boolean requestExpired(long nowMs) {
        return expiresAtMs <= 0L || nowMs >= expiresAtMs;
    }

    public Response routeResponse() {
        var binding = decodeBinding;
        if (binding.decodeSelector() == null || binding.status() == null) { return routeResponse; }
        Response result = Response.copyOf(routeResponse);
        addSelectedDecode(result, binding.status());
        return result;
    }

    Response deliveryResponse(boolean enqueuedByMaster) {
        Response result = Response.buildSuccessResponse(routeResponse, enqueuedByMaster);
        var binding = decodeBinding;
        if (binding.decodeSelector() != null) {
            addSelectedDecode(result, Objects.requireNonNull(binding.status(), "Decode must be bound before delivery"));
        }
        return result;
    }

    private static void addSelectedDecode(Response result, ServerStatus status) {
        var roles = result.getServerStatus();
        // Keep the public selection order (a preselected VIT may precede the PD pair).
        int index = 0;
        while (index < roles.size() && !roles.get(index).getRole().supportsPrefill()) { index++; }
        if (index < roles.size() && roles.get(index).getRole() == RoleType.PDFUSION) { index++; }
        roles.add(index, ServerStatus.copyOf(status));
    }

    /** Called once under the slot lock after the exact prepared permit transfers. */
    void commitDecodeBinding(ScheduledRequest.DecodeBinding selected) {
        decodeBinding = selected;
    }

    public ServerStatus prefill() { return prefill; }
    public PrefillEndpoint prefillEp() { return prefillEp; }
    public ServerStatus decode() { return decodeBinding.status(); }
    public DecodeEndpoint decodeEp() { return decodeBinding.endpoint(); }
    public DecodeEndpoint.ReservationHandle decodeReservation() {
        return decodeBinding.reservation();
    }
    public ScheduledRequest.DecodeBinding decodeBinding() { return decodeBinding; }
    public int maxInflightBatchesPerPrefillWorker() {
        return maxInflightBatchesPerPrefillWorker;
    }

    /** Advisory only: delivery still acquires the exact Decode permit. */
    public boolean decodeReadyForDispatch() {
        return decodeBinding.isAbsent() || decodeBinding.endpoint() == null
                || decodeBinding.endpoint().shouldRetryDispatch(requestId(), decodeBinding.capacity());
    }
}
