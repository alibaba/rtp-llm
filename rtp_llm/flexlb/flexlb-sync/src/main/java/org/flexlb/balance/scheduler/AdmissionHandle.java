package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.dao.loadbalance.Response;

import java.util.Map;
import java.util.Objects;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;

/** One exact placement operation, shared by the caller and request state. */
public final class AdmissionHandle implements AutoCloseable {
    private final RequestContext owner;
    private final RequestRoute withdrawing;
    private volatile boolean resolved;

    AdmissionHandle(RequestContext owner, RequestRoute withdrawing) {
        this.owner = Objects.requireNonNull(owner);
        this.withdrawing = withdrawing;
    }

    /** Claim an unstarted request for preemption; finish or restore the returned handle. */
    public static AdmissionHandle tryBeginPreemption(
            RequestLifecycle lifecycle,
            DecodeEndpoint endpoint,
            DecodeResources.ReservationHandle victim,
            int incomingPriority) {
        var requests = lifecycle.requests;
        if (!lifecycle.registerAdmission()) {
            return null;
        }
        AdmissionHandle token = null;
        RequestContext context = requests.findActive(victim.requestId());
        try {
            if (context == null
                    || !context.data().getConfig().isQueue()
                    || context.data().getLifecycle() != lifecycle) {
                return null;
            }
            synchronized (context) {
                RequestRoute item = context.snapshot().route();
                // Context serializes the exact queue claim with delivery permit transfer.
                if (item == null
                        || item.priority() >= incomingPriority
                        || item.decodeEp() != endpoint
                        || !Objects.equals(item.decodeReservation(), victim)
                        || !requests.isCurrent(context)
                        || context.data().getFuture().isDone()
                        || item.requestExpired(System.currentTimeMillis())
                        || !endpoint.ownership(victim).withdrawable()) {
                    return null;
                }
                token = context.beginPlacement(item);
                return token;
            }
        } finally {
            if (token == null) lifecycle.finishAdmission();
        }
    }

    public RequestContext owner() {
        return owner;
    }

    public RequestRoute withdrawingRoute() {
        return withdrawing;
    }

    void recordDiagnostics(Map<String, Object> value) {
        if (value == null) {
            return;
        }
        synchronized (owner()) {
            var current = owner().snapshot();
            if (!resolved
                    && current.placement() != null
                    && current.placement().operation() == this
                    && current.cancellation() == null) {
                owner().data().setSchedulingDiagnostics(Map.copyOf(value));
            }
        }
    }

    public void finish() {
        complete(
                withdrawing == null
                        ? ContextResult.PublicationResult.PUBLISHED
                        : ContextResult.PublicationResult.RESTORED);
    }

    public void complete(ContextResult.PublicationResult result) {
        checkArgument(
                result == ContextResult.PublicationResult.PUBLISHED
                        || result == ContextResult.PublicationResult.RESTORED
                        || result == ContextResult.PublicationResult.WITHDRAWN,
                "invalid placement completion");
        resolve(result, null);
    }

    public void terminate(Response failure) {
        Objects.requireNonNull(failure);
        checkArgument(!failure.isSuccess(), "admission termination requires a failure");
        resolve(ContextResult.PublicationResult.FAILED, failure);
    }

    private void resolve(ContextResult.PublicationResult result, Response failure) {
        synchronized (this) {
            if (resolved) return;
            resolved = true;
        }
        owner.data().getLifecycle().completePlacement(this, result, failure);
    }

    @Override
    public void close() {
        checkState(resolved, "admission must be explicitly finished");
    }
}
