package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.dao.loadbalance.Response;

import java.util.Map;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;

/** One caller's placement admission. Context retains only its immutable token. */
public final class AdmissionHandle implements AutoCloseable {
    final ContextResult.PlacementToken token;
    private final RequestLifecycle scheduler;
    private final AtomicBoolean resolved = new AtomicBoolean();
    private final Object placementFailureLock = new Object();
    private Runnable placementFailureAction;
    private boolean placementFailureNotified;

    AdmissionHandle(ContextResult.PlacementToken token, RequestLifecycle scheduler) {
        this.token = Objects.requireNonNull(token);
        this.scheduler = Objects.requireNonNull(scheduler);
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
        boolean transferred = false;
        ContextResult.PlacementToken token = null;
        RequestContext context = requests.findActive(victim.requestId());
        try {
            if (context == null
                    || !context.data().getConfig().isQueue()
                    || context.data().getScheduler() != lifecycle) {
                return null;
            }
            synchronized (context) {
                var state = context.snapshot();
                RequestRoute item = state.route();
                // Context serializes the exact queue claim with delivery permit transfer.
                if (item == null
                        || item.priority() >= incomingPriority
                        || item.decodeEp() != endpoint
                        || !Objects.equals(item.decodeReservation(), victim)
                        || !requests.isCurrent(context)
                        || state.phase() != ContextResult.Phase.OPEN
                        || state.cancellation() != null
                        || state.delivery() != null
                        || state.placement() != null
                        || state.preemption() != null
                        || state.engine().decodeAccepted()
                        || context.data().getFuture().isDone()
                        || item.requestExpired(System.currentTimeMillis())
                        || !endpoint.ownership(victim).withdrawable()) {
                    return null;
                }
                var claim = context.beginPlacement(item);
                if (!(claim
                        instanceof ContextResult.Granted<ContextResult.PlacementToken> granted)) {
                    return null;
                }
                token = granted.value();
                AdmissionHandle handle = new AdmissionHandle(token, lifecycle);
                transferred = true;
                return handle;
            }
        } finally {
            if (!transferred) {
                if (token != null) {
                    context.completePlacement(
                            token,
                            ContextResult.PublicationResult.FAILED,
                            System.currentTimeMillis());
                }
                lifecycle.finishAdmission();
            }
        }
    }

    public RequestContext owner() {
        return token.owner();
    }

    public RequestRoute withdrawingRoute() {
        return token.withdrawing();
    }

    void recordDiagnostics(Map<String, Object> value) {
        if (value == null) {
            return;
        }
        synchronized (owner()) {
            var current = owner().snapshot();
            if (!resolved.get()
                    && current.placement() != null
                    && current.placement().operation() == token
                    && current.cancellation() == null) {
                owner().data().setSchedulingDiagnostics(Map.copyOf(value));
            }
        }
    }

    public void finish() {
        complete(
                token.withdrawing() == null
                        ? ContextResult.PublicationResult.PUBLISHED
                        : ContextResult.PublicationResult.RESTORED);
    }

    public void complete(ContextResult.PublicationResult result) {
        checkArgument(
                result == ContextResult.PublicationResult.PUBLISHED
                        || result == ContextResult.PublicationResult.RESTORED
                        || result == ContextResult.PublicationResult.WITHDRAWN,
                "invalid placement completion");
        if (resolved.compareAndSet(false, true)) {
            scheduler.completePlacement(this, result, null);
        }
    }

    public void terminate(Response failure) {
        Objects.requireNonNull(failure);
        checkArgument(!failure.isSuccess(), "admission termination requires a failure");
        if (resolved.compareAndSet(false, true)) {
            scheduler.completePlacement(this, ContextResult.PublicationResult.FAILED, failure);
        }
    }

    /**
     * Registers the resource action which the scheduler must run if this placement fails.
     * The action is deliberately owned by the resource admission component; this handle only
     * provides the single failure notification point.
     */
    public void onPlacementFailure(Runnable action) {
        Objects.requireNonNull(action, "action");
        synchronized (placementFailureLock) {
            checkState(!resolved.get(), "admission already resolved");
            checkState(!placementFailureNotified, "placement failure notification already started");
            checkState(
                    placementFailureAction == null, "placement failure action already registered");
            placementFailureAction = action;
        }
    }

    /** Runs the registered resource action at most once. */
    void notifyPlacementFailure() {
        Runnable action;
        synchronized (placementFailureLock) {
            if (placementFailureNotified) {
                return;
            }
            placementFailureNotified = true;
            action = placementFailureAction;
            placementFailureAction = null;
        }
        if (action != null) {
            action.run();
        }
    }

    @Override
    public void close() {
        checkState(resolved.get(), "admission must be explicitly finished");
    }
}
