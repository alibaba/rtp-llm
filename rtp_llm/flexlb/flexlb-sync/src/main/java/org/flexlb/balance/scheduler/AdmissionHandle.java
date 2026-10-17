package org.flexlb.balance.scheduler;

import org.flexlb.dao.loadbalance.Response;

import java.util.Map;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;

/** One caller's placement scope. Context retains only its immutable token. */
public final class AdmissionHandle implements AutoCloseable {
    final ContextResult.PlacementToken token;
    private final AbstractRequestScheduler scheduler;
    private final AtomicBoolean resolved = new AtomicBoolean();

    AdmissionHandle(ContextResult.PlacementToken token, AbstractRequestScheduler scheduler) {
        this.token = Objects.requireNonNull(token);
        this.scheduler = Objects.requireNonNull(scheduler);
    }

    public RequestContext owner() { return token.owner(); }
    public RequestRoute withdrawingRoute() { return token.withdrawing(); }

    void recordDiagnostics(Map<String, Object> value) {
        if (value == null) { return; }
        synchronized (owner()) {
            var current = owner().snapshot();
            if (!resolved.get() && current.placement() != null
                    && current.placement().operation() == token && current.cancellation() == null) {
                owner().data().setSchedulingDiagnostics(Map.copyOf(value));
            }
        }
    }

    public void finish() {
        if (resolved.compareAndSet(false, true)) { scheduler.completePlacement(this, null); }
    }

    public void terminate(Response failure) {
        Objects.requireNonNull(failure);
        checkArgument(!failure.isSuccess(), "admission termination requires a failure");
        if (resolved.compareAndSet(false, true)) { scheduler.completePlacement(this, failure); }
    }

    @Override public void close() { checkState(resolved.get(), "admission must be explicitly finished"); }
}
