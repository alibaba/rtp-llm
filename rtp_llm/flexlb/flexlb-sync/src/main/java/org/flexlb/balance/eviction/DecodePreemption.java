package org.flexlb.balance.eviction;

import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.WorkerEndpoint;
import org.flexlb.balance.scheduler.RequestContext;

import java.util.Objects;
import java.util.concurrent.CompletableFuture;

import static com.google.common.base.Preconditions.checkArgument;

/**
 * Scheduler-facing capacity reclamation service, replaceable independently of the queue.
 * Implementations own withdrawal/cancel fencing and settlement. A successful result transfers
 * one exact reservation to the caller, which must adopt it or release it if placement is obsolete.
 * Completing or cancelling the returned future is not evidence that a victim released capacity.
 */
public interface DecodePreemption extends AutoCloseable {
    /** Null means no attempt; otherwise completion reports the settled acquisition outcome. */
    CompletableFuture<PreemptionResult> tryReclaim(RequestContext context, WorkerEndpoint endpoint);

    /** Stop new acquisition and prevent in-flight attempts from committing after shutdown. */
    void stopAdmission();

    /** Wait for accepted reclamation and rollback, using an absolute System.nanoTime deadline. */
    boolean awaitIdle(long deadlineNanos);

    /** Stop new reclamation and drain owned work before releasing implementation resources. */
    @Override
    void close();

    public record PreemptionResult(
            DecodeResources.ReservationHandle reservation, boolean controlFailure, String detail) {
        public boolean committed() {
            return reservation != null;
        }

        public PreemptionResult {
            checkArgument(
                    reservation == null || !controlFailure,
                    "committed preemption cannot be a control failure");
            Objects.requireNonNull(detail, "detail");
        }
    }
}
