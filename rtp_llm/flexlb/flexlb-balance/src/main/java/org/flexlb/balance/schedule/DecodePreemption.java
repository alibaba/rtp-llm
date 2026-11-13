package org.flexlb.balance.schedule;

import org.flexlb.worker.endpoint.WorkerEndpoint;

import java.util.concurrent.CompletableFuture;

/**
 * Scheduler-facing capacity reclamation service, replaceable independently of the queue.
 * Implementations own withdrawal/cancel fencing and settlement. A successful result transfers
 * one exact reservation to the caller, which must adopt it or release it if placement is obsolete.
 * Completing or cancelling the returned future is not evidence that a request selected for preemption released capacity.
 */
public interface DecodePreemption extends AutoCloseable {
    /**
     * Null means no attempt; otherwise completion reports the settled acquisition outcome.
     */
    CompletableFuture<RequestResult> tryPreempt(RequestContext context, WorkerEndpoint endpoint);

    /**
     * Stop new acquisition and prevent in-flight attempts from committing after shutdown.
     */
    void stopAdmission();

    /**
     * Wait for accepted reclamation and rollback, using an absolute System.nanoTime deadline.
     */
    boolean awaitIdle(long deadlineNanos);

    /**
     * Stop new reclamation and drain owned work before releasing implementation resources.
     */
    @Override
    void close();
}
