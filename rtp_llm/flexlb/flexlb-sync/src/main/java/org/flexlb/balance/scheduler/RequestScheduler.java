package org.flexlb.balance.scheduler;

import org.flexlb.dao.loadbalance.Response;

import java.util.concurrent.CompletableFuture;

/** Request scheduling and cancellation. ApplicationLifecycle owns service shutdown. */
public interface RequestScheduler {
    /** Assembly access to shared state transitions, deadlines and settlement. */
    RequestLifecycle lifecycle();

    void start();

    /** Wait for pending scheduling work until the absolute System.nanoTime deadline. */
    boolean awaitSchedulingCompletion(long deadlineNanos);

    /** Stop scheduling and release scheduler-owned workers. RequestLifecycle settles requests. */
    void close();

    /**
     * Accept a request and return its scheduling result, not an enqueue acknowledgement.
     * The returned future retains caller cancellation semantics. Response completion
     * does not imply that the request's resource obligations have ended.
     */
    CompletableFuture<Response> submit(RequestContext request);

    /**
     * Install caller cancellation after the owner and result future are bound.
     * QUEUE overrides this to invoke the callback before enqueueing. The default preserves
     * DIRECT's immediate submission and invokes the callback after submit returns.
     * Rejected registrations do not invoke it.
     */
    default CompletableFuture<Response> submit(RequestContext request, Runnable onRegistered) {
        java.util.Objects.requireNonNull(onRegistered, "onRegistered");
        CompletableFuture<Response> result = submit(request);
        if (request != null
                && request.data().getScheduler() != null
                && request.data().getFuture() == result) {
            onRegistered.run();
        }
        return result;
    }

    /**
     * Request cancellation from the owning scheduler. Acceptance does not imply
     * immediate resource release. Returns null when no matching request is known.
     */
    default RequestState cancel(long requestId, long batchId, CancelReason reason) {
        return lifecycle().cancel(requestId, batchId, reason);
    }
}
