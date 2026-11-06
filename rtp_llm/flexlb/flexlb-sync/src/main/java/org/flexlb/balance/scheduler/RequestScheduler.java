package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.WorkerEndpoint;

/**
 * Defines replaceable scheduling decisions and triggers.
 */
public interface RequestScheduler extends AutoCloseable {
    /**
     * Start this implementation's scheduling triggers.
     */
    void start();

    /**
     * Run available scheduling decisions. Implementations serialize concurrent invocations.
     */
    void schedule();

    /**
     * Schedule a newly registered request or one whose previous route has been fully released.
     * The implementation decides when to run it and preserves its original ordering on retry.
     */
    void onRequest(RequestContext request);

    void onEndpointChanged(WorkerEndpoint endpoint);

    /**
     * Select the failure policy. After successful withdrawal and cleanup, retry uses onRequest.
     */
    default boolean shouldRetryRouteFailure(ContextResult.RouteFailure failure) {
        return false;
    }

    /**
     * Number of requests waiting for placement capacity or preemption availability.
     */
    default int blockedRequestCount() {
        return 0;
    }

    /**
     * Stop producing scheduling work and cancel owned timers; do not close shared facilities.
     */
    void stop();

    /**
     * Wait for accepted scheduling work, using an absolute System.nanoTime deadline.
     */
    boolean awaitStopped(long deadlineNanos);

    /**
     * Stop, drain scheduler-owned work and release scheduler-owned resources.
     */
    @Override
    void close();
}
