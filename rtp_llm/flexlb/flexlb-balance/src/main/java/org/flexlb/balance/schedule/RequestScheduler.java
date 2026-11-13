package org.flexlb.balance.schedule;

import org.flexlb.worker.endpoint.WorkerEndpoint;

/**
 * Defines replaceable scheduling decisions and triggers.
 */
public interface RequestScheduler extends AutoCloseable {
    /**
     * Start this implementation's scheduling triggers.
     */
    void start();

    /**
     * Schedule a newly registered request or one whose previous route has been fully released.
     * The implementation decides when to run it and preserves its original ordering on retry.
     */
    void scheduleRequest(RequestContext request);

    void onEndpointChanged(WorkerEndpoint endpoint);

    /**
     * Number of queued requests not yet handed to delivery; immediate schedulers report zero.
     */
    default int queuedRequestCount() {
        return 0;
    }

    /**
     * Number of registered queued requests without an active decision or pending preemption result.
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
