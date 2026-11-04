package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.WorkerEndpoint;

/** Scheduling owns its triggers and execution; Lifecycle owns requests and settlement. */
public interface RequestScheduler extends AutoCloseable {
    /** Start this implementation's scheduling triggers. */
    void start();

    /** Run available scheduling decisions. Implementations serialize concurrent invocations. */
    void schedule();

    void onRequest(RequestContext request);

    /** Invalidate exact membership; resource settlement remains with RequestLifecycle. */
    void onRequestChanged(RequestContext request);

    void onEndpointChanged(WorkerEndpoint endpoint);

    /** Number of requests waiting for placement capacity or preemption availability. */
    default int blockedRequestCount() {
        return 0;
    }

    /** Stop producing scheduling work and cancel owned timers; do not close shared facilities. */
    void stop();

    /** Wait for accepted scheduling work, using an absolute System.nanoTime deadline. */
    boolean awaitStopped(long deadlineNanos);

    /** Stop, drain scheduler-owned work and release scheduler-owned resources. */
    @Override
    void close();
}
