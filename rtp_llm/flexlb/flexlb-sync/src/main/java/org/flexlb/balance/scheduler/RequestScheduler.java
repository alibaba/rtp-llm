package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.WorkerEndpoint;

/** Scheduling decisions; request admission and settlement belong to RequestLifecycle. */
public interface RequestScheduler extends AutoCloseable {
    /** Run available decisions; return the next check time in System.nanoTime units.
     * Long.MAX_VALUE means wait for a request or endpoint notification.
     * The caller serializes invocations of this method.
     */
    long schedule(long nowNanos);

    void onEndpointChanged(WorkerEndpoint endpoint);

    /** A request whose lifecycle and cancellation callback are already registered. */
    void onRequest(RequestContext request);

    /** Lifecycle notification; queue removal must not settle or release the request. */
    void onRequestChanged(RequestContext request);

    /** Stop scheduling and release scheduler-owned workers. RequestLifecycle settles requests. */
    @Override
    void close();

}
