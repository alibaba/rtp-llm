package org.flexlb.balance.schedule;

import org.flexlb.balance.strategy.RoleSelection;
import org.flexlb.worker.endpoint.WorkerEndpoint;

import java.util.List;
import java.util.Set;
import java.util.function.Consumer;

/**
 * Owns resource waits and exposes short checks or result handling to the scheduler's bounded executor.
 * A completed resource result is transferred to the supplied handler for adoption or exact rollback.
 */
public interface RequestPreemption extends AutoCloseable {
    void setWakeup(Runnable wakeup);

    boolean start(
            RequestContext.SelectionAttempt selection,
            WorkerEndpoint endpoint,
            Consumer<RoleSelection> resultHandler);

    List<Work> readyWork();

    Set<RequestContext> waitingRequests();

    void requestCheck(WorkerEndpoint endpoint);

    void requestPeriodicCheck();

    void cancel(RequestContext request);

    void stopAdmission();

    boolean isIdle();

    @Override
    void close();

    record Work(RequestContext.SelectionAttempt selection, Runnable action) {}
}
