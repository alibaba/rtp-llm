package org.flexlb.balance.schedule;

import org.flexlb.balance.strategy.RoleSelection;
import org.flexlb.worker.endpoint.WorkerEndpoint;

import java.util.List;
import java.util.function.BooleanSupplier;
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

    /**
     * Returns work for installed waits. The scheduler decides when to run each task.
     */
    List<Work> pendingWork();

    default void onCapacityChanged(WorkerEndpoint endpoint) {}

    default boolean isChanged(long snapshotVersion) {
        return snapshotVersion() != snapshotVersion;
    }

    default long snapshotVersion() {
        return 0L;
    }

    void cancel(RequestContext request);

    void stopAdmission();

    boolean isIdle();

    @Override
    void close();

    /**
     * Exposes the work for a resource wait and whether its result is ready to be delivered.
     */
    record Work(
            RequestContext.SelectionAttempt selection,
            Runnable action,
            BooleanSupplier resultReady) {
        public Work(RequestContext.SelectionAttempt selection, Runnable action) {
            this(selection, action, () -> true);
        }

        public boolean isResultReady() {
            return resultReady.getAsBoolean();
        }
    }
}
