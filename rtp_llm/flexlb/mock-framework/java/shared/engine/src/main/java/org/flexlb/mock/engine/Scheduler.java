package org.flexlb.mock.engine;

import java.util.Objects;
import org.flexlb.mock.compute.ForwardBatch;
import org.flexlb.mock.compute.SimulatedCompute;

/** A scheduling domain owns its queues, reservations and post-Forward transitions. */
public interface Scheduler<T> {
    record Work<T>(T owner, ForwardBatch batch) {
        public Work { Objects.requireNonNull(owner); Objects.requireNonNull(batch); }
    }
    /** Return reserved work, or null until submission/capacity/output readiness wakes the loop. */
    Work<T> schedule();
    void completed(T owner, SimulatedCompute.Outcome outcome);
    void failed(T owner, RuntimeException error);
}
