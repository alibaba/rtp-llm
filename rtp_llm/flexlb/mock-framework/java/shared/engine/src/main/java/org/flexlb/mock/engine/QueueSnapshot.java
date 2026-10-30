package org.flexlb.mock.engine;

/** Scheduler-defined request counts. Loading and transfer-ready requests are not compute waiters. */
public record QueueSnapshot(int waiting, int running, int loading, int ready) {
    public QueueSnapshot {
        if (waiting < 0 || running < 0 || loading < 0 || ready < 0)
            throw new IllegalArgumentException("Negative request count");
    }
}
