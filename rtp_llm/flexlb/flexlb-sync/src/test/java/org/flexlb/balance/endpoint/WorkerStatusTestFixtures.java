package org.flexlb.balance.endpoint;

import org.flexlb.dao.master.WorkerStatus;

/** Explicit successful-poll setup for endpoint/scheduling tests without an RPC driver. */
public final class WorkerStatusTestFixtures {
    private WorkerStatusTestFixtures() { }

    public static WorkerStatus readyForScheduling(WorkerStatus worker) {
        worker.lock.lock();
        try {
            worker.recordSuccessfulPoll(true, null);
            return worker;
        } finally { worker.lock.unlock(); }
    }
}
