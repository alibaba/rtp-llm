package org.flexlb.balance.endpoint;

import org.flexlb.dao.master.WorkerStatus;

/**
 * Endpoint for roles that only need status-based routing and no local
 * prefill/decode reservation state.
 */
public class SimpleWorkerEndpoint extends WorkerEndpoint {

    public SimpleWorkerEndpoint(WorkerStatus status) {
        super(status);
    }

    /**
     * Number of running tasks reported by the engine status.
     */
    public long runningTaskCount() {
        return status.getRunningTaskList() == null ? 0 : status.getRunningTaskList().size();
    }
}
