package org.flexlb.balance.endpoint;

import org.flexlb.dao.master.WorkerStatus;

import java.util.OptionalLong;

/**
 * Endpoint for roles that only need status-based routing and no local
 * prefill/decode reservation state.
 */
public class SimpleWorkerEndpoint extends WorkerEndpoint {

    public SimpleWorkerEndpoint(WorkerStatus status) {
        super(status);
    }

    @Override
    public OptionalLong getLoadMetric() {
        WorkerStatus.EngineObservation status =
                getStatus().committedEngineObservation();
        return OptionalLong.of(
                status.runningTaskList() == null
                        ? 0L : status.runningTaskList().size());
    }
}
