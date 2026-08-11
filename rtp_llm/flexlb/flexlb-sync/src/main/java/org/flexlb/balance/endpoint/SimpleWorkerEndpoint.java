package org.flexlb.balance.endpoint;

import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.route.RoleType;

/**
 * Endpoint for roles that only need status-based routing and no local
 * prefill/decode reservation state.
 */
public class SimpleWorkerEndpoint extends WorkerEndpoint {

    public SimpleWorkerEndpoint(WorkerStatus status) {
        this(new EndpointId(status.getRole() == null ? RoleType.VIT : status.getRole(),
                status.getIpPort(), 0), status);
    }

    public SimpleWorkerEndpoint(EndpointId endpointId, WorkerStatus status) {
        super(endpointId, status);
    }

    @Override
    public long getLoadMetric() {
        return status.getRunningTaskList() == null ? 0 : status.getRunningTaskList().size();
    }
}
