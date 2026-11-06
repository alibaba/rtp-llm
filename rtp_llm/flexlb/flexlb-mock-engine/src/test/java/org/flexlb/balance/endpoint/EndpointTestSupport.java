package org.flexlb.balance.endpoint;

import org.flexlb.balance.scheduler.PlacementAvailability;
import org.flexlb.balance.scheduler.RequestRepository;
import org.flexlb.dao.master.WorkerStatus;

public final class EndpointTestSupport {
    private EndpointTestSupport() {}

    public static java.util.concurrent.locks.ReentrantLock stateLock(WorkerEndpoint endpoint) {
        return endpoint.stateLock();
    }

    public static DecodeEndpoint decode(WorkerStatus status, RequestRepository repository) {
        return new DecodeEndpoint(status, repository, new PlacementAvailability());
    }
}
