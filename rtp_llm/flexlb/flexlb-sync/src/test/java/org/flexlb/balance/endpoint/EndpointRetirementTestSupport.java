package org.flexlb.balance.endpoint;

import org.flexlb.dao.route.RoleType;

/** Allows retirement scheduling faults to be injected into the actual endpoint gate. */
final class EndpointRetirementTestSupport extends WorkerEndpoint {
    private final Runnable continuation;

    EndpointRetirementTestSupport(Runnable continuation) {
        super(EndpointTestSupport.workerStatus(RoleType.PREFILL, "127.0.0.1", 8080, 8081));
        this.continuation = continuation;
    }

    @Override
    void continueRetirementAfterHandoff() {
        continuation.run();
    }
}
