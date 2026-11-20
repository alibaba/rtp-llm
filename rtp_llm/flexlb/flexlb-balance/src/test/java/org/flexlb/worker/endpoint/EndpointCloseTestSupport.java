package org.flexlb.worker.endpoint;

import org.flexlb.config.service.RoleType;

/**
 * Allows cleanup scheduling faults to be injected into the actual endpoint gate.
 */
final class EndpointCloseTestSupport extends WorkerEndpoint {
    private final Runnable continuation;

    EndpointCloseTestSupport(Runnable continuation) {
        super(EndpointTestSupport.workerStatus(RoleType.PREFILL, "127.0.0.1", 8080, 8081));
        this.continuation = continuation;
    }

    @Override
    protected void closeEndpoint() {
        continuation.run();
    }
}
