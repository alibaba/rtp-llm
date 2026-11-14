package org.flexlb.mock;

import org.flexlb.engine.grpc.EngineRpcService;

/**
 * Runs a mock Decode gRPC worker with configurable response timing, failures and status
 * observations.
 */
public class MockDecodeWorker extends MockWorker {

    /**
     * Create a decode mock worker with the given behavior.
     */
    public MockDecodeWorker(MockWorkerBehavior behavior) {
        super(behavior.toBuilder().roleType(EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE).build());
    }

    /**
     * Create a decode mock worker with default behavior.
     */
    public MockDecodeWorker() {
        this(MockWorkerBehavior.builder().build());
    }
}
