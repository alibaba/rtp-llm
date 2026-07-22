package org.flexlb.mock;

import org.flexlb.engine.grpc.EngineRpcService;

/**
 * Mock prefill worker — a gRPC server that simulates a prefill engine
 * without loading any model.
 *
 * <p>Role-specific behavior:
 * <ul>
 *   <li>Configurable {@code enqueueDelayMs} to simulate prefill compute time</li>
 *   <li>Can fail EnqueueBatch to test dispatch failure handling</li>
 * </ul>
 */
public class MockPrefillWorker extends MockWorker {

    /**
     * Create a prefill mock worker with the given behavior.
     */
    public MockPrefillWorker(MockWorkerBehavior behavior) {
        super(behavior.toBuilder()
                .roleType(EngineRpcService.RoleTypePB.ROLE_TYPE_PREFILL)
                .build());
    }

    /**
     * Create a prefill mock worker with default behavior.
     */
    public MockPrefillWorker() {
        this(MockWorkerBehavior.builder().build());
    }
}
