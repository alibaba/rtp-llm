package org.flexlb.mock;

import org.flexlb.engine.grpc.EngineRpcService;

/**
 * Mock decode worker — a gRPC server that simulates a decode engine
 * without loading any model.
 *
 * <p>Role-specific behavior:
 * <ul>
 *   <li>Configurable {@code enqueueDelayMs} to simulate decode generation time</li>
 *   <li>Returns healthy WorkerStatus with available KV cache</li>
 * </ul>
 *
 * <p>In PD-separated architecture, the decode worker receives requests only
 * after the prefill worker completes.  The mock decode worker's gRPC port
 * is used for status sync and cancel propagation.
 */
public class MockDecodeWorker extends MockWorker {

    /**
     * Create a decode mock worker with the given behavior.
     */
    public MockDecodeWorker(MockWorkerBehavior behavior) {
        super(behavior.toBuilder()
                .roleType(EngineRpcService.RoleTypePB.ROLE_TYPE_DECODE)
                .build());
    }

    /**
     * Create a decode mock worker with default behavior.
     */
    public MockDecodeWorker() {
        this(MockWorkerBehavior.builder().build());
    }
}
