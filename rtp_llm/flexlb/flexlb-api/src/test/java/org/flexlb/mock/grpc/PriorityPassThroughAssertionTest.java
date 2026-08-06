package org.flexlb.mock.grpc;

import org.flexlb.dao.loadbalance.Response;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.mock.FlexLBMockTestBase;
import org.junit.jupiter.api.Test;

import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Priority pass-through assertion framework: proves the mock worker captures
 * the full {@code GenerateInputPB} of every dispatched request so that
 * scheduler-side config fields (e.g. the Stage 2 {@code priority} field) can
 * be asserted byte-accurately on the engine side.
 */
class PriorityPassThroughAssertionTest extends FlexLBMockTestBase {

    @Test
    void mockCapturesGenerateConfigForEngineSideAssertions() throws Exception {
        Response response = submitRequest(9101).get(5, TimeUnit.SECONDS);
        assertTrue(response.isSuccess());

        // The mock records the complete EnqueueBatch payload — dig out the
        // GenerateInputPB and verify config fields survive the wire intact.
        EngineRpcService.EnqueueBatchRequestPB recorded =
                mockPrefillWorker.getRpcService().getEnqueuedRequests().get(0);
        EngineRpcService.GenerateInputPB input = recorded.getDpSlots(0).getRequests(0).getInput();
        assertEquals(9101, input.getRequestId());
        assertEquals(8, input.getGenerateConfig().getMaxNewTokens());
        assertEquals(77, input.getGenerateConfig().getGroupTimeout().getValue());
    }

    @Test
    void generateConfigHasNoPriorityFieldBeforeStage2ProtoChange() {
        // Parity guard: the priority field is a Stage 2 protocol change. When
        // it lands, this assertion flips and the capture above becomes the
        // vehicle for asserting engine-side priority pass-through.
        assertNull(EngineRpcService.GenerateConfigPB.getDescriptor().findFieldByName("priority"),
                "GenerateConfigPB.priority not expected before the Stage 2 proto change");
    }
}
