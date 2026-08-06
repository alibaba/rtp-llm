package org.flexlb.mock.grpc;

import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.mock.FlexLBMockTestBase;
import org.junit.jupiter.api.Test;

import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
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
        BalanceContext ctx = createBalanceContext(9101);
        ctx.getRequest().setPriority(40);
        Response response = scheduler.submit(ctx).get(5, TimeUnit.SECONDS);
        assertTrue(response.isSuccess());

        // The mock records the complete EnqueueBatch payload — dig out the
        // GenerateInputPB and verify config fields survive the wire intact.
        EngineRpcService.EnqueueBatchRequestPB recorded =
                mockPrefillWorker.getRpcService().getEnqueuedRequests().get(0);
        EngineRpcService.GenerateInputPB input = recorded.getDpSlots(0).getRequests(0).getInput();
        assertEquals(9101, input.getRequestId());
        assertEquals(8, input.getGenerateConfig().getMaxNewTokens());
        assertEquals(77, input.getGenerateConfig().getGroupTimeout().getValue());
        // Stage 2b: normalized priority injected by PrefillEndpoint.buildInput
        // survives the wire to the engine side.
        assertEquals(40, input.getGenerateConfig().getPriority());
    }

    @Test
    void generateConfigHasPriorityFieldAfterStage2bProtoChange() {
        // Flipped Stage 2 parity guard: the Stage 2b proto change added
        // GenerateConfigPB.priority, so the field must now be present and the
        // capture above asserts engine-side priority pass-through.
        assertNotNull(EngineRpcService.GenerateConfigPB.getDescriptor().findFieldByName("priority"),
                "GenerateConfigPB.priority expected after the Stage 2b proto change");
    }
}
