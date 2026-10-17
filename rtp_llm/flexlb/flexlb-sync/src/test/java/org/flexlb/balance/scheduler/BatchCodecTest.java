package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

class BatchCodecTest {
    @Test void resultsFollowClaimOrderRatherThanWireOrder() {
        var claims = List.of(claim(1), claim(2));
        var response = EngineRpcService.EnqueueBatchResponsePB.newBuilder().setBatchId(17)
                .addSuccesses(EngineRpcService.EnqueueBatchSuccessPB.newBuilder().setRequestId(2))
                .addErrors(EngineRpcService.EnqueueBatchErrorPB.newBuilder().setRequestId(1)).build();
        var results = BatchCodec.validate(17, claims, response, null);
        assertEquals(List.of(DeliveryResult.Status.PREFILL_REJECTED, DeliveryResult.Status.DELIVERED),
                results.stream().map(DeliveryResult::status).toList());
        assertTrue(results.getFirst().cause().getMessage().contains("missing error_info"));
    }

    @ParameterizedTest @ValueSource(strings = {"missing", "unknown", "duplicate", "contradictory", "wrong batch"})
    void malformedMemberInvalidatesEveryAcknowledgement(String kind) {
        var response = EngineRpcService.EnqueueBatchResponsePB.newBuilder().setBatchId(17)
                .addSuccesses(EngineRpcService.EnqueueBatchSuccessPB.newBuilder().setRequestId(1));
        if (!kind.equals("missing")) { response.addSuccesses(EngineRpcService.EnqueueBatchSuccessPB.newBuilder().setRequestId(2)); }
        switch (kind) {
            case "unknown" -> response.addErrors(EngineRpcService.EnqueueBatchErrorPB.newBuilder().setRequestId(99));
            case "duplicate" -> response.addSuccesses(EngineRpcService.EnqueueBatchSuccessPB.newBuilder().setRequestId(1));
            case "contradictory" -> response.addErrors(EngineRpcService.EnqueueBatchErrorPB.newBuilder().setRequestId(1));
            case "wrong batch" -> response.setBatchId(18);
        }
        var results = BatchCodec.validate(17, List.of(claim(1), claim(2)), response.build(), null);
        assertEquals(2, results.size());
        assertTrue(results.stream().allMatch(value -> value.status() == DeliveryResult.Status.UNCERTAIN));
        assertSame(results.get(0).cause(), results.get(1).cause());
    }

    @Test void filteringCompletedPayloadCannotParseOrReadRequestAgain() throws Exception {
        var first = claim(1);
        var second = claim(2);
        var data1 = spy(first.owner().data());
        var data2 = spy(second.owner().data());
        doReturn(data1).when(first.owner()).data();
        doReturn(data2).when(second.owner()).data();
        doThrow(new AssertionError("must not parse during filtering")).when(data2).getGenerateInput();
        // A fully prepared payload is the only source for trimming at the send boundary.
        var prepared = EngineRpcService.EnqueueBatchRequestPB.newBuilder().setBatchId(17).setFetchAttachTimeoutMs(123)
                .addDpSlots(EngineRpcService.EnqueueBatchDpSlotPB.newBuilder().setDpRank(3)
                        .addRequests(EngineRpcService.EnqueueBatchExternalInputPB.newBuilder().setInput(
                                EngineRpcService.GenerateInputPB.newBuilder().setRequestId(1)))
                        .addRequests(EngineRpcService.EnqueueBatchExternalInputPB.newBuilder().setInput(
                                EngineRpcService.GenerateInputPB.newBuilder().setRequestId(2)))).build();
        var retained = BatchCodec.retain(prepared, List.of(second));
        assertEquals(2, prepared.getDpSlots(0).getRequestsCount());
        assertEquals(1, retained.getDpSlots(0).getRequestsCount());
        assertEquals(2, retained.getDpSlots(0).getRequests(0).getInput().getRequestId());
        assertEquals(123, retained.getFetchAttachTimeoutMs());
        verify(data1, never()).getGenerateInput();
        verify(data2, never()).getGenerateInput();
    }

    private static DeliveryClaim claim(long id) {
        var ctx = spy(new RequestContext(SchedulingTestConfig.batchConfig()));
        var route = mock(RequestRoute.class);
        when(route.ctx()).thenReturn(ctx);
        when(route.requestId()).thenReturn(id);
        return new DeliveryClaim(ctx, 1, route, DeliveryClaimKind.BATCH_ENQUEUE, 17);
    }
}
