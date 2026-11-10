package org.flexlb.balance.scheduler;

import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.doReturn;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Verifies immutable payload construction and exact-member response ordering.
 */
class BatchEnqueueHelperTest {
    @Test
    void resultsFollowClaimOrderRatherThanWireOrder() {
        var claims = List.of(claim(1), claim(2));
        var response =
                EngineRpcService.EnqueueBatchResponsePB.newBuilder()
                        .setBatchId(17)
                        .addSuccesses(
                                EngineRpcService.EnqueueBatchSuccessPB.newBuilder().setRequestId(2))
                        .addErrors(
                                EngineRpcService.EnqueueBatchErrorPB.newBuilder().setRequestId(1))
                        .build();
        var results = EngineDelivery.validate(17, claims, response, null);
        assertEquals(
                List.of(Delivery.Result.Status.PREFILL_REJECTED, Delivery.Result.Status.DELIVERED),
                results.stream().map(Delivery.Result::status).toList());
        assertTrue(results.getFirst().cause().getMessage().contains("missing error_info"));
    }

    @ParameterizedTest
    @ValueSource(strings = {"missing", "unknown", "duplicate", "contradictory", "wrong batch"})
    void malformedMemberInvalidatesEveryAcknowledgement(String kind) {
        var response =
                EngineRpcService.EnqueueBatchResponsePB.newBuilder()
                        .setBatchId(17)
                        .addSuccesses(
                                EngineRpcService.EnqueueBatchSuccessPB.newBuilder()
                                        .setRequestId(1));
        if (!kind.equals("missing")) {
            response.addSuccesses(
                    EngineRpcService.EnqueueBatchSuccessPB.newBuilder().setRequestId(2));
        }
        switch (kind) {
            case "unknown" ->
                    response.addErrors(
                            EngineRpcService.EnqueueBatchErrorPB.newBuilder().setRequestId(99));
            case "duplicate" ->
                    response.addSuccesses(
                            EngineRpcService.EnqueueBatchSuccessPB.newBuilder().setRequestId(1));
            case "contradictory" ->
                    response.addErrors(
                            EngineRpcService.EnqueueBatchErrorPB.newBuilder().setRequestId(1));
            case "wrong batch" -> response.setBatchId(18);
        }
        var results =
                EngineDelivery.validate(17, List.of(claim(1), claim(2)), response.build(), null);
        assertEquals(2, results.size());
        assertTrue(
                results.stream()
                        .allMatch(value -> value.status() == Delivery.Result.Status.UNCERTAIN));
        assertSame(results.get(0).cause(), results.get(1).cause());
    }

    @Test
    void filteringCompletedPayloadCannotParseOrReadRequestAgain() throws Exception {
        var first = claim(1);
        var second = claim(2);
        var data1 = spy(first.owner().data());
        var data2 = spy(second.owner().data());
        var firstOwner = first.owner();
        var secondOwner = second.owner();
        doReturn(data1).when(firstOwner).data();
        doReturn(data2).when(secondOwner).data();
        doThrow(new AssertionError("must not parse during filtering"))
                .when(data2)
                .getGenerateInput();
        // A fully prepared payload is the only source for trimming at the send boundary.
        var prepared =
                EngineRpcService.EnqueueBatchRequestPB.newBuilder()
                        .setBatchId(17)
                        .setFetchAttachTimeoutMs(123)
                        .addDpSlots(
                                EngineRpcService.EnqueueBatchDpSlotPB.newBuilder()
                                        .setDpRank(3)
                                        .addRequests(
                                                EngineRpcService.EnqueueBatchExternalInputPB
                                                        .newBuilder()
                                                        .setInput(
                                                                EngineRpcService.GenerateInputPB
                                                                        .newBuilder()
                                                                        .setRequestId(1)))
                                        .addRequests(
                                                EngineRpcService.EnqueueBatchExternalInputPB
                                                        .newBuilder()
                                                        .setInput(
                                                                EngineRpcService.GenerateInputPB
                                                                        .newBuilder()
                                                                        .setRequestId(2))))
                        .build();
        var retained = EngineDelivery.retain(prepared, List.of(second));
        assertEquals(2, prepared.getDpSlots(0).getRequestsCount());
        assertEquals(1, retained.getDpSlots(0).getRequestsCount());
        assertEquals(2L, retained.getDpSlots(0).getRequests(0).getInput().getRequestId());
        assertEquals(123, retained.getFetchAttachTimeoutMs());
        verify(data1, never()).getGenerateInput();
        verify(data2, never()).getGenerateInput();
    }

    @ParameterizedTest
    @ValueSource(booleans = {false, true})
    void payloadBoundCoversFrozenRoleAddressesWithoutParsing(boolean separateDecode)
            throws Exception {
        var context = spy(new RequestContext(SchedulingTestConfig.batchConfig()));
        context.data()
                .setRequest(org.flexlb.dao.loadbalance.Request.builder().requestId("91").build());
        var data = spy(context.data());
        doReturn(data).when(context).data();
        data.setGenerateInputPb(
                EngineRpcService.GenerateInputPB.newBuilder()
                        .setRequestId(91L)
                        .addTokenIds(1000)
                        .build()
                        .toByteString());
        var prefill = mock(org.flexlb.balance.endpoint.PrefillEndpoint.class);
        String address = "worker-" + "节点".repeat(80);
        when(prefill.getIp()).thenReturn(address);
        var status = new org.flexlb.dao.loadbalance.ServerStatus();
        status.setRole(org.flexlb.dao.route.RoleType.PREFILL);
        status.setServerIp(address);
        status.setHttpPort(65535);
        status.setGrpcPort(65535);
        status.setSuccess(true);
        status.setRequestId(context.data().getRequestId());
        var selections = new java.util.ArrayList<RoleSelection>();
        selections.add(SelectionTestSupport.metadata(prefill, status, 0L));
        if (separateDecode) {
            var decode = mock(org.flexlb.balance.endpoint.DecodeEndpoint.class);
            when(decode.getIp()).thenReturn(address);
            var decoded = org.flexlb.dao.loadbalance.ServerStatus.copyOf(status);
            decoded.setRole(org.flexlb.dao.route.RoleType.DECODE);
            selections.add(SelectionTestSupport.metadata(decode, decoded, 0L));
        }
        var member =
                DeliveryTestSupport.member(
                        new RouteResult(context, selections), Delivery.SendMode.BATCH_ENQUEUE, 17L);
        long bound = 16L + EngineDelivery.payloadSizeUpperBound(member.route());
        verify(data, never()).getGenerateInput();
        assertTrue(bound >= EngineDelivery.prepare(17L, List.of(member)).getSerializedSize());
    }

    private static Delivery.Member claim(long id) {
        var ctx = spy(new RequestContext(SchedulingTestConfig.batchConfig()));
        ctx.data()
                .setRequest(
                        org.flexlb.dao.loadbalance.Request.builder()
                                .requestId(String.valueOf(id))
                                .build());
        var route = RequestProtocolTestSupport.mockSelectedRoute(ctx);
        return DeliveryTestSupport.member(route, Delivery.SendMode.BATCH_ENQUEUE, 17);
    }
}
