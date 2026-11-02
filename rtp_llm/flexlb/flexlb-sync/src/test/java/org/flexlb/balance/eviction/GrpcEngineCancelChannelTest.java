package org.flexlb.balance.eviction;

import org.flexlb.balance.preemption.CancelTarget;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.client.EngineGrpcClient;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Verifies that cancellation reaches Decode and carries the original Prefill address.
 */
class GrpcEngineCancelChannelTest {
    @Test
    void cancellationUsesDecodeAndCarriesPrefillForAdmissionRace() {
        EngineGrpcClient client = mock(EngineGrpcClient.class);
        when(client.cancelAsync(eq("10.0.0.2"), eq(9091), any(), eq(100L)))
                .thenReturn(
                        CompletableFuture.completedFuture(
                                EngineRpcService.CancelResponsePB.newBuilder()
                                        .setStatus(
                                                EngineRpcService.CancelStatusPB
                                                        .CANCEL_STATUS_ACCEPTED)
                                        .build()));
        GrpcEngineCancelChannel channel = new GrpcEngineCancelChannel(client);

        assertEquals(
                EngineCancelChannel.CancelAck.ACCEPTED,
                channel.cancel(new CancelTarget("10.0.0.2", 9091, "10.0.0.1:9090"), "42", 100L)
                        .join());

        ArgumentCaptor<EngineRpcService.CancelRequestPB> request =
                ArgumentCaptor.forClass(EngineRpcService.CancelRequestPB.class);
        verify(client).cancelAsync(eq("10.0.0.2"), eq(9091), request.capture(), eq(100L));
        assertEquals(42L, request.getValue().getRequestId());
        assertEquals("10.0.0.1:9090", request.getValue().getPrefillAddress());
    }
}
