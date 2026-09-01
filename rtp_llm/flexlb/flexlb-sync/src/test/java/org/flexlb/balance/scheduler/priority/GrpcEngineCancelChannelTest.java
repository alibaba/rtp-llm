package org.flexlb.balance.scheduler.priority;

import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class GrpcEngineCancelChannelTest {

    @Test
    void cancelRoutesToPrefillControlOwner_notDecodeResourceOwner() throws Exception {
        EngineGrpcClient client = mock(EngineGrpcClient.class);
        when(client.cancelAsync(eq("10.0.0.1"), eq(8081),
                any(EngineRpcService.CancelRequestPB.class), eq(1000L)))
                .thenReturn(CompletableFuture.completedFuture(
                        EngineRpcService.CancelResponsePB.newBuilder()
                                .setStatus(EngineRpcService.CancelStatusPB.CANCEL_STATUS_ACCEPTED)
                                .build()));

        GrpcEngineCancelChannel channel = new GrpcEngineCancelChannel(client);
        EngineCancelChannel.CancelTarget target =
                new EngineCancelChannel.CancelTarget("10.0.0.1", 8081);

        EngineCancelChannel.CancelOutcome outcome = channel.cancel(target, 77L, 1000L)
                .get(1, TimeUnit.SECONDS);

        assertEquals(EngineCancelChannel.CancelAck.ACCEPTED, outcome.ack());
        ArgumentCaptor<EngineRpcService.CancelRequestPB> request =
                ArgumentCaptor.forClass(EngineRpcService.CancelRequestPB.class);
        verify(client).cancelAsync(eq("10.0.0.1"), eq(8081), request.capture(), eq(1000L));
        assertEquals(77L, request.getValue().getRequestId());
    }

    @Test
    void cancelMapsEngineNotFoundWithoutTreatingItAsAccepted() throws Exception {
        EngineGrpcClient client = mock(EngineGrpcClient.class);
        when(client.cancelAsync(eq("10.0.0.1"), eq(8081),
                any(EngineRpcService.CancelRequestPB.class), eq(1000L)))
                .thenReturn(CompletableFuture.completedFuture(
                        EngineRpcService.CancelResponsePB.newBuilder()
                                .setStatus(EngineRpcService.CancelStatusPB.CANCEL_STATUS_NOT_FOUND)
                                .build()));

        GrpcEngineCancelChannel channel = new GrpcEngineCancelChannel(client);
        EngineCancelChannel.CancelOutcome outcome = channel.cancel(
                        new EngineCancelChannel.CancelTarget("10.0.0.1", 8081), 78L, 1000L)
                .get(1, TimeUnit.SECONDS);

        assertEquals(EngineCancelChannel.CancelAck.NOT_FOUND, outcome.ack());
    }

    @Test
    void cancelMapsEngineTombstoneAsStrongFence() throws Exception {
        EngineGrpcClient client = mock(EngineGrpcClient.class);
        when(client.cancelAsync(eq("10.0.0.1"), eq(8081),
                any(EngineRpcService.CancelRequestPB.class), eq(1000L)))
                .thenReturn(CompletableFuture.completedFuture(
                        EngineRpcService.CancelResponsePB.newBuilder()
                                .setStatus(EngineRpcService.CancelStatusPB.CANCEL_STATUS_TOMBSTONED)
                                .build()));

        GrpcEngineCancelChannel channel = new GrpcEngineCancelChannel(client);
        EngineCancelChannel.CancelOutcome outcome = channel.cancel(
                        new EngineCancelChannel.CancelTarget("10.0.0.1", 8081), 79L, 1000L)
                .get(1, TimeUnit.SECONDS);

        assertEquals(EngineCancelChannel.CancelAck.TOMBSTONED, outcome.ack());
    }
}
