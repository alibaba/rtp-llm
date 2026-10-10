package org.flexlb.balance.eviction;

import io.grpc.Context;
import org.flexlb.balance.preemption.CancelTarget;
import org.flexlb.balance.scheduler.CancelReason;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.concurrent.CompletableFuture;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class GrpcEngineCancelChannelTest {
    private final EngineGrpcClient client = mock(EngineGrpcClient.class);
    private final GrpcEngineCancelChannel channel = new GrpcEngineCancelChannel(client);
    private final CancelTarget target = new CancelTarget("127.0.0.1", 9090);

    @ParameterizedTest
    @EnumSource(CancelReason.class)
    void sendsExplicitReasonAndPreservesWeakAcknowledgement(CancelReason reason) {
        when(client.getWorkerStatusAsync(eq("127.0.0.1"), eq(9090), any(), anyLong()))
                .thenReturn(CompletableFuture.completedFuture(EngineRpcService.WorkerStatusPB.newBuilder()
                        .setSupportsRequestCleanup(true).build()));
        when(client.cancelAsync(eq("127.0.0.1"), eq(9090), any(), anyLong()))
                .thenReturn(CompletableFuture.completedFuture(EngineRpcService.CancelResponsePB.newBuilder()
                        .setStatus(EngineRpcService.CancelStatusPB.CANCEL_STATUS_ACCEPTED).build()));

        assertEquals(EngineCancelChannel.CancelAck.ACCEPTED, channel.cancel(target, 71, reason, 1000).join());
        verify(client).cancelAsync(eq("127.0.0.1"), eq(9090), eq(EngineRpcService.CancelRequestPB.newBuilder()
                .setRequestId(71).setReason(EngineRpcService.RequestCancelReasonPB.valueOf("REQUEST_CANCEL_REASON_" + reason.name()))
                .build()), eq(1000L));
        if (reason == CancelReason.PRIORITY_PREEMPTED) {
            verify(client, never()).getWorkerStatusAsync(any(), anyInt(), any(), anyLong());
        }
    }

    @ParameterizedTest
    @EnumSource(value = CancelReason.class, names = "PRIORITY_PREEMPTED", mode = EnumSource.Mode.EXCLUDE)
    void ordinaryCancelDoesNotReachAnOldEngine(CancelReason reason) {
        when(client.getWorkerStatusAsync(eq("127.0.0.1"), eq(9090), any(), anyLong()))
                .thenReturn(CompletableFuture.completedFuture(EngineRpcService.WorkerStatusPB.getDefaultInstance()));

        assertEquals(EngineCancelChannel.CancelAck.UNSUPPORTED, channel.cancel(target, 72, reason, 1000).join());
        verify(client, never()).cancelAsync(any(), anyInt(), any(), anyLong());
    }

    @Test
    void callerCancellationDoesNotCancelCleanupAfterAsynchronousCapabilityCheck() {
        var status = new CompletableFuture<EngineRpcService.WorkerStatusPB>();
        when(client.getWorkerStatusAsync(eq("127.0.0.1"), eq(9090), any(), anyLong())).thenReturn(status);
        when(client.cancelAsync(eq("127.0.0.1"), eq(9090), any(), anyLong())).thenAnswer(call -> {
            assertFalse(Context.current().isCancelled());
            return CompletableFuture.completedFuture(EngineRpcService.CancelResponsePB.newBuilder()
                    .setStatus(EngineRpcService.CancelStatusPB.CANCEL_STATUS_TOMBSTONED).build());
        });
        try (var caller = Context.current().withCancellation()) {
            Context previous = caller.attach();
            CompletableFuture<EngineCancelChannel.CancelAck> result;
            try {
                result = channel.cancel(target, 73, CancelReason.CLIENT_CANCELLED, 1000);
                caller.cancel(new IllegalStateException("client disconnected"));
                status.complete(EngineRpcService.WorkerStatusPB.newBuilder().setSupportsRequestCleanup(true).build());
            } finally {
                caller.detach(previous);
            }
            assertEquals(EngineCancelChannel.CancelAck.REQUEST_FENCED, result.join());
        }
    }

    @Test
    void failedCapabilityCheckDoesNotSendAnUnverifiedCancel() {
        when(client.getWorkerStatusAsync(eq("127.0.0.1"), eq(9090), any(), anyLong()))
                .thenReturn(CompletableFuture.failedFuture(new IllegalStateException("status unavailable")));
        assertEquals(EngineCancelChannel.CancelAck.FAILED, channel.cancel(target, 74, CancelReason.SHUTDOWN, 1000).join());
        verify(client, never()).cancelAsync(any(), anyInt(), any(), anyLong());
    }
}
