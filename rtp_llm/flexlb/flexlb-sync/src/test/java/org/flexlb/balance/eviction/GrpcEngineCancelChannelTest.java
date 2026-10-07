package org.flexlb.balance.eviction;

import org.flexlb.balance.preemption.CancelTarget;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.client.EngineGrpcClient;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import org.mockito.ArgumentCaptor;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

class GrpcEngineCancelChannelTest {
    private static final CancelTarget TARGET = new CancelTarget("10.0.0.1", 8081);

    @ParameterizedTest
    @ValueSource(strings = {"007", "+7", "-0", "request-a", "9223372036854775808"})
    void invalidIdsCannotReachOrCancelAnotherEngineRequest(String requestId) {
        EngineGrpcClient client = mock(EngineGrpcClient.class);
        GrpcEngineCancelChannel channel = new GrpcEngineCancelChannel(client);

        assertThrows(CompletionException.class, () -> channel.cancel(TARGET, requestId, 100L).join());

        verifyNoInteractions(client);
    }

    @ParameterizedTest
    @ValueSource(longs = {0L, 7L, 9007199254740993L, Long.MIN_VALUE, Long.MAX_VALUE})
    void sendsTheExactEngineInt64IdAtTheCancellationBoundary(long requestId) {
        EngineGrpcClient client = mock(EngineGrpcClient.class);
        when(client.cancelAsync(eq(TARGET.prefillIp()), eq(TARGET.prefillGrpcPort()), any(), eq(100L)))
                .thenReturn(CompletableFuture.completedFuture(EngineRpcService.CancelResponsePB.newBuilder()
                        .setStatus(EngineRpcService.CancelStatusPB.CANCEL_STATUS_ACCEPTED).build()));
        GrpcEngineCancelChannel channel = new GrpcEngineCancelChannel(client);

        assertEquals(EngineCancelChannel.CancelAck.ACCEPTED,
                channel.cancel(TARGET, Long.toString(requestId), 100L).join());

        ArgumentCaptor<EngineRpcService.CancelRequestPB> request =
                ArgumentCaptor.forClass(EngineRpcService.CancelRequestPB.class);
        verify(client).cancelAsync(eq(TARGET.prefillIp()), eq(TARGET.prefillGrpcPort()), request.capture(), eq(100L));
        assertEquals(requestId, request.getValue().getRequestId());
    }
}
