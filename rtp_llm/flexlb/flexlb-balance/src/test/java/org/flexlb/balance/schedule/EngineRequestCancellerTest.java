package org.flexlb.balance.schedule;

import org.flexlb.worker.rpc.EngineGrpcClient;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.RETURNS_DEEP_STUBS;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Verifies that exhausted cancellation rounds retain the exact route and retry until cleanup is confirmed.
 */
@Timeout(15)
class EngineRequestCancellerTest {
    @Test
    void unresolvedRoundRetriesWithoutReleasingTheRoute() throws Exception {
        var context = mock(RequestContext.class, RETURNS_DEEP_STUBS);
        var route = mock(RequestRoute.class, RETURNS_DEEP_STUBS);
        var cleaned = new AtomicBoolean();
        when(route.ctx()).thenReturn(context);
        when(context.getRequestId()).thenReturn("41");
        when(context.snapshot().committedRoute()).thenReturn(route);
        when(context.snapshot().cancellationReason()).thenReturn(CancelReason.CLIENT_CANCELLED);
        when(context.snapshot().prefillWorkerReleaseConfirmed()).thenAnswer(call -> cleaned.get());
        when(context.snapshot().decodeWorkerReleaseConfirmed()).thenAnswer(call -> cleaned.get());
        var client = mock(EngineGrpcClient.class);
        var calls = new AtomicInteger();
        when(client.cancelAsync(any(), anyInt(), any(), anyLong()))
                .thenAnswer(
                        call ->
                                CompletableFuture.completedFuture(
                                        calls.incrementAndGet() <= 8
                                                ? EngineGrpcClient.CancelReply.FAILED
                                                : EngineGrpcClient.CancelReply.REQUEST_CLEANED));
        var cancellation = new EngineRequestCanceller(client);
        try {
            cancellation
                    .cancelRequest(
                            route,
                            (exact, reply) -> {
                                assertSame(route, exact);
                                if (reply == EngineGrpcClient.CancelReply.REQUEST_CLEANED)
                                    cleaned.set(true);
                            },
                            failure -> {})
                    .get(6, TimeUnit.SECONDS);
            assertEquals(8, calls.get());
            assertFalse(cleaned.get());
            assertSame(route, context.snapshot().committedRoute());
            RequestProtocolTestSupport.awaitCondition(
                    () -> {
                        cancellation.retryUnresolvedStops();
                        return cleaned.get();
                    });
            assertEquals(9, calls.get());
        } finally {
            cancellation.remove(route);
            cancellation.awaitStops();
        }
    }
}
