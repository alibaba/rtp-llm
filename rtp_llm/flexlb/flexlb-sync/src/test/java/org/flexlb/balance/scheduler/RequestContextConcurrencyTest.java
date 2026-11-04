package org.flexlb.balance.scheduler;

import org.flexlb.dao.loadbalance.Request;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static org.flexlb.balance.scheduler.ContextResult.Cancellation;
import static org.flexlb.balance.scheduler.ContextResult.PublicationResult;
import static org.flexlb.balance.scheduler.ContextResult.SendState;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/** Exercises real Context transitions; the route supplies immutable identity only. */
@Timeout(30)
class RequestContextConcurrencyTest {
    enum Order {
        CANCEL_FIRST,
        SEND_FIRST,
        TOGETHER
    }

    @ParameterizedTest
    @EnumSource(Order.class)
    void cancelAndSendHaveOneLegalOrderAndSnapshotsNeverLoseCancellation(Order order)
            throws Exception {
        var workers = Executors.newFixedThreadPool(3, Thread.ofPlatform().daemon().factory());
        try {
            for (int round = 0; round < 100; round++) {
                var context = new RequestContext(SchedulingTestConfig.batchConfig());
                context.data().setRequest(Request.builder().requestId(round + 1L).build());
                context.register(round + 1L, 1000L, 100_000L, 10_000L);
                var route = mock(RequestRoute.class);
                when(route.ctx()).thenReturn(context);
                when(route.requestId()).thenReturn(round + 1L);
                var placement = context.beginPlacement(null);
                assertTrue(context.bindPlacement(placement, route, PublicationResult.PREPARED));
                assertTrue(context.bindPlacement(placement, route, PublicationResult.PUBLISHED));
                context.completePlacement(placement, PublicationResult.PUBLISHED, 1000L);
                var member = new Delivery.Member(route, null);
                var prepared =
                        context.prepareDelivery(member, Delivery.SendMode.BATCH_ENQUEUE, 1L, 1001L);
                assertNotNull(prepared);
                context.publishDelivery(prepared);
                var before = context.snapshot();
                var start = new CyclicBarrier(3);
                var cancelled = new CountDownLatch(1);
                var sent = new CountDownLatch(1);
                var cancel =
                        workers.submit(
                                () -> {
                                    start.await(5, TimeUnit.SECONDS);
                                    if (order == Order.SEND_FIRST)
                                        assertTrue(sent.await(5, TimeUnit.SECONDS));
                                    try {
                                        return context.requestCancel(
                                                new Cancellation(
                                                        1L,
                                                        CancelReason.CLIENT_CANCELLED,
                                                        false,
                                                        1002L,
                                                        null,
                                                        "concurrent cancellation"));
                                    } finally {
                                        cancelled.countDown();
                                    }
                                });
                var send =
                        workers.submit(
                                () -> {
                                    start.await(5, TimeUnit.SECONDS);
                                    if (order == Order.CANCEL_FIRST)
                                        assertTrue(cancelled.await(5, TimeUnit.SECONDS));
                                    try {
                                        return context.beginSend(member, 1002L);
                                    } finally {
                                        sent.countDown();
                                    }
                                });
                var reader =
                        workers.submit(
                                () -> {
                                    start.await(5, TimeUnit.SECONDS);
                                    boolean sawCancellation = false;
                                    for (int sample = 0; sample < 200; sample++) {
                                        var snapshot = context.snapshot();
                                        assertSame(member, snapshot.delivery().identity());
                                        if (sawCancellation) assertNotNull(snapshot.cancellation());
                                        if (snapshot.cancellation() != null) {
                                            sawCancellation = true;
                                            assertNotNull(snapshot.pendingEnd());
                                            assertFalse(snapshot.canConfirmDelivery());
                                        }
                                    }
                                    return null;
                                });
                assertTrue(cancel.get(5, TimeUnit.SECONDS));
                var result = send.get(5, TimeUnit.SECONDS);
                reader.get(5, TimeUnit.SECONDS);
                if (order == Order.CANCEL_FIRST)
                    assertEquals(RequestContext.SendResult.CANCELLED, result);
                if (order == Order.SEND_FIRST)
                    assertEquals(RequestContext.SendResult.STARTED, result);
                assertTrue(
                        result == RequestContext.SendResult.CANCELLED
                                || result == RequestContext.SendResult.STARTED);
                var after = context.snapshot();
                assertEquals(
                        result == RequestContext.SendResult.STARTED
                                ? SendState.SENDING
                                : SendState.NOT_STARTED,
                        after.delivery().send());
                assertNotNull(after.cancellation());
                assertFalse(context.confirmDelivery(1003L));
                assertNotEquals(
                        RequestContext.SendResult.STARTED, context.beginSend(member, 1003L));
                assertNull(before.cancellation(), "published snapshots must remain immutable");
                assertEquals(SendState.NOT_STARTED, before.delivery().send());
            }
        } finally {
            workers.shutdownNow();
            assertTrue(workers.awaitTermination(5, TimeUnit.SECONDS));
        }
    }
}
