package org.flexlb.balance.scheduler;

import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static org.flexlb.balance.scheduler.ContextResult.SendState;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

/**
 * Exercises the real scheduler's atomic delivery commit against concurrent cancellation.
 */
@Timeout(30)
class RequestContextConcurrencyTest {
    @ParameterizedTest
    @EnumSource(Order.class)
    void cancelAndCommitHaveOneLegalOrderAndSnapshotsNeverLoseCancellation(Order order)
            throws Exception {
        var config = SchedulingTestConfig.batchConfig();
        var owner =
                SchedulerTestSupport.create(
                        SchedulingTestConfig.configService(config),
                        mock(org.flexlb.service.monitor.DeliveryMetricsReporter.class),
                        mock(org.flexlb.service.monitor.RequestSchedulerReporter.class),
                        mock(org.flexlb.service.TheoryCacheHitReporter.class));
        var runtime = SchedulerTestSupport.runtime(owner);
        var channel = mock(org.flexlb.engine.grpc.EngineGrpcClient.class);
        when(channel.cancelAsync(
                        org.mockito.ArgumentMatchers.any(),
                        org.mockito.ArgumentMatchers.anyInt(),
                        org.mockito.ArgumentMatchers.any(),
                        org.mockito.ArgumentMatchers.anyLong()))
                .thenReturn(
                        java.util.concurrent.CompletableFuture.completedFuture(
                                org.flexlb.engine.grpc.EngineGrpcClient.CancelReply.FAILED));
        org.flexlb.balance.scheduler.DeliveryTestSupport.setField(
                runtime.dispatcher, "grpcClient", channel);
        var workers = Executors.newFixedThreadPool(3, Thread.ofPlatform().daemon().factory());
        try {
            for (int round = 0; round < 100; round++) {
                var context = RequestProtocolTestSupport.context(config, round + 1L);
                var future = RequestProtocolTestSupport.submitForManualPlacement(owner, context);
                var endpoint = mock(org.flexlb.balance.endpoint.PrefillEndpoint.class);
                when(endpoint.getIp()).thenReturn("127.0.0.1");
                when(endpoint.getHttpPort()).thenReturn(8080);
                when(endpoint.getGrpcPort()).thenReturn(8090);
                var route =
                        SchedulingTestConfig.createRoute(
                                context,
                                null,
                                null,
                                null,
                                endpoint,
                                null,
                                null,
                                System.currentTimeMillis());
                RequestProtocolTestSupport.bind(
                        owner, new RequestProtocolTestSupport.Registered(route, future));
                var transfers = new java.util.concurrent.atomic.AtomicInteger();
                var member =
                        new Delivery.Member(
                                route,
                                RequestProtocolTestSupport.handoff(
                                        () -> {
                                            transfers.incrementAndGet();
                                            return true;
                                        }));
                var before = context.snapshot();
                var start = new CyclicBarrier(3);
                var cancelled = new CountDownLatch(1);
                var committed = new CountDownLatch(1);
                var cancel =
                        workers.submit(
                                () -> {
                                    start.await(5, TimeUnit.SECONDS);
                                    if (order == Order.COMMIT_FIRST)
                                        assertTrue(committed.await(5, TimeUnit.SECONDS));
                                    try {
                                        return owner.cancel(
                                                context, 0L, CancelReason.CLIENT_CANCELLED);
                                    } finally {
                                        cancelled.countDown();
                                    }
                                });
                var commit =
                        workers.submit(
                                () -> {
                                    start.await(5, TimeUnit.SECONDS);
                                    if (order == Order.CANCEL_FIRST)
                                        assertTrue(cancelled.await(5, TimeUnit.SECONDS));
                                    try {
                                        return org.flexlb.balance.scheduler.DeliveryTestSupport
                                                .transport(owner.dispatcher)
                                                .commit(
                                                        member,
                                                        Delivery.SendMode.BATCH_ENQUEUE,
                                                        1L,
                                                        null);
                                    } finally {
                                        committed.countDown();
                                    }
                                });
                var reader =
                        workers.submit(
                                () -> {
                                    start.await(5, TimeUnit.SECONDS);
                                    boolean sawCancellation = false;
                                    for (int sample = 0; sample < 200; sample++) {
                                        var snapshot = context.snapshot();
                                        if (snapshot.delivery() != null) {
                                            assertSame(member, snapshot.delivery().identity());
                                            assertEquals(
                                                    SendState.SENDING, snapshot.delivery().send());
                                        }
                                        if (sawCancellation) assertNotNull(snapshot.cancellation());
                                        if (snapshot.cancellation() != null) {
                                            sawCancellation = true;
                                            assertTrue(
                                                    snapshot.pendingEnd() != null
                                                            || snapshot.outcome() != null);
                                            assertFalse(snapshot.canConfirmDelivery());
                                        }
                                    }
                                    return null;
                                });
                assertNotNull(cancel.get(5, TimeUnit.SECONDS));
                var result = commit.get(5, TimeUnit.SECONDS);
                reader.get(5, TimeUnit.SECONDS);
                if (order == Order.CANCEL_FIRST) assertNull(result);
                if (order == Order.COMMIT_FIRST) assertSame(member, result);
                var after = context.snapshot();
                if (result == null) assertNull(after.delivery());
                else {
                    assertSame(member, result);
                    assertEquals(SendState.SENDING, after.delivery().send());
                }
                assertEquals(result == null ? 0 : 1, transfers.get());
                assertNotNull(after.cancellation());
                assertFalse(context.confirmDelivery(System.currentTimeMillis()));
                assertNull(
                        org.flexlb.balance.scheduler.DeliveryTestSupport.transport(owner.dispatcher)
                                .commit(
                                        new Delivery.Member(route, null),
                                        Delivery.SendMode.BATCH_ENQUEUE,
                                        1L,
                                        null));
                assertNull(before.cancellation(), "published snapshots must remain immutable");
                assertNull(before.delivery(), "the pre-commit snapshot must not acquire a sender");
                if (result != null) {
                    owner.onDeliveryResult(
                            new ContextResult.TransportResult(
                                    member,
                                    Delivery.Result.notSent(
                                            new java.util.concurrent.CancellationException()),
                                    System.currentTimeMillis()));
                }
                assertFalse(future.get(5, TimeUnit.SECONDS).isSuccess());
                RequestProtocolTestSupport.awaitCondition(
                        () -> !owner.repository.isCurrent(context));
                org.mockito.Mockito.verify(endpoint, org.mockito.Mockito.times(1))
                        .releaseRequest(route);
                member.close();
            }
        } finally {
            workers.shutdownNow();
            assertTrue(workers.awaitTermination(5, TimeUnit.SECONDS));
            SchedulerTestSupport.shutdown(runtime);
        }
    }

    enum Order {
        CANCEL_FIRST,
        COMMIT_FIRST,
        TOGETHER
    }
}
