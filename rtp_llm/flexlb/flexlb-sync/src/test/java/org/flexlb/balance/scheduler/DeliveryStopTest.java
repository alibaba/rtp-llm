package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.EndpointRegistry;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.Test;

import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.RETURNS_DEEP_STUBS;
import static org.mockito.Mockito.any;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;

class DeliveryStopTest {
    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(ints = {1, 3})
    void cleanupFactsEnterTheSchedulerOncePerRemoteReply(int failures) throws Exception {
        var runtime = runtime();
        var claim = claim();
        when(claim.owner().snapshot().delivery().send())
                .thenReturn(ContextResult.SendState.DELIVERED);
        when(claim.owner().snapshot().cancellation()).thenReturn(null);
        when(claim.owner().snapshot().delivery().abandonment())
                .thenReturn(CancelReason.CLIENT_CANCELLED);
        when(claim.item().prefillEp())
                .thenReturn(mock(org.flexlb.balance.endpoint.PrefillEndpoint.class));
        var scheduler = claim.owner().data().getScheduler();
        SchedulerTestSupport.useRealCancelReply(scheduler);
        var applications = new java.util.concurrent.atomic.AtomicInteger();
        var firstFailure = new java.util.concurrent.atomic.AtomicReference<IllegalStateException>();
        var first =
                new java.util.concurrent.atomic.AtomicReference<ContextResult.DeliveryEvidence>();
        doAnswer(
                        call -> {
                            var evidence =
                                    call.getArgument(1, ContextResult.CleanupAcknowledgement.class);
                            assertEquals(ContextResult.CancelEvidenceKind.CLEANED, evidence.kind());
                            assertSame(claim, evidence.delivery());
                            int count = applications.incrementAndGet();
                            if (count == 1) {
                                first.set(evidence);
                            } else {
                                assertNotSame(first.get(), evidence);
                            }
                            if (count <= failures) {
                                var failure =
                                        new IllegalStateException(
                                                "cleanup fact application failed");
                                firstFailure.compareAndSet(null, failure);
                                throw failure;
                            }
                            return null;
                        })
                .when(scheduler)
                .onDeliveryEvidence(any(), any());
        var client = mock(EngineGrpcClient.class);
        when(client.cancelAsync(
                        any(),
                        org.mockito.ArgumentMatchers.anyInt(),
                        any(),
                        org.mockito.ArgumentMatchers.anyLong()))
                .thenReturn(
                        java.util.concurrent.CompletableFuture.completedFuture(
                                EngineGrpcClient.CancelReply.REQUEST_CLEANED));
        var dispatcher =
                new DeliveryDispatcher(
                        null,
                        DeliveryTestSupport.service(SchedulingTestConfig.batchConfig()),
                        new DeliveryMetricsReporter(null));
        try {
            assertEquals(
                    DeliveryDispatcher.StopResult.REMOTE_CLEANED,
                    dispatcher.stop(claim, client, runtime).get(2, TimeUnit.SECONDS));
            assertEquals(failures + 1, applications.get());
            org.mockito.Mockito.verify(client, org.mockito.Mockito.times(failures + 1))
                    .cancelAsync(
                            any(),
                            org.mockito.ArgumentMatchers.anyInt(),
                            any(),
                            org.mockito.ArgumentMatchers.anyLong());
            assertFalse(runtime.signals().delivery(claim).toCompletableFuture().isDone());
            runtime.signals().completeDelivery(claim);
            SchedulerTestSupport.awaitRequestWork(runtime);
            org.mockito.Mockito.verify(scheduler, org.mockito.Mockito.never())
                    .onPreemption(any(), any());
        } finally {
            runtime.signals().completeDelivery(claim);
            var reported = assertThrows(IllegalStateException.class, runtime::shutdown);
            assertSame(firstFailure.get(), reported);
            assertEquals(failures - 1, reported.getSuppressed().length);
        }
    }

    @Test
    void expiredQueuedPreemptionCannotDiscardAnAcceptedOrdinaryStop() throws Exception {
        var runtime = runtime();
        var original = stopWorkers(runtime);
        var queued = new java.util.concurrent.ConcurrentLinkedQueue<Runnable>();
        var paused = mock(java.util.concurrent.ExecutorService.class);
        doAnswer(
                        call -> {
                            queued.add(call.getArgument(0));
                            return null;
                        })
                .when(paused)
                .execute(any(Runnable.class));
        org.springframework.test.util.ReflectionTestUtils.setField(runtime, "stopWorkers", paused);
        var claim = claim();
        when(claim.owner().snapshot().delivery().send())
                .thenReturn(ContextResult.SendState.DELIVERED);
        when(claim.owner().snapshot().cancellation()).thenReturn(null);
        when(claim.owner().snapshot().delivery().abandonment())
                .thenReturn(CancelReason.CLIENT_CANCELLED);
        when(claim.item().prefillEp())
                .thenReturn(mock(org.flexlb.balance.endpoint.PrefillEndpoint.class));
        var client = mock(EngineGrpcClient.class);
        when(client.cancelAsync(
                        any(),
                        org.mockito.ArgumentMatchers.anyInt(),
                        any(),
                        org.mockito.ArgumentMatchers.anyLong()))
                .thenReturn(
                        java.util.concurrent.CompletableFuture.completedFuture(
                                EngineGrpcClient.CancelReply.REQUEST_CLEANED));
        var dispatcher =
                new DeliveryDispatcher(
                        null,
                        DeliveryTestSupport.service(SchedulingTestConfig.batchConfig()),
                        new DeliveryMetricsReporter(null));
        var registration =
                new PreemptionRegistration(
                        claim.owner(),
                        1L,
                        new org.flexlb.balance.endpoint.DecodeResources.ReservationHandle(
                                1L, 41L, 1L),
                        new org.flexlb.balance.preemption.CancelTarget("127.0.0.1", 9000));
        try {
            var preemption =
                    dispatcher.preempt(
                            registration, client, runtime, 25L, null, "priority preemption");
            var stop = dispatcher.stop(claim, client, runtime);
            assertEquals(PreemptionRegistration.Reply.UNKNOWN, preemption.get(2, TimeUnit.SECONDS));
            Runnable task;
            while ((task = queued.poll()) != null) {
                task.run();
            }
            assertEquals(
                    DeliveryDispatcher.StopResult.REMOTE_CLEANED, stop.get(2, TimeUnit.SECONDS));
            org.mockito.Mockito.verify(client)
                    .cancelAsync(
                            any(),
                            org.mockito.ArgumentMatchers.anyInt(),
                            any(),
                            org.mockito.ArgumentMatchers.anyLong());
        } finally {
            runtime.signals().completeDelivery(claim);
            org.springframework.test.util.ReflectionTestUtils.setField(
                    runtime, "stopWorkers", original);
            Runnable task;
            while ((task = queued.poll()) != null) {
                task.run();
            }
            runtime.shutdown();
        }
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(booleans = {true, false})
    void rejectedPreemptionFactIsReportedOnceEvenAfterWorkSubmissionRejection(boolean rejectWork)
            throws Exception {
        var runtime = org.mockito.Mockito.spy(runtime());
        var claim = claim();
        when(claim.owner().snapshot().delivery()).thenReturn(null);
        var scheduler = claim.owner().data().getScheduler();
        var rejectedFact = new IllegalStateException("transient fact application failure");
        SchedulerTestSupport.useRealCancelReply(scheduler);
        when(scheduler.onPreemption(any(), any())).thenThrow(rejectedFact);
        if (rejectWork) {
            org.mockito.Mockito.doThrow(
                            new java.util.concurrent.RejectedExecutionException(
                                    "request work closed"))
                    .when(runtime)
                    .executeRequestWork(any(), any());
        }
        var dispatcher =
                new DeliveryDispatcher(
                        null,
                        DeliveryTestSupport.service(SchedulingTestConfig.batchConfig()),
                        new DeliveryMetricsReporter(null));
        var client = mock(EngineGrpcClient.class);
        when(client.cancelAsync(
                        any(),
                        org.mockito.ArgumentMatchers.anyInt(),
                        any(),
                        org.mockito.ArgumentMatchers.anyLong()))
                .thenReturn(
                        java.util.concurrent.CompletableFuture.completedFuture(
                                EngineGrpcClient.CancelReply.REQUEST_CLEANED));
        var registration =
                new PreemptionRegistration(
                        claim.owner(),
                        1L,
                        new org.flexlb.balance.endpoint.DecodeResources.ReservationHandle(
                                1L, 41L, 1L),
                        new org.flexlb.balance.preemption.CancelTarget("127.0.0.1", 9000));
        try {
            assertEquals(
                    PreemptionRegistration.Reply.UNKNOWN,
                    dispatcher
                            .preempt(
                                    registration,
                                    client,
                                    runtime,
                                    1000L,
                                    null,
                                    "priority preemption")
                            .get(2, TimeUnit.SECONDS));
            SchedulerTestSupport.awaitRequestWork(runtime);
            org.mockito.Mockito.verify(scheduler).onPreemption(any(), any());
            org.mockito.Mockito.verify(client)
                    .cancelAsync(
                            any(),
                            org.mockito.ArgumentMatchers.anyInt(),
                            any(),
                            org.mockito.ArgumentMatchers.anyLong());
            assertTrue(
                    ((java.util.Map<?, ?>)
                                    org.springframework.test.util.ReflectionTestUtils.getField(
                                            dispatcher, "sessions"))
                            .isEmpty());
            assertFalse(runtime.signals().preemption(registration).toCompletableFuture().isDone());
        } finally {
            assertSame(rejectedFact, assertThrows(IllegalStateException.class, runtime::shutdown));
        }
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(booleans = {true, false})
    void queuedPreemptionExpiresWithoutSendingAfterItsDeadline(boolean hasDelivery)
            throws Exception {
        var runtime = runtime();
        var original = stopWorkers(runtime);
        var queued = new java.util.concurrent.ConcurrentLinkedQueue<Runnable>();
        var paused = mock(java.util.concurrent.ExecutorService.class);
        doAnswer(
                        call -> {
                            queued.add(call.getArgument(0));
                            return null;
                        })
                .when(paused)
                .execute(any(Runnable.class));
        org.springframework.test.util.ReflectionTestUtils.setField(runtime, "stopWorkers", paused);
        var claim = claim();
        if (!hasDelivery) {
            when(claim.owner().snapshot().delivery()).thenReturn(null);
        }
        try {
            var client = mock(EngineGrpcClient.class);
            var registration =
                    new PreemptionRegistration(
                            claim.owner(),
                            1L,
                            new org.flexlb.balance.endpoint.DecodeResources.ReservationHandle(
                                    1L, 41L, 1L),
                            new org.flexlb.balance.preemption.CancelTarget("127.0.0.1", 9000));
            var dispatcher =
                    new DeliveryDispatcher(
                            null,
                            DeliveryTestSupport.service(SchedulingTestConfig.batchConfig()),
                            new DeliveryMetricsReporter(null));
            var answer =
                    dispatcher.preempt(
                            registration, client, runtime, 25L, null, "priority preemption");
            assertEquals(PreemptionRegistration.Reply.UNKNOWN, answer.get(2, TimeUnit.SECONDS));
            Runnable task;
            while ((task = queued.poll()) != null) {
                task.run();
            }
            verifyNoInteractions(client);
            assertTrue(
                    ((java.util.Map<?, ?>)
                                    org.springframework.test.util.ReflectionTestUtils.getField(
                                            dispatcher, "sessions"))
                            .isEmpty());
            assertFalse(runtime.signals().delivery(claim).toCompletableFuture().isDone());
        } finally {
            runtime.signals().completeDelivery(claim);
            org.springframework.test.util.ReflectionTestUtils.setField(
                    runtime, "stopWorkers", original);
            Runnable task;
            while ((task = queued.poll()) != null) {
                task.run();
            }
            runtime.shutdown();
        }
    }

    @Test
    void alreadyPhysicalSettledDoesNotSubmitToClosedExecutorOrLeaveDedupEntry() {
        var runtime = runtime();
        try {
            var claim = claim();
            runtime.signals().completeDelivery(claim);
            stopWorkers(runtime).shutdown();
            when(claim.owner().snapshot().delivery().send())
                    .thenReturn(ContextResult.SendState.NOT_SENT);
            var cleanup =
                    new DeliveryDispatcher(
                            null,
                            DeliveryTestSupport.service(SchedulingTestConfig.batchConfig()),
                            new DeliveryMetricsReporter(null));
            assertDoesNotThrow(
                    () -> cleanup.stop(claim, mock(EngineGrpcClient.class), runtime).join());
            assertDoesNotThrow(
                    () -> cleanup.stop(claim, mock(EngineGrpcClient.class), runtime).join());
        } finally {
            runtime.shutdown();
        }
    }

    @Test
    void physicalSettlementWinningExecutorRejectionDoesNotRecordFalseFailure() {
        var runtime = runtime();
        var original = stopWorkers(runtime);
        try {
            var claim = claim();
            var rejecting = mock(java.util.concurrent.ExecutorService.class);
            doAnswer(
                            call -> {
                                runtime.signals().completeDelivery(claim);
                                throw new java.util.concurrent.RejectedExecutionException(
                                        "closed after physical settlement");
                            })
                    .when(rejecting)
                    .execute(any(Runnable.class));
            org.springframework.test.util.ReflectionTestUtils.setField(
                    runtime, "stopWorkers", rejecting);
            assertDoesNotThrow(
                    () ->
                            new DeliveryDispatcher(
                                            null,
                                            DeliveryTestSupport.service(
                                                    SchedulingTestConfig.batchConfig()),
                                            new DeliveryMetricsReporter(null))
                                    .stop(claim, mock(EngineGrpcClient.class), runtime)
                                    .join());
            assertTrue(runtime.signals().delivery(claim).toCompletableFuture().isDone());
        } finally {
            org.springframework.test.util.ReflectionTestUtils.setField(
                    runtime, "stopWorkers", original);
            // Real shutdown rethrows any recorded facility failure, so this checks the entire
            // lifecycle.
            assertDoesNotThrow(runtime::shutdown);
        }
    }

    @Test
    void stopDeduplicatesTheExactClaimAndRejectsAnEqualValuedClaim() throws Exception {
        var runtime = runtime();
        try {
            var channel = mock(EngineGrpcClient.class);
            var claim = claim();
            var equalValue =
                    new DeliveryClaim(
                            claim.owner(),
                            claim.sequence(),
                            claim.item(),
                            claim.kind(),
                            claim.batchId());
            var cleanup =
                    new DeliveryDispatcher(
                            null,
                            DeliveryTestSupport.service(SchedulingTestConfig.batchConfig()),
                            new DeliveryMetricsReporter(null));
            var first = cleanup.stop(claim, channel, runtime);
            assertSame(first, cleanup.stop(claim, channel, runtime));
            assertEquals(
                    DeliveryDispatcher.StopResult.STALE,
                    cleanup.stop(equalValue, channel, runtime).get(2, TimeUnit.SECONDS));
            assertFalse(first.isDone(), "rejecting a stale claim must not finish the current stop");
            assertFalse(runtime.signals().delivery(claim).toCompletableFuture().isDone());
            runtime.signals().completeDelivery(claim);
            first.get(2, TimeUnit.SECONDS);
            assertFalse(
                    runtime.signals().delivery(equalValue).toCompletableFuture().isDone(),
                    "physical settlement belongs only to the exact claim");
            verifyNoInteractions(channel);
        } finally {
            runtime.shutdown();
        }
    }

    @Test
    void rejectedStartupUsesAcceptedWorkAndDoesNotInventPhysicalSettlement() throws Exception {
        var runtime = runtime();
        var claim = claim();
        try {
            stopWorkers(runtime).shutdown();
            when(claim.owner().snapshot().delivery().send())
                    .thenReturn(ContextResult.SendState.NOT_SENT);
            var cleanup =
                    new DeliveryDispatcher(
                            null,
                            DeliveryTestSupport.service(SchedulingTestConfig.batchConfig()),
                            new DeliveryMetricsReporter(null));
            var first = cleanup.stop(claim, mock(EngineGrpcClient.class), runtime);
            assertEquals(DeliveryDispatcher.StopResult.NOT_SENT, first.get(2, TimeUnit.SECONDS));
            assertSame(first, cleanup.stop(claim, mock(EngineGrpcClient.class), runtime));
            assertFalse(runtime.signals().delivery(claim).toCompletableFuture().isDone());
            runtime.signals().completeDelivery(claim);
            var after = cleanup.stop(claim, mock(EngineGrpcClient.class), runtime);
            assertNotSame(first, after);
            assertDoesNotThrow(after::join);
        } finally {
            assertDoesNotThrow(runtime::shutdown);
        }
    }

    @Test
    void settlementBeforeQueuedPreemptionDoesNotLeaveItsCallerWaiting() throws Exception {
        var runtime = runtime();
        var original = stopWorkers(runtime);
        var queued = new java.util.concurrent.ConcurrentLinkedQueue<Runnable>();
        var paused = mock(java.util.concurrent.ExecutorService.class);
        doAnswer(
                        call -> {
                            queued.add(call.getArgument(0));
                            return null;
                        })
                .when(paused)
                .execute(any(Runnable.class));
        org.springframework.test.util.ReflectionTestUtils.setField(runtime, "stopWorkers", paused);
        try {
            var claim = claim();
            var client = mock(EngineGrpcClient.class);
            var cleanup =
                    new DeliveryDispatcher(
                            null,
                            DeliveryTestSupport.service(SchedulingTestConfig.batchConfig()),
                            new DeliveryMetricsReporter(null));
            cleanup.stop(claim, client, runtime);
            runtime.signals().completeDelivery(claim);
            var registration =
                    new PreemptionRegistration(
                            claim.owner(),
                            1L,
                            new org.flexlb.balance.endpoint.DecodeResources.ReservationHandle(
                                    1L, 41L, 1L),
                            new org.flexlb.balance.preemption.CancelTarget("127.0.0.1", 9000));
            var answer =
                    cleanup.preempt(
                            registration, client, runtime, 500L, null, "priority preemption");
            Runnable task;
            while ((task = queued.poll()) != null) {
                task.run();
            }
            assertEquals(PreemptionRegistration.Reply.UNKNOWN, answer.get(1, TimeUnit.SECONDS));
            verifyNoInteractions(client);
        } finally {
            org.springframework.test.util.ReflectionTestUtils.setField(
                    runtime, "stopWorkers", original);
            runtime.shutdown();
        }
    }

    private static java.util.concurrent.ExecutorService stopWorkers(SchedulerRuntime runtime) {
        return (java.util.concurrent.ExecutorService)
                org.springframework.test.util.ReflectionTestUtils.getField(runtime, "stopWorkers");
    }

    private static DeliveryClaim claim() {
        var context = mock(RequestContext.class, RETURNS_DEEP_STUBS);
        var claim =
                new DeliveryClaim(
                        context, 1, mock(RequestRoute.class), DeliveryClaimKind.BATCH_ENQUEUE, 17);
        when(context.snapshot().delivery().identity()).thenReturn(claim);
        when(context.snapshot().delivery().send()).thenReturn(ContextResult.SendState.NOT_STARTED);
        return claim;
    }

    private static SchedulerRuntime runtime() {
        var config = DeliveryTestSupport.service(SchedulingTestConfig.batchConfig());
        return new SchedulerRuntime(
                new RequestRepository(),
                mock(EndpointRegistry.class),
                mock(DeliveryMetricsReporter.class),
                mock(RequestSchedulerReporter.class),
                new DeliveryDispatcher(null, config, new DeliveryMetricsReporter(null)),
                config,
                mock(EngineGrpcClient.class));
    }
}
