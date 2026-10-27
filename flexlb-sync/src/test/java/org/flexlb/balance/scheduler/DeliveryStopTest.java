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
    @Test
    void acceptance_preemptionRepliesBelongToTheirRegistration() throws Exception {
        var runtime = runtime();
        var claim = sentClaim(runtime);
        var firstRpc = new java.util.concurrent.CompletableFuture<EngineGrpcClient.CancelReply>();
        var secondRpc = new java.util.concurrent.CompletableFuture<EngineGrpcClient.CancelReply>();
        when(runtime.engineClient.cancelAsync(
                        any(),
                        org.mockito.ArgumentMatchers.anyInt(),
                        any(),
                        org.mockito.ArgumentMatchers.anyLong()))
                .thenReturn(firstRpc, secondRpc);
        var scheduler = claim.owner().data().getScheduler();
        org.mockito.Mockito.doReturn(true).when(scheduler).onPreemption(any(), any());
        var dispatcher =
                new DeliveryDispatcher(
                        null,
                        DeliveryTestSupport.service(SchedulingTestConfig.batchConfig()),
                        new DeliveryMetricsReporter(null));
        var first = preemption(claim, 1L);
        var second = preemption(claim, 2L);
        try {
            var firstAnswer = dispatcher.preempt(first, null, "first", 5_000L);
            RequestProtocolTestSupport.awaitCondition(() -> firstRpc.getNumberOfDependents() != 0);
            var secondAnswer = dispatcher.preempt(second, null, "second", 5_000L);

            firstRpc.complete(EngineGrpcClient.CancelReply.NOT_FOUND);
            assertEquals(PreemptionRegistration.Reply.NOT_FOUND, firstAnswer.get(2, TimeUnit.SECONDS));
            assertFalse(
                    secondAnswer.isDone(),
                    "a later registration must not consume the earlier registration's reply");

            secondRpc.complete(EngineGrpcClient.CancelReply.REQUEST_CLEANED);
            assertEquals(PreemptionRegistration.Reply.WAITING, secondAnswer.get(2, TimeUnit.SECONDS));
        } finally {
            runtime.signals().completeDelivery(claim);
            runtime.shutdown();
        }
    }

    @Test
    void acceptance_preemptionTimeoutDoesNotDiscardLateRemoteReceipt() throws Exception {
        var runtime = runtime();
        var claim = sentClaim(runtime);
        var rpc = new java.util.concurrent.CompletableFuture<EngineGrpcClient.CancelReply>();
        when(runtime.engineClient.cancelAsync(
                        any(),
                        org.mockito.ArgumentMatchers.anyInt(),
                        any(),
                        org.mockito.ArgumentMatchers.anyLong()))
                .thenReturn(rpc);
        var scheduler = claim.owner().data().getScheduler();
        var dispatcher =
                new DeliveryDispatcher(
                        null,
                        DeliveryTestSupport.service(SchedulingTestConfig.batchConfig()),
                        new DeliveryMetricsReporter(null));
        try {
            var answer = dispatcher.preempt(preemption(claim, 1L), null, "deadline", 25L);
            RequestProtocolTestSupport.awaitCondition(() -> rpc.getNumberOfDependents() != 0);
            assertEquals(PreemptionRegistration.Reply.UNKNOWN, answer.get(2, TimeUnit.SECONDS));

            rpc.complete(EngineGrpcClient.CancelReply.REQUEST_CLEANED);
            org.mockito.Mockito.verify(scheduler, org.mockito.Mockito.timeout(2_000L))
                    .onPreemption(any(), any());
        } finally {
            runtime.signals().completeDelivery(claim);
            runtime.shutdown();
        }
    }

    @Test
    void acceptance_rejectedPreemptionDoesNotReuseAnotherRegistrationReply() throws Exception {
        var runtime = runtime();
        var claim = sentClaim(runtime);
        var firstRpc = new java.util.concurrent.CompletableFuture<EngineGrpcClient.CancelReply>();
        when(runtime.engineClient.cancelAsync(
                        any(),
                        org.mockito.ArgumentMatchers.anyInt(),
                        any(),
                        org.mockito.ArgumentMatchers.anyLong()))
                .thenReturn(firstRpc);
        var dispatcher =
                new DeliveryDispatcher(
                        null,
                        DeliveryTestSupport.service(SchedulingTestConfig.batchConfig()),
                        new DeliveryMetricsReporter(null));
        try {
            var first = dispatcher.preempt(preemption(claim, 1L), null, "first", 5_000L);
            RequestProtocolTestSupport.awaitCondition(() -> firstRpc.getNumberOfDependents() != 0);
            var stop = dispatcher.stop(claim);
            var rejected = dispatcher.preempt(preemption(claim, 2L), null, "rejected", 5_000L);

            assertEquals(PreemptionRegistration.Reply.UNKNOWN, rejected.get(2, TimeUnit.SECONDS));
            firstRpc.complete(EngineGrpcClient.CancelReply.REQUEST_CLEANED);
            assertEquals(DeliveryDispatcher.StopResult.REMOTE_CLEANED, stop.get(2, TimeUnit.SECONDS));
            assertEquals(PreemptionRegistration.Reply.UNKNOWN, first.get(2, TimeUnit.SECONDS));
        } finally {
            firstRpc.complete(EngineGrpcClient.CancelReply.REQUEST_CLEANED);
            runtime.signals().completeDelivery(claim);
            runtime.shutdown();
        }
    }

    @Test
    void acceptance_unsentStopDoesNotIssueRemoteCancel() throws Exception {
        var runtime = runtime();
        var claim = claim(runtime);
        when(claim.owner().snapshot().delivery().send()).thenReturn(ContextResult.SendState.NOT_SENT);
        var dispatcher =
                new DeliveryDispatcher(
                        null,
                        DeliveryTestSupport.service(SchedulingTestConfig.batchConfig()),
                        new DeliveryMetricsReporter(null));
        try {
            var result = dispatcher.stop(claim).get(2, TimeUnit.SECONDS);
            assertEquals(DeliveryDispatcher.StopResult.NOT_SENT, result);
            verifyNoInteractions(runtime.engineClient);
            assertFalse(runtime.signals().delivery(claim).toCompletableFuture().isDone());
        } finally {
            runtime.signals().completeDelivery(claim);
            runtime.shutdown();
        }
    }

    @Test
    void acceptance_remoteProofDoesNotReleaseLocalSettlement() throws Exception {
        var runtime = runtime();
        var claim = sentClaim(runtime);
        when(runtime.engineClient.cancelAsync(
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
                    dispatcher.stop(claim).get(2, TimeUnit.SECONDS));
            assertFalse(
                    runtime.signals().delivery(claim).toCompletableFuture().isDone(),
                    "remote proof must not fabricate local settlement");
            runtime.signals().completeDelivery(claim);
            assertTrue(runtime.signals().delivery(claim).toCompletableFuture().isDone());
        } finally {
            runtime.signals().completeDelivery(claim);
            runtime.shutdown();
        }
    }

    @Test
    void acceptance_exactClaimHasOneOperation() throws Exception {
        var runtime = runtime();
        var claim = claim(runtime);
        var equalValue =
                new DeliveryClaim(
                        claim.owner(), claim.sequence(), claim.item(), claim.kind(), claim.batchId());
        var dispatcher =
                new DeliveryDispatcher(
                        null,
                        DeliveryTestSupport.service(SchedulingTestConfig.batchConfig()),
                        new DeliveryMetricsReporter(null));
        try {
            var first = dispatcher.stop(claim);
            assertSame(first, dispatcher.stop(claim));
            assertEquals(
                    DeliveryDispatcher.StopResult.STALE,
                    dispatcher.stop(equalValue).get(2, TimeUnit.SECONDS));
            runtime.signals().completeDelivery(claim);
            first.get(2, TimeUnit.SECONDS);
            verifyNoInteractions(runtime.engineClient);
        } finally {
            runtime.signals().completeDelivery(claim);
            runtime.shutdown();
        }
    }

    @Test
    void physicalSettlementWakesStopWithoutWaitingForThePreemptionRpcDeadline() throws Exception {
        var runtime = runtime();
        var claim = sentClaim(runtime);
        var reply = new java.util.concurrent.CompletableFuture<EngineGrpcClient.CancelReply>();
        when(runtime.engineClient.cancelAsync(
                        any(),
                        org.mockito.ArgumentMatchers.anyInt(),
                        any(),
                        org.mockito.ArgumentMatchers.anyLong()))
                .thenReturn(reply);
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
        var closed = new java.util.concurrent.CompletableFuture<Void>();
        Thread closer = null;
        try {
            var preemption = dispatcher.preempt(registration, null, "priority preemption", 5_000L);
            RequestProtocolTestSupport.awaitCondition(() -> reply.getNumberOfDependents() != 0);
            preemption.cancel(false);
            dispatcher.stop(claim);
            var sessions =
                    (java.util.Map<?, ?>)
                            org.springframework.test.util.ReflectionTestUtils.getField(
                                    dispatcher, "sessions");
            var task = sessions.get(claim);
            RequestProtocolTestSupport.awaitCondition(
                    () -> {
                        var runner =
                                (Thread)
                                        org.springframework.test.util.ReflectionTestUtils.getField(
                                                task, "runner");
                        return runner != null && runner.getState() == Thread.State.TIMED_WAITING;
                    });
            runtime.signals().completeDelivery(claim);
            closer =
                    Thread.ofVirtual()
                            .start(
                                    () -> {
                                        try {
                                            dispatcher.awaitStops(runtime);
                                            runtime.shutdown();
                                            closed.complete(null);
                                        } catch (Throwable failure) {
                                            closed.completeExceptionally(failure);
                                        }
                                    });
            closed.get(2, TimeUnit.SECONDS);
            assertFalse(reply.isDone(), "local shutdown must not fabricate a remote response");
        } finally {
            reply.complete(EngineGrpcClient.CancelReply.REQUEST_CLEANED);
            runtime.signals().completeDelivery(claim);
            if (closer != null) {
                closer.join(6_000L);
            }
            runtime.shutdown();
        }
    }

    @Test
    void stopUsesVirtualThreadAndAppliesReplyOnItsCompletingThread() throws Exception {
        var runtime = runtime();
        var claim = sentClaim(runtime);
        var reply = new java.util.concurrent.CompletableFuture<EngineGrpcClient.CancelReply>();
        var sender = new java.util.concurrent.CompletableFuture<Thread>();
        var applied = new java.util.concurrent.CompletableFuture<Thread>();
        var scheduler = claim.owner().data().getScheduler();
        var channel = runtime.engineClient;
        when(channel.cancelAsync(
                        any(),
                        org.mockito.ArgumentMatchers.anyInt(),
                        any(),
                        org.mockito.ArgumentMatchers.anyLong()))
                .thenAnswer(
                        call -> {
                            assertFalse(Thread.holdsLock(claim.owner()));
                            sender.complete(Thread.currentThread());
                            return reply;
                        });
        doAnswer(
                        call -> {
                            applied.complete(Thread.currentThread());
                            return null;
                        })
                .when(scheduler)
                .onDeliveryEvidence(any(), any());
        var dispatcher =
                new DeliveryDispatcher(
                        null,
                        DeliveryTestSupport.service(SchedulingTestConfig.batchConfig()),
                        new DeliveryMetricsReporter(null));
        try {
            var stop = dispatcher.stop(claim);
            assertTrue(sender.get(2, TimeUnit.SECONDS).isVirtual());
            RequestProtocolTestSupport.awaitCondition(() -> reply.getNumberOfDependents() != 0);
            reply.complete(EngineGrpcClient.CancelReply.REQUEST_CLEANED);
            assertSame(Thread.currentThread(), applied.get(2, TimeUnit.SECONDS));
            assertEquals(
                    DeliveryDispatcher.StopResult.REMOTE_CLEANED, stop.get(2, TimeUnit.SECONDS));
            assertSame(stop, dispatcher.stop(claim));
            assertFalse(runtime.signals().delivery(claim).toCompletableFuture().isDone());
        } finally {
            runtime.signals().completeDelivery(claim);
            runtime.shutdown();
        }
    }

    @Test
    void lateCleanupProofSurvivesRetryExhaustionAndDrainWaitsForItsApplication() throws Exception {
        var runtime = runtime();
        var claim = sentClaim(runtime);
        var lateReply = new java.util.concurrent.CompletableFuture<EngineGrpcClient.CancelReply>();
        var calls = new java.util.concurrent.atomic.AtomicInteger();
        var proofs = new java.util.concurrent.atomic.AtomicInteger();
        var scheduler = claim.owner().data().getScheduler();
        var channel = runtime.engineClient;
        when(channel.cancelAsync(
                        any(),
                        org.mockito.ArgumentMatchers.anyInt(),
                        any(),
                        org.mockito.ArgumentMatchers.anyLong()))
                .thenAnswer(
                        call ->
                                calls.incrementAndGet() == 1
                                        ? lateReply
                                        : java.util.concurrent.CompletableFuture.completedFuture(
                                                EngineGrpcClient.CancelReply.FAILED));
        doAnswer(
                        call -> {
                            var evidence =
                                    call.getArgument(1, ContextResult.CleanupAcknowledgement.class);
                            if (evidence.kind() == ContextResult.CancelEvidenceKind.CLEANED) {
                                proofs.incrementAndGet();
                            }
                            return null;
                        })
                .when(scheduler)
                .onDeliveryEvidence(any(), any());
        var dispatcher =
                new DeliveryDispatcher(
                        null,
                        DeliveryTestSupport.service(SchedulingTestConfig.batchConfig()),
                        new DeliveryMetricsReporter(null));
        var drained = new java.util.concurrent.CompletableFuture<Void>();
        Thread closer = null;
        try {
            var stop = dispatcher.stop(claim);
            assertEquals(DeliveryDispatcher.StopResult.UNRESOLVED, stop.get(8, TimeUnit.SECONDS));
            assertEquals(8, calls.get());
            assertSame(stop, dispatcher.stop(claim));
            closer =
                    Thread.ofVirtual()
                            .start(
                                    () -> {
                                        try {
                                            dispatcher.awaitStops(runtime);
                                            drained.complete(null);
                                        } catch (Throwable failure) {
                                            drained.completeExceptionally(failure);
                                        }
                                    });
            assertThrows(
                    java.util.concurrent.TimeoutException.class,
                    () -> drained.get(50, TimeUnit.MILLISECONDS));
            lateReply.complete(EngineGrpcClient.CancelReply.REQUEST_CLEANED);
            drained.get(2, TimeUnit.SECONDS);
            assertEquals(1, proofs.get());
            assertEquals(8, calls.get());
            assertFalse(runtime.signals().delivery(claim).toCompletableFuture().isDone());
        } finally {
            lateReply.complete(EngineGrpcClient.CancelReply.REQUEST_CLEANED);
            runtime.signals().completeDelivery(claim);
            if (closer != null) {
                closer.join(2_000);
            }
            assertThrows(IllegalStateException.class, runtime::shutdown);
        }
    }

    @Test
    void aWaitingStopDoesNotOccupyTheSenderForOtherRequests() throws Exception {
        var runtime = runtime();
        var first = claim(runtime);
        var second = sentClaim(runtime);
        var channel = runtime.engineClient;
        when(channel.cancelAsync(
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
            var waiting = dispatcher.stop(first);
            var other = dispatcher.stop(second);
            assertEquals(
                    DeliveryDispatcher.StopResult.REMOTE_CLEANED, other.get(2, TimeUnit.SECONDS));
            assertFalse(waiting.isDone());
            runtime.signals().completeDelivery(first);
            assertEquals(
                    DeliveryDispatcher.StopResult.ALREADY_SETTLED,
                    waiting.get(2, TimeUnit.SECONDS));
        } finally {
            runtime.signals().completeDelivery(first);
            runtime.signals().completeDelivery(second);
            runtime.shutdown();
        }
    }

    private static DeliveryClaim sentClaim(SchedulerRuntime runtime) {
        var claim = claim(runtime);
        when(claim.owner().snapshot().delivery().send())
                .thenReturn(ContextResult.SendState.DELIVERED);
        when(claim.owner().snapshot().cancellation()).thenReturn(null);
        when(claim.owner().snapshot().delivery().abandonment())
                .thenReturn(CancelReason.CLIENT_CANCELLED);
        when(claim.item().prefillEp())
                .thenReturn(mock(org.flexlb.balance.endpoint.PrefillEndpoint.class));
        return claim;
    }

    private static PreemptionRegistration preemption(DeliveryClaim claim, long attemptToken) {
        return new PreemptionRegistration(
                claim.owner(),
                attemptToken,
                new org.flexlb.balance.endpoint.DecodeResources.ReservationHandle(
                        attemptToken, 41L, attemptToken),
                new org.flexlb.balance.preemption.CancelTarget("127.0.0.1", 9000));
    }

    @Test
    void staleStopRetiresWithoutWaitingForAnIrrelevantReplyOrReleasingTheReplacement()
            throws Exception {
        var runtime = runtime();
        var claim = sentClaim(runtime);
        var reply = new java.util.concurrent.CompletableFuture<EngineGrpcClient.CancelReply>();
        var channel = runtime.engineClient;
        when(channel.cancelAsync(
                        any(),
                        org.mockito.ArgumentMatchers.anyInt(),
                        any(),
                        org.mockito.ArgumentMatchers.anyLong()))
                .thenReturn(reply);
        var dispatcher =
                new DeliveryDispatcher(
                        null,
                        DeliveryTestSupport.service(SchedulingTestConfig.batchConfig()),
                        new DeliveryMetricsReporter(null));
        var replacement =
                new DeliveryClaim(claim.owner(), 2L, claim.item(), claim.kind(), claim.batchId());
        try {
            var stop = dispatcher.stop(claim);
            RequestProtocolTestSupport.awaitCondition(() -> reply.getNumberOfDependents() != 0);
            when(claim.owner().snapshot().delivery().identity()).thenReturn(replacement);
            var sessions =
                    (java.util.Map<?, ?>)
                            org.springframework.test.util.ReflectionTestUtils.getField(
                                    dispatcher, "sessions");
            org.springframework.test.util.ReflectionTestUtils.invokeMethod(
                    sessions.get(claim), "signal");
            assertEquals(DeliveryDispatcher.StopResult.STALE, stop.get(2, TimeUnit.SECONDS));
            assertTrue(sessions.isEmpty(), "stale identity no longer owns cleanup");
            reply.complete(EngineGrpcClient.CancelReply.REQUEST_CLEANED);
            dispatcher.awaitStops(runtime);
            assertTrue(sessions.isEmpty());
            assertFalse(runtime.signals().delivery(replacement).toCompletableFuture().isDone());
        } finally {
            reply.complete(EngineGrpcClient.CancelReply.REQUEST_CLEANED);
            runtime.signals().completeDelivery(claim);
            runtime.shutdown();
        }
    }

    @org.junit.jupiter.params.ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(ints = {1, 3})
    void cleanupFactsEnterTheSchedulerOncePerRemoteReply(int failures) throws Exception {
        var runtime = runtime();
        var claim = claim(runtime);
        when(claim.owner().snapshot().delivery().send())
                .thenReturn(ContextResult.SendState.DELIVERED);
        when(claim.owner().snapshot().cancellation()).thenReturn(null);
        when(claim.owner().snapshot().delivery().abandonment())
                .thenReturn(CancelReason.CLIENT_CANCELLED);
        when(claim.item().prefillEp())
                .thenReturn(mock(org.flexlb.balance.endpoint.PrefillEndpoint.class));
        var scheduler = claim.owner().data().getScheduler();
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
        var client = runtime.engineClient;
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
                    dispatcher.stop(claim).get(2, TimeUnit.SECONDS));
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
        var claim = claim(runtime);
        when(claim.owner().snapshot().delivery().send())
                .thenReturn(ContextResult.SendState.DELIVERED);
        when(claim.owner().snapshot().cancellation()).thenReturn(null);
        when(claim.owner().snapshot().delivery().abandonment())
                .thenReturn(CancelReason.CLIENT_CANCELLED);
        when(claim.item().prefillEp())
                .thenReturn(mock(org.flexlb.balance.endpoint.PrefillEndpoint.class));
        var client = runtime.engineClient;
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
            var preemption = dispatcher.preempt(registration, null, "priority preemption", 25L);
            var stop = dispatcher.stop(claim);
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
        var claim = claim(runtime);
        when(claim.owner().snapshot().delivery()).thenReturn(null);
        var scheduler = claim.owner().data().getScheduler();
        var rejectedFact = new IllegalStateException("transient fact application failure");
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
        var client = runtime.engineClient;
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
                            .preempt(registration, null, "priority preemption", 1000L)
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
        var claim = claim(runtime);
        if (!hasDelivery) {
            when(claim.owner().snapshot().delivery()).thenReturn(null);
        }
        try {
            var client = runtime.engineClient;
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
            var answer = dispatcher.preempt(registration, null, "priority preemption", 25L);
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
            var claim = claim(runtime);
            runtime.signals().completeDelivery(claim);
            org.springframework.test.util.ReflectionTestUtils.setField(
                    runtime,
                    "stopWorkers",
                    (java.util.concurrent.Executor)
                            work -> {
                                throw new java.util.concurrent.RejectedExecutionException();
                            });
            when(claim.owner().snapshot().delivery().send())
                    .thenReturn(ContextResult.SendState.NOT_SENT);
            var cleanup =
                    new DeliveryDispatcher(
                            null,
                            DeliveryTestSupport.service(SchedulingTestConfig.batchConfig()),
                            new DeliveryMetricsReporter(null));
            assertDoesNotThrow(() -> cleanup.stop(claim).join());
            assertDoesNotThrow(() -> cleanup.stop(claim).join());
        } finally {
            runtime.shutdown();
        }
    }

    @Test
    void physicalSettlementWinningExecutorRejectionDoesNotRecordFalseFailure() {
        var runtime = runtime();
        var original = stopWorkers(runtime);
        try {
            var claim = claim(runtime);
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
                                    .stop(claim)
                                    .join());
            assertTrue(runtime.signals().delivery(claim).toCompletableFuture().isDone());
        } finally {
            org.springframework.test.util.ReflectionTestUtils.setField(
                    runtime, "stopWorkers", original);
            // Real shutdown rethrows any recorded facility failure, so this checks the entire
            // lifecycle.
            assertThrows(RuntimeException.class, runtime::shutdown);
        }
    }

    @Test
    void stopDeduplicatesTheExactClaimAndRejectsAnEqualValuedClaim() throws Exception {
        var runtime = runtime();
        try {
            var channel = runtime.engineClient;
            var claim = claim(runtime);
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
            var first = cleanup.stop(claim);
            assertSame(first, cleanup.stop(claim));
            assertEquals(
                    DeliveryDispatcher.StopResult.STALE,
                    cleanup.stop(equalValue).get(2, TimeUnit.SECONDS));
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
        var claim = claim(runtime);
        try {
            org.springframework.test.util.ReflectionTestUtils.setField(
                    runtime,
                    "stopWorkers",
                    (java.util.concurrent.Executor)
                            work -> {
                                throw new java.util.concurrent.RejectedExecutionException();
                            });
            when(claim.owner().snapshot().delivery().send())
                    .thenReturn(ContextResult.SendState.NOT_SENT);
            var cleanup =
                    new DeliveryDispatcher(
                            null,
                            DeliveryTestSupport.service(SchedulingTestConfig.batchConfig()),
                            new DeliveryMetricsReporter(null));
            var rejected = cleanup.stop(claim);
            assertThrows(java.util.concurrent.CompletionException.class, rejected::join);
            assertFalse(runtime.signals().delivery(claim).toCompletableFuture().isDone());
            runtime.signals().completeDelivery(claim);
            assertDoesNotThrow(() -> runtime.signals().delivery(claim).toCompletableFuture().join());
        } finally {
            assertThrows(RuntimeException.class, runtime::shutdown);
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
            var claim = claim(runtime);
            var client = runtime.engineClient;
            var cleanup =
                    new DeliveryDispatcher(
                            null,
                            DeliveryTestSupport.service(SchedulingTestConfig.batchConfig()),
                            new DeliveryMetricsReporter(null));
            cleanup.stop(claim);
            runtime.signals().completeDelivery(claim);
            var registration =
                    new PreemptionRegistration(
                            claim.owner(),
                            1L,
                            new org.flexlb.balance.endpoint.DecodeResources.ReservationHandle(
                                    1L, 41L, 1L),
                            new org.flexlb.balance.preemption.CancelTarget("127.0.0.1", 9000));
            var answer = cleanup.preempt(registration, null, "priority preemption", 500L);
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

    private static java.util.concurrent.Executor stopWorkers(SchedulerRuntime runtime) {
        return (java.util.concurrent.Executor)
                org.springframework.test.util.ReflectionTestUtils.getField(runtime, "stopWorkers");
    }

    private static DeliveryClaim claim(SchedulerRuntime runtime) {
        var context = mock(RequestContext.class, RETURNS_DEEP_STUBS);
        org.springframework.test.util.ReflectionTestUtils.setField(
                context.data().getScheduler(), "runtime", runtime);
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
