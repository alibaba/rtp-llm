package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.engine.grpc.EngineGrpcClient;
import org.flexlb.service.RecentCacheKeyTraceReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.params.provider.EnumSource;
import org.springframework.test.util.ReflectionTestUtils;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.argThat;
import static org.mockito.Mockito.*;

class DeliveryClaimTest {
    @Test @org.junit.jupiter.api.Timeout(15)
    void alreadyCompletedCancelReplySettlesOutsideTaskAndContextMonitors() throws Exception {
        try (var f = new Fixture()) {
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.onDeliveryEvidence(f.claim, new ContextResult.TransportResult(f.claim, DeliveryResult.delivered(), System.currentTimeMillis()));
            SchedulerTestSupport.awaitRequestWork(f.owner.runtime);
            when(f.channel.cancelAsync(any(), anyInt(), any(), anyLong()))
                    .thenReturn(CompletableFuture.completedFuture(EngineGrpcClient.CancelReply.REQUEST_CLEANED));
            f.owner.cancel(41L, 0L, CancelReason.CLIENT_CANCELLED);

            var observed = f.settlement().thenRun(() -> {
                assertFalse(Thread.holdsLock(f.context));
                f.owner.cancel(41L, 0L, CancelReason.CLIENT_CANCELLED);
            });
            f.driver();
            observed.get(5, TimeUnit.SECONDS);
            f.awaitArchive();
            SchedulerTestSupport.awaitRequestWork(f.owner.runtime);
            verify(f.prefill, times(1)).release(null, f.claim.item());
            verify(f.decode).release(f.reservation, DecodeResources.ReleaseReason.REMOTE_CLEANUP);
        }
    }

    @Test void decodeProofAndTerminalEffectsReleaseEachResourceOnce() throws Exception {
        try (var f = new Fixture()) {
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.onDeliveryEvidence(f.claim, new ContextResult.TransportResult(f.claim, DeliveryResult.delivered(), System.currentTimeMillis()));
            SchedulerTestSupport.awaitRequestWork(f.owner.runtime);
            f.applyDecodeTerminal();
            SchedulerTestSupport.awaitRequestWork(f.owner.runtime);

            // Concurrent status and terminal effects still release each exact resource once.
            assertTrue(f.settlement().isDone());
            assertFalse(f.owner.requests.isCurrent(f.context));
            assertFalse(f.prefillOwned.get());
            assertFalse(f.decodeOwned.get());
            verify(f.prefill, times(1)).release(null, f.claim.item());
            verify(f.decode, never()).release(any(), any());
        }
    }

    @Test void terminalBeforeSenderFinishesQueuesCleanupOnlyAfterProofAndShutdownDrainsIt() throws Exception {
        try (var f = new Fixture()) {
            assertTrue(f.owner.tryStartSend(f.claim));
            var executor = observeWorkers(f.owner.runtime);

            f.applyDecodeTerminal();
            SchedulerTestSupport.awaitRequestWork(f.owner.runtime);
            verify(executor, times(1)).execute(any(Runnable.class));
            assertFalse(f.settlement().isDone());
            assertTrue(f.owner.requests.isCurrent(f.context));
            assertTrue(f.prefillOwned.get(), "the sender still pins Prefill ownership");
            verify(f.prefill, never()).release(isNull(), any());

            f.owner.runtime.requests().stopAccepting();
            f.owner.onDeliveryEvidence(f.claim, new ContextResult.TransportResult(f.claim, DeliveryResult.delivered(), System.currentTimeMillis()));
            f.owner.runtime.shutdown();

            assertTrue(f.settlement().isDone());
            assertFalse(f.owner.requests.isCurrent(f.context));
            assertFalse(f.prefillOwned.get());
            assertFalse(f.decodeOwned.get());
            verify(f.prefill, times(1)).release(null, f.claim.item());
            verify(f.decode, never()).release(any(), any());
        }
    }

    @Test @org.junit.jupiter.api.Timeout(15)
    void repeatedSuccessfulTerminalBeforeSenderExitDoesNotStartRemoteCancellation() throws Exception {
        try (var f = new Fixture()) {
            assertTrue(f.owner.tryStartSend(f.claim));
            boolean senderFinished = false;
            try {
                f.applyDecodeTerminal();
                SchedulerTestSupport.awaitRequestWork(f.owner.runtime);
                assertTrue(f.context.data().getFuture().get(2, TimeUnit.SECONDS).isSuccess());

                // A repeated Worker terminal resumes cleanup while the original sender still owns the RPC.
                f.applyDecodeTerminal();
                SchedulerTestSupport.awaitRequestWork(f.owner.runtime);
                verify(f.channel, never()).cancelAsync(any(), anyInt(), any(), anyLong());
                assertNull(f.context.snapshot().delivery().abandonment(),
                        "successful execution must not acquire a remote-cancellation obligation");
                assertFalse(f.settlement().isDone(), "the original sender must still exit");
                assertTrue(f.prefillOwned.get());

                f.owner.onDeliveryEvidence(f.claim, new ContextResult.TransportResult(f.claim, DeliveryResult.delivered(), System.currentTimeMillis()));
                senderFinished = true;
                f.settlement().get(2, TimeUnit.SECONDS);
                f.awaitArchive();
                SchedulerTestSupport.awaitRequestWork(f.owner.runtime);

                assertFalse(f.prefillOwned.get());
                assertFalse(f.decodeOwned.get());
                assertTrue(f.context.data().getFuture().join().isSuccess());
                verify(f.prefill, times(1)).release(null, f.claim.item());
                verify(f.decode, never()).release(any(), any());
                verify(f.channel, never()).cancelAsync(any(), anyInt(), any(), anyLong());
            } finally {
                // Keep a failing regression from leaving an in-flight sender or cleanup task behind.
                if (!senderFinished) { f.owner.onDeliveryEvidence(f.claim, new ContextResult.TransportResult(f.claim, DeliveryResult.delivered(), System.currentTimeMillis())); }
                if (!f.settlement().isDone()) {
                    f.owner.onDeliveryEvidence(f.claim, new ContextResult.CleanupAcknowledgement(f.claim, ContextResult.CancelEvidenceKind.CLEANED, System.currentTimeMillis()));
                }
                SchedulerTestSupport.awaitRequestWork(f.owner.runtime);
            }
        }
    }

    @Test void rejectedWorkerSubmissionRetainsTheCleanupOwnerAndRecoversOffCaller() throws Exception {
        try (var f = new Fixture()) {
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.onDeliveryEvidence(f.claim, new ContextResult.TransportResult(f.claim, DeliveryResult.delivered(), System.currentTimeMillis()));
            SchedulerTestSupport.awaitRequestWork(f.owner.runtime);
            f.clearDecodeStateOwnership();
            f.recordDecodeStatus(f.decode, DecodeResources.DecodeRequestStatus.terminal(f.reservation, 0L));
            Runnable terminal = f.prepareTerminalEffects();
            f.owner.requestCleanup(f.claim.owner());
            assertNotNull(terminal);
            var executor = observeWorkers(f.owner.runtime);
            var rejected = new IllegalStateException("queue acceptance rejected");
            doAnswer(call -> {
                assertFalse(Thread.holdsLock(f.context), "queue submission must run outside the request monitor");
                f.owner.requestCleanup(f.context);
                f.owner.requestCleanup(f.context);
                assertTrue(f.prefillOwned.get(), "a synchronous event cannot consume an unaccepted queue claim");
                throw rejected;
            }).when(executor).execute(any(Runnable.class));

            f.owner.requestCleanup(f.context);
            SchedulerTestSupport.awaitRequestWork(f.owner.runtime);
            assertFalse(f.prefillOwned.get(), "accepted cleanup recovers without losing its queue claim");
            assertTrue(f.owner.requests.isCurrent(f.context), "terminal effects still own the archive gate");
            terminal.run();
            SchedulerTestSupport.awaitRequestWork(f.owner.runtime);
            assertFalse(f.owner.requests.isCurrent(f.context));
            verify(f.prefill, times(1)).release(null, f.claim.item());
            verify(f.decode, never()).release(any(), any());
        }
    }

    @Test void runningCleanupStillAcceptsANotificationAndPreventsEarlyShutdown() throws Exception {
        try (var f = new Fixture()) {
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.onDeliveryEvidence(f.claim, new ContextResult.TransportResult(f.claim, DeliveryResult.delivered(), System.currentTimeMillis()));
            SchedulerTestSupport.awaitRequestWork(f.owner.runtime);
            var executor = observeWorkers(f.owner.runtime);
            var entered = new java.util.concurrent.CountDownLatch(1);
            var release = new java.util.concurrent.CountDownLatch(1);
            doAnswer(call -> {
                assertFalse(Thread.holdsLock(f.context));
                entered.countDown();
                RequestProtocolTestSupport.await(release);
                assertTrue(f.prefillOwned.compareAndSet(true, false));
                return true;
            }).when(f.prefill).release(null, f.claim.item());
            var shutdownFailure = new AtomicReference<Throwable>();
            Thread shutdown = new Thread(() -> {
                try { f.owner.runtime.shutdown(); }
                catch (Throwable failure) { shutdownFailure.set(failure); }
            });
            try {
                f.applyDecodeTerminal();
                assertTrue(entered.await(2, TimeUnit.SECONDS));
                f.owner.requestCleanup(f.context);
                int submissions = mockingDetails(executor).getInvocations().size();
                for (int i = 0; i < 100; i++) { f.owner.requestCleanup(f.context); }
                assertEquals(submissions, mockingDetails(executor).getInvocations().size(),
                        "running cleanup merges notifications without queuing more tasks");
                shutdown.start();
                RequestProtocolTestSupport.awaitCondition(() -> shutdown.getState() == Thread.State.WAITING
                        || shutdown.getState() == Thread.State.TIMED_WAITING);
                assertTrue(shutdown.isAlive());
                assertTrue(f.owner.requests.isCurrent(f.context));
                assertTrue(f.prefillOwned.get());
                assertFalse(f.decodeOwned.get(), "the real Decode terminal has already removed its owner");
                assertFalse(f.settlement().isDone(), "the running Prefill release still owns settlement");
            } finally {
                release.countDown();
                shutdown.join(2_000);
            }
            assertFalse(shutdown.isAlive());
            assertNull(shutdownFailure.get());
            assertFalse(f.owner.requests.isCurrent(f.context));
            f.settlement().get(2, TimeUnit.SECONDS);
            assertFalse(f.prefillOwned.get());
            assertFalse(f.decodeOwned.get());
            verify(f.prefill, times(1)).release(null, f.claim.item());
            verify(f.decode, never()).release(any(), any());
        }
    }

    @Test @org.junit.jupiter.api.Timeout(15)
    void settlementAndShutdownSeeCleanupRegisteredBeforeItsSubmission() throws Exception {
        try (var f = new Fixture()) {
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.onDeliveryEvidence(f.claim, new ContextResult.TransportResult(f.claim, DeliveryResult.delivered(), System.currentTimeMillis()));
            SchedulerTestSupport.awaitRequestWork(f.owner.runtime);
            f.clearDecodeStateOwnership();
            Runnable terminal = f.prepareTerminalEffects();
            assertNotNull(terminal);
            var delegate = SchedulerTestSupport.requestWorkers(f.owner.runtime);
            var workers = observeWorkers(f.owner.runtime);
            var entered = new java.util.concurrent.CountDownLatch(1);
            var release = new java.util.concurrent.CountDownLatch(1);
            doAnswer(call -> {
                assertFalse(Thread.holdsLock(f.context));
                entered.countDown();
                RequestProtocolTestSupport.await(release);
                delegate.execute(call.getArgument(0));
                return null;
            }).when(workers).execute(any(Runnable.class));
            var producer = new Thread(() -> f.owner.requestCleanup(f.context));
            var closer = new Thread(() -> SchedulerTestSupport.closeRequestWork(f.owner.runtime));
            try {
                producer.start();
                assertTrue(entered.await(5, TimeUnit.SECONDS));
                f.owner.onDeliveryEvidence(f.claim, new ContextResult.TransportResult(f.claim,
                        DeliveryResult.delivered(), System.currentTimeMillis()));
                assertFalse(f.settlement().isDone(), "real cleanup has not reached the worker yet");
                assertTrue(f.prefillOwned.get());
                assertFalse(f.decodeOwned.get(), "DecodeState already supplied exact terminal settlement");
                terminal.run();
                closer.start();
                RequestProtocolTestSupport.awaitCondition(() -> closer.getState() == Thread.State.WAITING);
                assertTrue(closer.isAlive(), "pre-registered cleanup blocks executor close before execute returns");
                assertTrue(f.owner.requests.isCurrent(f.context));
            } finally {
                release.countDown();
                producer.join(5_000);
                closer.join(5_000);
            }
            assertFalse(closer.isAlive());
            assertFalse(f.owner.requests.isCurrent(f.context));
            f.settlement().get(2, TimeUnit.SECONDS);
            assertFalse(f.prefillOwned.get());
            assertFalse(f.decodeOwned.get());
            verify(f.prefill, times(1)).release(null, f.claim.item());
        }
    }

    private static java.util.concurrent.ExecutorService observeWorkers(SchedulerRuntime runtime) {
        var original = SchedulerTestSupport.requestWorkers(runtime);
        var observed = mock(java.util.concurrent.ExecutorService.class,
                org.mockito.AdditionalAnswers.delegatesTo(original));
        ReflectionTestUtils.setField(runtime, "requestWorkers", observed);
        return observed;
    }

    @Test @org.junit.jupiter.api.Timeout(15)
    void acceptedStatusBlocksCloseBeforeItsEffectIsSubmitted() throws Exception {
        try (var f = new Fixture()) {
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.onDeliveryEvidence(f.claim, new ContextResult.TransportResult(f.claim, DeliveryResult.delivered(), System.currentTimeMillis()));
            SchedulerTestSupport.awaitRequestWork(f.owner.runtime);
            var delegate = SchedulerTestSupport.requestWorkers(f.owner.runtime);
            var workers = observeWorkers(f.owner.runtime);
            var entered = new java.util.concurrent.CountDownLatch(1);
            var release = new java.util.concurrent.CountDownLatch(1);
            doAnswer(call -> {
                assertFalse(Thread.holdsLock(f.context));
                entered.countDown();
                RequestProtocolTestSupport.await(release);
                delegate.execute(call.getArgument(0));
                return null;
            }).when(workers).execute(any(Runnable.class));
            var source = new Thread(f::applyDecodeTerminal);
            var closer = new Thread(() -> SchedulerTestSupport.closeRequestWork(f.owner.runtime));
            try {
                source.start();
                assertTrue(entered.await(5, TimeUnit.SECONDS));
                synchronized (f.context) { assertNotNull(f.context.snapshot().pendingEnd()); }
                closer.start();
                RequestProtocolTestSupport.awaitCondition(() -> closer.getState() == Thread.State.WAITING);
                assertTrue(closer.isAlive(), "the terminal action has an owner before worker submission");
            } finally {
                release.countDown();
                source.join(5_000);
                closer.join(5_000);
            }
            assertFalse(closer.isAlive());
            assertTrue(f.settlement().isDone());
            assertFalse(f.owner.requests.isCurrent(f.context));
            verify(f.prefill, times(1)).release(null, f.claim.item());
        }
    }

    @Test void cancellationBeforeSendProducesNoRemoteCancel() throws Exception {
        try (var f = new Fixture()) {
            f.owner.cancel(41L, 0L, CancelReason.CLIENT_CANCELLED);
            assertFalse(f.owner.tryStartSend(f.claim));
            assertFalse(f.settlement().isDone(), "the sender still owns the prepared delivery");
            verifyNoInteractions(f.channel);
            f.owner.onDeliveryEvidence(f.claim, new ContextResult.TransportResult(f.claim, DeliveryResult.notSent(new IllegalStateException("cancelled before send")), System.currentTimeMillis()));
            f.settlement().get(2, TimeUnit.SECONDS);
            f.awaitArchive();
        }
    }

    @Test void earlyRemoteCleanupCannotReleaseTheSenderObligation() throws Exception {
        try (var f = new Fixture()) {
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.cancel(41L, 0L, CancelReason.CLIENT_CANCELLED);
            verify(f.channel, timeout(2_000)).cancelAsync(any(), anyInt(), eq(CancelReason.CLIENT_CANCELLED.toRequest(41L)), anyLong());
            f.cancel.complete(EngineGrpcClient.CancelReply.REQUEST_CLEANED);
            assertFalse(f.settlement().isDone());
            assertTrue(f.owner.requests.isCurrent(f.context));
            SchedulerTestSupport.runtime(f.owner).requests().stopAccepting();

            f.owner.onDeliveryEvidence(f.claim, new ContextResult.TransportResult(f.claim, DeliveryResult.delivered(), System.currentTimeMillis()));
            f.settlement().get(2, TimeUnit.SECONDS);
            f.awaitArchive();
            assertFalse(f.context.data().getFuture().join().isSuccess(), "late ACK cannot replace cancellation");
            SchedulerTestSupport.runtime(f.owner).shutdown();

        }
    }

    @Test void prefillFenceWaitsForExactDecodeSettlement() throws Exception {
        try (var f = new Fixture()) {
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.onDeliveryEvidence(f.claim, new ContextResult.TransportResult(f.claim, DeliveryResult.uncertain(new IllegalStateException("reply lost")), System.currentTimeMillis()));
            f.owner.cancel(f.context, 0L, CancelReason.CLIENT_CANCELLED);
            f.cancel.complete(EngineGrpcClient.CancelReply.REQUEST_FENCED);
            assertFalse(f.settlement().isDone());
            f.recordDecodeStatus(RequestProtocolTestSupport.decodeEndpoint(), DecodeResources.DecodeRequestStatus.terminal(f.reservation, 0L));
            f.owner.requestCleanup(f.claim.owner());
            assertFalse(f.settlement().isDone(), "another endpoint is not proof");
            f.recordDecodeStatus(f.decode, DecodeResources.DecodeRequestStatus.terminal(
                    new DecodeResources.ReservationHandle(1L, 41L, 99L), 0L));
            f.owner.requestCleanup(f.claim.owner());
            assertFalse(f.settlement().isDone(), "another reservation is not proof");
            f.applyDecodeTerminal();
            f.settlement().get(2, TimeUnit.SECONDS);
        }
    }

    @Test void failedResponseDeliveryAfterSelectedSuccessStillCancelsEngine() throws Exception {
        try (var f = new Fixture()) {
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.onDeliveryEvidence(f.claim, new ContextResult.TransportResult(f.claim, DeliveryResult.delivered(), System.currentTimeMillis()));
            assertTrue(f.context.data().getFuture().get(2, TimeUnit.SECONDS).isSuccess());
            f.owner.cancel(f.context, 0L, CancelReason.CLIENT_CANCELLED);
            verify(f.channel, timeout(2_000)).cancelAsync(any(), anyInt(), eq(CancelReason.CLIENT_CANCELLED.toRequest(41L)), anyLong());
            assertTrue(f.context.data().getFuture().join().isSuccess(), "selected result remains immutable");
            assertFalse(f.settlement().isDone());
            f.cancel.complete(EngineGrpcClient.CancelReply.REQUEST_CLEANED);
            f.settlement().get(2, TimeUnit.SECONDS);
            f.awaitArchive();
        }
    }

    @Test void rejectedRequestWorkerRecoversWithoutAnotherStopSubmission() throws Exception {
        try (var f = new Fixture()) {
            assertTrue(f.owner.tryStartSend(f.claim));
            SchedulerTestSupport.requestWorkers(f.owner.runtime).shutdown();
            f.owner.cancel(41L, 0L, CancelReason.CLIENT_CANCELLED);
            f.owner.onDeliveryEvidence(f.claim, new ContextResult.TransportResult(f.claim, DeliveryResult.delivered(), System.currentTimeMillis()));
            assertFalse(f.settlement().isDone(), "executor failure is not remote cleanup proof");
            assertTrue(f.owner.requests.isCurrent(f.context));
            verify(f.channel, timeout(2000)).cancelAsync(any(), anyInt(), any(), anyLong());
            f.cancel.complete(EngineGrpcClient.CancelReply.REQUEST_CLEANED);
            f.assertArchivedAfterCleanup();
            assertNull(SchedulerTestSupport.failure(f.owner));
        }
    }

    @Test void cleanupTimeoutWithoutProofKeepsTheResourceObligationPending() throws Exception {
        try (var f = new Fixture()) {
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.cancel(41L, 0L, CancelReason.CLIENT_CANCELLED);
            f.owner.onDeliveryEvidence(f.claim, new ContextResult.TransportResult(f.claim, DeliveryResult.delivered(), System.currentTimeMillis()));

            assertEquals(DeliveryCleanup.Result.UNRESOLVED, f.driver().get(6, TimeUnit.SECONDS));
            SchedulerTestSupport.awaitRequestWork(f.owner.runtime);

            assertAll(
                    () -> assertFalse(f.settlement().isDone(), "cleanup execution failure is not resource settlement"),
                    () -> assertTrue(f.owner.requests.isCurrent(f.context)),
                    () -> assertEquals(ContextResult.ClaimKind.BLOCKED, f.context.beginSettlement().kind()),
                    () -> verify(f.prefill, never()).release(isNull(), any()),
                    () -> verify(f.decode, never()).release(any(), any()));
        }
    }

    @Test void activeDecodeHeartbeatCannotPublishSuccessWhileNotSentEffectsArePending() throws Exception {
        try (var f = new Fixture()) {
            var original = SchedulerTestSupport.requestWorkers(f.owner.runtime);
            var held = mock(java.util.concurrent.ExecutorService.class);
            var accepted = new java.util.ArrayList<Runnable>();
            var completions = SchedulerTestSupport.responseCompletions(f.owner.runtime);
            var originalReplies = ReflectionTestUtils.getField(completions, "completionWorkers");
            var heldReplies = mock(java.util.concurrent.ThreadPoolExecutor.class);
            org.mockito.Mockito.doAnswer(call -> {
                accepted.add(call.getArgument(0));
                return null;
            }).when(held).execute(any(Runnable.class));
            doAnswer(call -> { accepted.add(call.getArgument(0)); return null; }).when(heldReplies).execute(any(Runnable.class));
            ReflectionTestUtils.setField(f.owner.runtime, "requestWorkers", held);
            ReflectionTestUtils.setField(completions, "completionWorkers", heldReplies);
            try {
                f.owner.onDeliveryEvidence(f.claim, new ContextResult.TransportResult(f.claim, DeliveryResult.notSent(new IllegalStateException("not sent")), System.currentTimeMillis()));
                assertFalse(accepted.isEmpty(), "failure effects are deliberately paused");
                f.owner.onEngineEvidence(f.context, f.decodeEvidence(f.decode,
                        DecodeResources.DecodeRequestStatus.active(f.reservation)));
                assertFalse(f.context.data().getFuture().isDone(), "ACTIVE cannot replace the selected send failure");
                assertFalse(f.settlement().isDone(), "NOT_SENT permits release but does not execute it");
                assertTrue(f.prefillOwned.get());
                assertTrue(f.decodeOwned.get());
            } finally {
                ReflectionTestUtils.setField(f.owner.runtime, "requestWorkers", original);
                ReflectionTestUtils.setField(completions, "completionWorkers", originalReplies);
                for (Runnable work : accepted) { work.run(); }
            }
            f.awaitArchive();
            f.settlement().get(2, TimeUnit.SECONDS);
            assertFalse(f.prefillOwned.get());
            assertFalse(f.decodeOwned.get());
            assertFalse(f.context.data().getFuture().get(2, TimeUnit.SECONDS).isSuccess());
        }
    }

    @Test void notSentQueuesCleanupBeforePublishingTheSettlementBarrier() throws Exception {
        try (var f = new Fixture()) {
            var original = SchedulerTestSupport.requestWorkers(f.owner.runtime);
            var held = mock(java.util.concurrent.ExecutorService.class);
            var accepted = new java.util.ArrayList<Runnable>();
            org.mockito.Mockito.doAnswer(call -> {
                assertFalse(f.settlement().isDone(), "cleanup must be accepted before the settlement barrier opens");
                accepted.add(call.getArgument(0));
                return null;
            }).when(held).execute(any(Runnable.class));
            ReflectionTestUtils.setField(f.owner.runtime, "requestWorkers", held);
            try {
                f.owner.onDeliveryEvidence(f.claim, new ContextResult.TransportResult(f.claim, DeliveryResult.notSent(new IllegalStateException("not sent")), System.currentTimeMillis()));
                assertFalse(accepted.isEmpty(), "the failure must have a cleanup owner");
                assertFalse(f.settlement().isDone(), "the held worker has not released either owner");
                assertTrue(f.prefillOwned.get());
                assertTrue(f.decodeOwned.get());
            } finally {
                ReflectionTestUtils.setField(f.owner.runtime, "requestWorkers", original);
                for (Runnable work : accepted) { work.run(); }
            }
            f.awaitArchive();
            f.settlement().get(2, TimeUnit.SECONDS);
            assertFalse(f.context.data().getFuture().get(2, TimeUnit.SECONDS).isSuccess());
            assertNull(f.context.snapshot().delivery().abandonment(), "NOT_SENT needs no remote abandonment protocol");
            assertFalse(f.prefillOwned.get());
            assertFalse(f.decodeOwned.get());
            verify(f.decode).release(f.reservation, DecodeResources.ReleaseReason.NOT_SENT);
            verify(f.channel, never()).cancelAsync(any(), anyInt(), any(), anyLong());
        }
    }

    @ParameterizedTest @org.junit.jupiter.params.provider.ValueSource(booleans = {false, true})
    @org.junit.jupiter.api.Timeout(15)
    void failedReleaseRecoversWithOrWithoutConcurrentCleanupNotification(boolean repeatNotification) throws Exception {
        try (var f = new Fixture()) {
            var entered = new java.util.concurrent.CountDownLatch(1);
            var release = new java.util.concurrent.CountDownLatch(1);
            var calls = new java.util.concurrent.atomic.AtomicInteger();
            doAnswer(call -> {
                assertFalse(Thread.holdsLock(f.context));
                if (calls.incrementAndGet() == 1) {
                    entered.countDown();
                    RequestProtocolTestSupport.await(release);
                    throw new IllegalStateException("temporary release failure");
                }
                assertTrue(f.prefillOwned.compareAndSet(true, false));
                return true;
            }).when(f.prefill).release(null, f.claim.item());
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.cancel(41L, 0L, CancelReason.CLIENT_CANCELLED);
            f.owner.onDeliveryEvidence(f.claim, new ContextResult.TransportResult(f.claim, DeliveryResult.uncertain(new IllegalStateException("reply lost")), System.currentTimeMillis()));
            SchedulerTestSupport.awaitRequestWork(f.owner.runtime);
            try {
                f.cancel.complete(EngineGrpcClient.CancelReply.REQUEST_CLEANED);
                assertTrue(entered.await(5, TimeUnit.SECONDS));
                if (repeatNotification) {
                    f.owner.onDeliveryEvidence(f.claim, new ContextResult.CleanupAcknowledgement(f.claim, ContextResult.CancelEvidenceKind.CLEANED, System.currentTimeMillis()));
                }
                assertTrue(f.owner.requests.isCurrent(f.context));
            } finally { release.countDown(); }
            f.settlement().get(2, TimeUnit.SECONDS);
            f.awaitArchive();
            SchedulerTestSupport.awaitRequestWork(f.owner.runtime);
            assertFalse(f.prefillOwned.get());
            assertFalse(f.decodeOwned.get());
            verify(f.prefill, times(2)).release(null, f.claim.item());
            f.assertDecodeOwnershipReleasedOnce();
            assertEquals(2, calls.get(), "a failed release must recover even without another notification");
            assertNotNull(SchedulerTestSupport.failure(f.owner), "the first failure remains observable");
        }
    }

    @Test void lateCleanedAckAfterTimeoutArchivesOnlyTheOriginalRequest() throws Exception {
        try (var f = new Fixture()) {
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.cancel(41L, 0L, CancelReason.CLIENT_CANCELLED);
            f.owner.onDeliveryEvidence(f.claim, new ContextResult.TransportResult(f.claim, DeliveryResult.delivered(), System.currentTimeMillis()));
            Response selected = f.context.data().getFuture().get(2, TimeUnit.SECONDS);

            assertEquals(DeliveryCleanup.Result.UNRESOLVED, f.driver().get(6, TimeUnit.SECONDS));
            f.cancel.complete(EngineGrpcClient.CancelReply.REQUEST_CLEANED);
            f.assertArchivedAfterCleanup();
            assertSame(selected, f.context.data().getFuture().join(), "cleanup cannot replace the selected response");

            var terminal = f.owner.requests.findTerminal(41L);
            assertTrue(f.owner.requests.removeExactTerminal(terminal, Long.MAX_VALUE));
            var replacement = RequestProtocolTestSupport.context(f.context.data().getConfig(), 41L);
            var replacementFuture = RequestProtocolTestSupport.register(f.owner, replacement);
            try {
                f.owner.onDeliveryEvidence(f.claim, new ContextResult.CleanupAcknowledgement(f.claim, ContextResult.CancelEvidenceKind.CLEANED, System.currentTimeMillis()));
                f.applyDecodeTerminal();
                f.owner.onEngineEvidence(f.claim.item().ctx(), new ContextResult.EngineEvidence((f.claim.item().prefillEp() == f.prefill ? f.claim.item() : null), null, org.flexlb.dao.route.RoleType.PREFILL, ContextResult.EngineKind.RETIRED, 0L, "Prefill generation retired", System.currentTimeMillis()));
                SchedulerTestSupport.awaitRequestWork(f.owner.runtime);

                assertSame(replacement, f.owner.requests.findActive(41L));
                assertFalse(replacementFuture.isDone(), "old cleanup facts cannot finish a reused request identity");
                verify(f.prefill, times(1)).release(null, f.claim.item());
                f.assertDecodeOwnershipReleasedOnce();
            } finally {
                replacementFuture.completeExceptionally(new IllegalStateException("test cleanup"));
            }
        }
    }

    @Test void endpointRetirementAfterCleanupTimeoutStillArchivesTheRequest() throws Exception {
        try (var f = new Fixture()) {
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.cancel(41L, 0L, CancelReason.CLIENT_CANCELLED);
            f.owner.onDeliveryEvidence(f.claim, new ContextResult.TransportResult(f.claim, DeliveryResult.delivered(), System.currentTimeMillis()));

            assertEquals(DeliveryCleanup.Result.UNRESOLVED, f.driver().get(6, TimeUnit.SECONDS));
            f.retirePrefill();
            f.retireDecode();
            SchedulerTestSupport.awaitRequestWork(f.owner.runtime);

            assertTrue(f.context.snapshot().delivery().prefillProof() && f.context.snapshot().delivery().decodeProof(),
                    "both exact generations have retired");
            f.assertArchivedAfterCleanup();
        }
    }

    @Test void lateFenceAfterTimeoutWaitsForTheExactDecodeTerminal() throws Exception {
        try (var f = new Fixture()) {
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.cancel(41L, 0L, CancelReason.CLIENT_CANCELLED);
            f.owner.onDeliveryEvidence(f.claim, new ContextResult.TransportResult(f.claim, DeliveryResult.delivered(), System.currentTimeMillis()));

            assertEquals(DeliveryCleanup.Result.UNRESOLVED, f.driver().get(6, TimeUnit.SECONDS));
            f.cancel.complete(EngineGrpcClient.CancelReply.REQUEST_FENCED);
            f.owner.onEngineEvidence(f.context, new ContextResult.EngineEvidence((f.context.snapshot().route() != null && f.context.snapshot().route().decodeEp() == RequestProtocolTestSupport.decodeEndpoint() ? f.context.snapshot().route() : null), DecodeResources.DecodeRequestStatus.terminal(f.reservation, 0L).reservation(), org.flexlb.dao.route.RoleType.DECODE, (DecodeResources.DecodeRequestStatus.terminal(f.reservation, 0L).kind() == DecodeResources.DecodeRequestStatus.Kind.ACTIVE ? ContextResult.EngineKind.ACTIVE : DecodeResources.DecodeRequestStatus.terminal(f.reservation, 0L).errorCode() == 0L ? ContextResult.EngineKind.COMPLETED : ContextResult.EngineKind.FAILED), DecodeResources.DecodeRequestStatus.terminal(f.reservation, 0L).errorCode(), null, System.currentTimeMillis()));
            f.owner.onEngineEvidence(f.context, new ContextResult.EngineEvidence((f.context.snapshot().route() != null && f.context.snapshot().route().decodeEp() == f.decode ? f.context.snapshot().route() : null), DecodeResources.DecodeRequestStatus.terminal(
                    new DecodeResources.ReservationHandle(1L, 41L, 99L), 0L).reservation(), org.flexlb.dao.route.RoleType.DECODE, (DecodeResources.DecodeRequestStatus.terminal(
                    new DecodeResources.ReservationHandle(1L, 41L, 99L), 0L).kind() == DecodeResources.DecodeRequestStatus.Kind.ACTIVE ? ContextResult.EngineKind.ACTIVE : DecodeResources.DecodeRequestStatus.terminal(
                    new DecodeResources.ReservationHandle(1L, 41L, 99L), 0L).errorCode() == 0L ? ContextResult.EngineKind.COMPLETED : ContextResult.EngineKind.FAILED), DecodeResources.DecodeRequestStatus.terminal(
                    new DecodeResources.ReservationHandle(1L, 41L, 99L), 0L).errorCode(), null, System.currentTimeMillis()));
            SchedulerTestSupport.awaitRequestWork(f.owner.runtime);
            assertFalse(f.settlement().isDone(), "Prefill fencing and foreign Decode facts do not settle the request");
            assertTrue(f.owner.requests.isCurrent(f.context));
            verify(f.decode, never()).release(any(), any());

            f.applyDecodeTerminal();
            f.assertArchivedAfterCleanup();
        }
    }

    @Test void lateCleanedAckAfterTimeoutStillWaitsForSenderExit() throws Exception {
        try (var f = new Fixture()) {
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.cancel(41L, 0L, CancelReason.CLIENT_CANCELLED);

            assertEquals(DeliveryCleanup.Result.UNRESOLVED, f.driver().get(6, TimeUnit.SECONDS));
            f.cancel.complete(EngineGrpcClient.CancelReply.REQUEST_CLEANED);
            SchedulerTestSupport.awaitRequestWork(f.owner.runtime);
            assertFalse(f.settlement().isDone(), "remote cleanup cannot settle an active sender");
            assertTrue(f.owner.requests.isCurrent(f.context));
            verify(f.prefill, never()).release(isNull(), any());
            verify(f.decode, never()).release(any(), any());

            f.owner.onDeliveryEvidence(f.claim, new ContextResult.TransportResult(f.claim, DeliveryResult.delivered(), System.currentTimeMillis()));
            f.assertArchivedAfterCleanup();
        }
    }

    @ParameterizedTest
    @EnumSource(value = EngineGrpcClient.CancelReply.class,
            names = {"ACCEPTED", "NOT_FOUND", "UNSUPPORTED", "FAILED"})
    void acknowledgementWithoutCleanupProofCannotReleaseAnUncertainSend(EngineGrpcClient.CancelReply ack) throws Exception {
        try (var f = new Fixture()) {
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.onDeliveryEvidence(f.claim, new ContextResult.TransportResult(f.claim, DeliveryResult.uncertain(new IllegalStateException("reply lost")), System.currentTimeMillis()));
            f.owner.cancel(f.context, 0L, CancelReason.CLIENT_CANCELLED);
            f.cancel.complete(ack);
            SchedulerTestSupport.awaitRequestWork(f.owner.runtime);
            assertFalse(f.owner.tryStartSend(f.claim), "uncertain delivery must not be sent again");
            assertFalse(f.settlement().isDone());
            assertTrue(f.owner.requests.isCurrent(f.context));
            assertEquals(ContextResult.ClaimKind.BLOCKED, f.context.beginSettlement().kind());
            verify(f.prefill, never()).release(isNull(), any());
            verify(f.decode, never()).release(any(), any());

            f.owner.onDeliveryEvidence(f.claim, new ContextResult.CleanupAcknowledgement(f.claim, ContextResult.CancelEvidenceKind.CLEANED, System.currentTimeMillis()));
            f.assertArchivedAfterCleanup();
        }
    }

    @ParameterizedTest
    @CsvSource({"true,true", "true,false", "false,true", "false,false"})
    void fenceDecodeProofAndSenderExitSettleInAnyOrder(boolean decodeFirst, boolean senderFirst) throws Exception {
        try (var f = new Fixture()) {
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.cancel(41L, 0L, CancelReason.CLIENT_CANCELLED);
            if (senderFirst) { f.owner.onDeliveryEvidence(f.claim, new ContextResult.TransportResult(f.claim, DeliveryResult.uncertain(new IllegalStateException("reply lost")), System.currentTimeMillis())); }
            Runnable decodeProof = f::applyDecodeTerminal;
            if (decodeFirst) { decodeProof.run(); }
            else { f.cancel.complete(EngineGrpcClient.CancelReply.REQUEST_FENCED); }
            SchedulerTestSupport.awaitRequestWork(f.owner.runtime);
            assertFalse(f.settlement().isDone(), "one side of the cleanup proof cannot settle delivery");
            verify(f.prefill, never()).release(isNull(), any());

            if (decodeFirst) { f.cancel.complete(EngineGrpcClient.CancelReply.REQUEST_FENCED); }
            else { decodeProof.run(); }
            SchedulerTestSupport.awaitRequestWork(f.owner.runtime);
            if (!senderFirst) {
                assertFalse(f.settlement().isDone(), "complete remote proof still needs sender exit");
                f.owner.onDeliveryEvidence(f.claim, new ContextResult.TransportResult(f.claim, DeliveryResult.uncertain(new IllegalStateException("reply lost")), System.currentTimeMillis()));
            }
            f.assertArchivedAfterCleanup();
            f.owner.onDeliveryEvidence(f.claim, new ContextResult.CleanupAcknowledgement(f.claim, ContextResult.CancelEvidenceKind.CLEANED, System.currentTimeMillis()));
            decodeProof.run();
            SchedulerTestSupport.awaitRequestWork(f.owner.runtime);
            verify(f.prefill, times(1)).release(null, f.claim.item());
            f.assertDecodeOwnershipReleasedOnce();
        }
    }

    @Test void repeatedCancellationPreservesFirstReasonAndStartsOneCleanup() throws Exception {
        try (var f = new Fixture()) {
            assertTrue(f.owner.tryStartSend(f.claim));
            assertFalse(f.owner.tryStartSend(f.claim), "send qualification is consumed once");
            f.owner.cancel(41L, 0L, CancelReason.CLIENT_CANCELLED);
            f.owner.cancel(41L, 0L, CancelReason.SHUTDOWN);
            f.owner.cancel(f.context, 0L, CancelReason.CLIENT_CANCELLED);
            SchedulerTestSupport.awaitRequestWork(f.owner.runtime);
            verify(f.channel, timeout(1000).times(1)).cancelAsync(any(), anyInt(),
                    eq(CancelReason.CLIENT_CANCELLED.toRequest(41L)), anyLong());
            f.owner.onDeliveryEvidence(f.claim, new ContextResult.TransportResult(f.claim, DeliveryResult.uncertain(new IllegalStateException("reply lost")), System.currentTimeMillis()));
            f.cancel.complete(EngineGrpcClient.CancelReply.REQUEST_CLEANED);
            f.assertArchivedAfterCleanup();
            verify(f.channel, times(1)).cancelAsync(any(), anyInt(), eq(CancelReason.CLIENT_CANCELLED.toRequest(41L)), anyLong());
        }
    }

    @ParameterizedTest
    @org.junit.jupiter.params.provider.MethodSource("cleanupEventOrders")
    void everyCleanupEventPermutationRequiresAllProofAndReleasesOnce(String outcome, String order) throws Exception {
        try (var f = new Fixture()) {
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.cancel(41L, 0L, CancelReason.CLIENT_CANCELLED);
            for (int index = 0; index < order.length(); index++) {
                switch (order.charAt(index)) {
                    case 'S' -> f.owner.onDeliveryEvidence(f.claim, new ContextResult.TransportResult(f.claim, switch (outcome) {
                        case "DELIVERED" -> DeliveryResult.delivered();
                        case "REJECTED" -> DeliveryResult.prefillRejected(new IllegalStateException("rejected"));
                        case "UNKNOWN" -> DeliveryResult.uncertain(new IllegalStateException("reply lost"));
                        default -> throw new AssertionError(outcome);
                    }, System.currentTimeMillis()));
                    case 'P' -> f.cancel.complete(EngineGrpcClient.CancelReply.REQUEST_FENCED);
                    case 'D' -> f.applyDecodeTerminal();
                    default -> throw new AssertionError(order);
                }
                SchedulerTestSupport.awaitRequestWork(f.owner.runtime);
                if (index < order.length() - 1) {
                    assertFalse(f.settlement().isDone(), "partial evidence: " + order.substring(0, index + 1));
                    assertTrue(f.owner.requests.isCurrent(f.context));
                    verify(f.prefill, never()).release(isNull(), any());
                }
            }
            f.assertArchivedAfterCleanup();
            assertFalse(f.owner.tryStartSend(f.claim));
            var archived = f.context.snapshot();
            var stale = f.context.recordDeliveryResult(new ContextResult.TransportResult(f.claim, DeliveryResult.delivered(), System.currentTimeMillis()));
            assertEquals(ContextResult.UpdateKind.STALE, stale.kind());
            assertSame(archived, stale.before());
            assertSame(archived, stale.after());
            f.owner.onDeliveryEvidence(f.claim, new ContextResult.CleanupAcknowledgement(f.claim, ContextResult.CancelEvidenceKind.CLEANED, System.currentTimeMillis()));
            f.applyDecodeTerminal();
            SchedulerTestSupport.awaitRequestWork(f.owner.runtime);
            verify(f.prefill, times(1)).release(null, f.claim.item());
            f.assertDecodeOwnershipReleasedOnce();
        }
    }

    static java.util.stream.Stream<org.junit.jupiter.params.provider.Arguments> cleanupEventOrders() {
        return java.util.stream.Stream.of("DELIVERED", "REJECTED", "UNKNOWN").flatMap(outcome ->
                java.util.stream.Stream.of("SPD", "SDP", "PSD", "PDS", "DSP", "DPS")
                        .map(order -> org.junit.jupiter.params.provider.Arguments.of(outcome, order)));
    }

    @ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(strings = {"foreignEndpoint", "foreignToken", "foreignGeneration", "foreignRequest", "notTerminal"})
    void foreignDecodeEvidenceCannotSettleTheExactDelivery(String mismatch) throws Exception {
        try (var f = new Fixture()) {
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.cancel(41L, 0L, CancelReason.CLIENT_CANCELLED);
            f.owner.onDeliveryEvidence(f.claim, new ContextResult.TransportResult(f.claim, DeliveryResult.uncertain(new IllegalStateException("reply lost")), System.currentTimeMillis()));
            f.cancel.complete(EngineGrpcClient.CancelReply.REQUEST_FENCED);
            DecodeEndpoint source = mismatch.equals("foreignEndpoint") ? RequestProtocolTestSupport.decodeEndpoint() : f.decode;
            var reservation = switch (mismatch) {
                case "foreignToken" -> new DecodeResources.ReservationHandle(1L, 41L, 2L);
                case "foreignGeneration" -> new DecodeResources.ReservationHandle(2L, 41L, 1L);
                case "foreignRequest" -> new DecodeResources.ReservationHandle(1L, 42L, 1L);
                default -> f.reservation;
            };
            var requestStatus = mismatch.equals("notTerminal")
                    ? DecodeResources.DecodeRequestStatus.active(reservation)
                    : DecodeResources.DecodeRequestStatus.terminal(reservation, 0L);
            f.recordDecodeStatus(source, requestStatus);
            f.owner.requestCleanup(f.claim.owner());
            SchedulerTestSupport.awaitRequestWork(f.owner.runtime);
            assertFalse(f.settlement().isDone());
            assertEquals(ContextResult.ClaimKind.BLOCKED, f.context.beginSettlement().kind());
            assertTrue(f.decodeOwned.get(), "foreign facts cannot release the exact Decode owner");
            verify(f.prefill, never()).release(isNull(), any());
            verify(f.decode, never()).release(any(), any());
            f.applyDecodeTerminal();
            f.assertArchivedAfterCleanup();
        }
    }

    @ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(ints = {1, 2, 3, 4, 5, 6, 7, 8})
    void cleanupCanSucceedAtEveryBoundedRetryAndCancelsItsDeadline(int successfulAttempt) throws Exception {
        try (var f = new Fixture()) {
            var calls = new java.util.concurrent.atomic.AtomicInteger();
            when(f.channel.cancelAsync(any(), anyInt(), any(), anyLong())).thenAnswer(call -> {
                assertTrue(Thread.currentThread().getName().startsWith("delivery-stop-"));
                assertFalse(Thread.holdsLock(f.context));
                return CompletableFuture.completedFuture(calls.incrementAndGet() == successfulAttempt
                        ? EngineGrpcClient.CancelReply.REQUEST_CLEANED : EngineGrpcClient.CancelReply.NOT_FOUND);
            });
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.cancel(41L, 0L, CancelReason.CLIENT_CANCELLED);
            f.owner.onDeliveryEvidence(f.claim, new ContextResult.TransportResult(f.claim, DeliveryResult.uncertain(new IllegalStateException("reply lost")), System.currentTimeMillis()));
            f.assertArchivedAfterCleanup();
            assertEquals(successfulAttempt, calls.get());
            assertNull(SchedulerTestSupport.failure(f.owner));
        }
    }

    @ParameterizedTest
    @EnumSource(value = EngineGrpcClient.CancelReply.class,
            names = {"REQUEST_CLEANED", "UNSUPPORTED"})
    @org.junit.jupiter.api.Timeout(15)
    void receivedAckDoesNotRetryWhileStopWorkIsQueued(EngineGrpcClient.CancelReply ack) throws Exception {
        try (var f = new Fixture()) {
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.cancel(41L, 0L, CancelReason.CLIENT_CANCELLED);
            f.owner.onDeliveryEvidence(f.claim, new ContextResult.TransportResult(f.claim, DeliveryResult.uncertain(new IllegalStateException("reply lost")), System.currentTimeMillis()));
            SchedulerTestSupport.awaitRequestWork(f.owner.runtime);
            var original = (java.util.concurrent.ExecutorService) ReflectionTestUtils.getField(f.owner.runtime, "stopWorkers");
            var queued = new java.util.concurrent.LinkedBlockingQueue<Runnable>();
            var held = mock(java.util.concurrent.ExecutorService.class);
            doAnswer(call -> { queued.add(call.getArgument(0)); return null; }).when(held).execute(any());
            ReflectionTestUtils.setField(f.owner.runtime, "stopWorkers", held);
            CompletableFuture<DeliveryCleanup.Result> driver;
            try {
                driver = f.driver();
                f.cancel.complete(ack);
                verify(f.channel, timeout(2_000)).cancelAsync(any(), anyInt(), argThat(request -> request.getRequestId() == 41L), eq(500L));
                verify(f.channel, after(900).times(1)).cancelAsync(any(), anyInt(),
                        argThat(request -> request.getRequestId() == 41L), eq(500L));
                assertFalse(f.settlement().isDone(), "queued ACK has not supplied release proof yet");
                assertEquals(1, queued.size(), "one received ACK must enqueue one proof application");
            } finally {
                ReflectionTestUtils.setField(f.owner.runtime, "stopWorkers", original);
                Runnable task;
                while ((task = queued.poll()) != null) { task.run(); }
            }
            if (ack == EngineGrpcClient.CancelReply.UNSUPPORTED) {
                assertFalse(f.settlement().isDone(), "UNSUPPORTED is not cleanup proof");
                f.owner.onDeliveryEvidence(f.claim, new ContextResult.CleanupAcknowledgement(f.claim, ContextResult.CancelEvidenceKind.CLEANED, System.currentTimeMillis()));
            }
            f.assertArchivedAfterCleanup();
            driver.get(2, TimeUnit.SECONDS);
            verify(f.channel, times(1)).cancelAsync(any(), anyInt(), argThat(request -> request.getRequestId() == 41L), anyLong());
            assertNull(SchedulerTestSupport.failure(f.owner));
        }
    }

    @Test void timedOutAcknowledgementStillSuppliesLateCleanupProof() throws Exception {
        try (var f = new Fixture()) {
            var first = new CompletableFuture<EngineGrpcClient.CancelReply>();
            when(f.channel.cancelAsync(any(), anyInt(), any(), anyLong())).thenReturn(first, new CompletableFuture<>());
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.cancel(41L, 0L, CancelReason.CLIENT_CANCELLED);
            f.owner.onDeliveryEvidence(f.claim, new ContextResult.TransportResult(f.claim, DeliveryResult.uncertain(new IllegalStateException("reply lost")), System.currentTimeMillis()));
            verify(f.channel, timeout(2_000).times(2)).cancelAsync(any(), anyInt(),
                    argThat(request -> request.getRequestId() == 41L), eq(500L));
            assertFalse(f.settlement().isDone());
            first.complete(EngineGrpcClient.CancelReply.REQUEST_CLEANED);
            f.assertArchivedAfterCleanup();
            verify(f.channel, times(2)).cancelAsync(any(), anyInt(), argThat(request -> request.getRequestId() == 41L), eq(500L));
        }
    }

    @Test void overallTimeoutStopsAnAlreadyScheduledRetry() throws Exception {
        try (var f = new Fixture()) {
            when(f.channel.cancelAsync(any(), anyInt(), any(), anyLong())).thenAnswer(call -> new CompletableFuture<>());
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.cancel(41L, 0L, CancelReason.CLIENT_CANCELLED);
            f.owner.onDeliveryEvidence(f.claim, new ContextResult.TransportResult(f.claim, DeliveryResult.uncertain(new IllegalStateException("reply lost")), System.currentTimeMillis()));
            var driver = f.driver();
            assertEquals(DeliveryCleanup.Result.UNRESOLVED, driver.get(6, TimeUnit.SECONDS));
            int count = mockingDetails(f.channel).getInvocations().size();
            assertSame(driver, f.driver(), "a stopped driver must not restart the same unsettled claim");
            assertTrue(count <= 8);
            assertFalse(f.settlement().isDone());
            assertNotNull(SchedulerTestSupport.failure(f.owner));
        }
    }

    @Test void retryExhaustionStillAcceptsAnEarlierTimedOutAttemptProof() throws Exception {
        try (var f = new Fixture()) {
            var first = new CompletableFuture<EngineGrpcClient.CancelReply>();
            var calls = new java.util.concurrent.atomic.AtomicInteger();
            when(f.channel.cancelAsync(any(), anyInt(), any(), anyLong())).thenAnswer(call ->
                    calls.incrementAndGet() == 1 ? first : CompletableFuture.completedFuture(EngineGrpcClient.CancelReply.NOT_FOUND));
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.cancel(41L, 0L, CancelReason.CLIENT_CANCELLED);
            f.owner.onDeliveryEvidence(f.claim, new ContextResult.TransportResult(f.claim, DeliveryResult.uncertain(new IllegalStateException("reply lost")), System.currentTimeMillis()));
            assertEquals(DeliveryCleanup.Result.UNRESOLVED, f.driver().get(6, TimeUnit.SECONDS));
            assertEquals(8, calls.get());
            assertFalse(f.settlement().isDone());
            first.complete(EngineGrpcClient.CancelReply.REQUEST_CLEANED);
            f.assertArchivedAfterCleanup();
            assertEquals(8, calls.get());
        }
    }

    @ParameterizedTest
    @EnumSource(value = EngineGrpcClient.CancelReply.class,
            names = {"ACCEPTED", "NOT_FOUND", "FAILED"})
    void retryExhaustionCannotInventResourceRelease(EngineGrpcClient.CancelReply ack) throws Exception {
        try (var f = new Fixture()) {
            when(f.channel.cancelAsync(any(), anyInt(), any(), anyLong())).thenAnswer(call -> CompletableFuture.completedFuture(ack));
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.cancel(41L, 0L, CancelReason.CLIENT_CANCELLED);
            f.owner.onDeliveryEvidence(f.claim, new ContextResult.TransportResult(f.claim, DeliveryResult.uncertain(new IllegalStateException("reply lost")), System.currentTimeMillis()));
            assertEquals(DeliveryCleanup.Result.UNRESOLVED, f.driver().get(6, TimeUnit.SECONDS));
            verify(f.channel, times(8)).cancelAsync(any(), anyInt(), argThat(request -> request.getRequestId() == 41L), anyLong());
            assertTrue(f.owner.requests.isCurrent(f.context));
            assertEquals(ContextResult.ClaimKind.BLOCKED, f.context.beginSettlement().kind());
            verify(f.prefill, never()).release(isNull(), any());
            verify(f.decode, never()).release(any(), any());
        }
    }

    @ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(booleans = {false, true})
    void unsupportedCancelStopsTransportWithoutProvingRelease(boolean timeout) throws Exception {
        try (var logs = new CleanupLogs(); var f = new Fixture()) {
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.cancel(41L, 0L, CancelReason.CLIENT_CANCELLED);
            f.owner.onDeliveryEvidence(f.claim, new ContextResult.TransportResult(f.claim, DeliveryResult.uncertain(new IllegalStateException("reply lost")), System.currentTimeMillis()));
            var driver = f.driver();
            f.cancel.complete(EngineGrpcClient.CancelReply.UNSUPPORTED);
            verify(f.channel, timeout(2_000)).cancelAsync(any(), anyInt(), argThat(request -> request.getRequestId() == 41L), anyLong());
            assertEquals(DeliveryCleanup.Result.UNSUPPORTED, driver.get(2, TimeUnit.SECONDS));
            assertNull(SchedulerTestSupport.failure(f.owner));
            assertFalse(f.settlement().isDone());
            assertTrue(f.prefillOwned.get());
            assertTrue(f.decodeOwned.get());
            f.owner.onDeliveryEvidence(f.claim, new ContextResult.CleanupAcknowledgement(f.claim, ContextResult.CancelEvidenceKind.CLEANED, System.currentTimeMillis()));
            f.assertArchivedAfterCleanup();
            if (!timeout) { driver.get(2, TimeUnit.SECONDS); }
            verify(f.channel, times(1)).cancelAsync(any(), anyInt(), argThat(request -> request.getRequestId() == 41L), anyLong());
            assertEquals(1, logs.count(ch.qos.logback.classic.Level.WARN));
            assertDoesNotThrow(f.owner.runtime::shutdown);
        }
    }

    @ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(booleans = {false, true})
    void lateUnsupportedStopsQueuedOrInFlightRetryAndStillAcceptsProof(boolean retryStarted) throws Exception {
        try (var logs = new CleanupLogs(); var f = new Fixture()) {
            var first = new CompletableFuture<EngineGrpcClient.CancelReply>();
            var second = new CompletableFuture<EngineGrpcClient.CancelReply>();
            when(f.channel.cancelAsync(any(), anyInt(), any(), anyLong())).thenReturn(first, second);
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.cancel(41L, 0L, CancelReason.CLIENT_CANCELLED);
            f.owner.onDeliveryEvidence(f.claim, new ContextResult.TransportResult(f.claim, DeliveryResult.uncertain(new IllegalStateException("reply lost")), System.currentTimeMillis()));
            var driver = f.driver();
            verify(f.channel, timeout(2_000).times(retryStarted ? 2 : 1)).cancelAsync(any(), anyInt(),
                    argThat(request -> request.getRequestId() == 41L), anyLong());
            first.complete(EngineGrpcClient.CancelReply.UNSUPPORTED);
            if (retryStarted) { second.complete(EngineGrpcClient.CancelReply.REQUEST_CLEANED); }
            else { f.owner.onDeliveryEvidence(f.claim, new ContextResult.CleanupAcknowledgement(f.claim, ContextResult.CancelEvidenceKind.CLEANED, System.currentTimeMillis())); }
            f.assertArchivedAfterCleanup();
            driver.get(2, TimeUnit.SECONDS);
            verify(f.channel, times(retryStarted ? 2 : 1)).cancelAsync(any(), anyInt(),
                    argThat(request -> request.getRequestId() == 41L), anyLong());
            assertEquals(1, logs.count(ch.qos.logback.classic.Level.WARN));
            assertNull(SchedulerTestSupport.failure(f.owner));
        }
    }

    @ParameterizedTest
    @EnumSource(value = EngineGrpcClient.CancelReply.class, names = {"NOT_FOUND", "ACCEPTED", "FAILED"})
    void lateReplyWithoutProofCannotScheduleDuplicateRetry(EngineGrpcClient.CancelReply ack) throws Exception {
        try (var f = new Fixture()) {
            var first = new CompletableFuture<EngineGrpcClient.CancelReply>();
            when(f.channel.cancelAsync(any(), anyInt(), any(), anyLong())).thenReturn(first, new CompletableFuture<>());
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.cancel(41L, 0L, CancelReason.CLIENT_CANCELLED);
            f.owner.onDeliveryEvidence(f.claim, new ContextResult.TransportResult(f.claim, DeliveryResult.uncertain(new IllegalStateException("reply lost")), System.currentTimeMillis()));
            var driver = f.driver();
            verify(f.channel, timeout(2_000).times(2)).cancelAsync(any(), anyInt(),
                    argThat(request -> request.getRequestId() == 41L), anyLong());
            first.complete(ack);
            assertSame(driver, f.driver());
            assertFalse(f.settlement().isDone());
            f.owner.onDeliveryEvidence(f.claim, new ContextResult.CleanupAcknowledgement(f.claim, ContextResult.CancelEvidenceKind.CLEANED, System.currentTimeMillis()));
            f.assertArchivedAfterCleanup();
            driver.get(2, TimeUnit.SECONDS);
            verify(f.channel, times(2)).cancelAsync(any(), anyInt(), argThat(request -> request.getRequestId() == 41L), anyLong());
        }
    }

    @Test void unsentStopTimeoutRetainsResponsibilityUntilSenderExits() throws Exception {
        try (var f = new Fixture()) {
            f.owner.cancel(41L, 0L, CancelReason.CLIENT_CANCELLED);
            assertEquals(DeliveryCleanup.Result.UNRESOLVED, f.driver().get(6, TimeUnit.SECONDS));
            SchedulerTestSupport.dispatcher(f.owner.runtime).awaitStops(f.owner.runtime);
            assertFalse(f.settlement().isDone());
            verifyNoInteractions(f.channel);
            f.owner.onDeliveryEvidence(f.claim, new ContextResult.TransportResult(f.claim,
                    DeliveryResult.notSent(new IllegalStateException("cancelled before send")), System.currentTimeMillis()));
            f.settlement().get(2, TimeUnit.SECONDS);
            f.awaitArchive();
            assertFalse(f.prefillOwned.get());
            assertFalse(f.decodeOwned.get());
        }
    }

    @Test void unsentVictimCancellationClosesPreemptionAfterLocalRelease() throws Exception {
        try (var f = new Fixture()) {
            var claim = f.context.beginPreemption(f.reservation, 92L,
                    new org.flexlb.balance.preemption.CancelTarget("127.0.0.1", 9000));
            var registration = ((ContextResult.Granted<PreemptionRegistration>) claim).value();
            f.owner.cancel(41L, 0L, CancelReason.CLIENT_CANCELLED);
            f.owner.onDeliveryEvidence(f.claim, new ContextResult.TransportResult(f.claim,
                    DeliveryResult.notSent(new IllegalStateException("cancelled before send")), System.currentTimeMillis()));
            f.settlement().get(2, TimeUnit.SECONDS);
            f.awaitArchive();
            assertNotNull(f.owner.runtime.signals().resolved(registration));
            verifyNoInteractions(f.channel);
        }
    }

    @Test void senderCompletionAdvancesTheSameStopRecordAfterItsInitialWaitExpires() throws Exception {
        try (var f = new Fixture()) {
            when(f.channel.cancelAsync(any(), anyInt(), any(), anyLong())).thenReturn(
                    CompletableFuture.completedFuture(EngineGrpcClient.CancelReply.NOT_FOUND),
                    CompletableFuture.completedFuture(EngineGrpcClient.CancelReply.REQUEST_CLEANED));
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.cancel(41L, 0L, CancelReason.CLIENT_CANCELLED);
            var record = f.stopRecord();
            assertEquals(DeliveryCleanup.Result.UNRESOLVED, f.driver().get(6, TimeUnit.SECONDS));
            assertSame(record, f.stopRecord());
            assertFalse(f.settlement().isDone());
            var dispatcher = SchedulerTestSupport.dispatcher(f.owner.runtime);
            ReflectionTestUtils.invokeMethod(dispatcher, "report", 51L, java.util.List.of(f.claim),
                    java.util.List.of(DeliveryResult.delivered()));
            f.assertArchivedAfterCleanup();
            verify(f.channel, times(2)).cancelAsync(any(), anyInt(), any(), anyLong());
            assertFalse(f.context.data().getFuture().join().isSuccess());
        }
    }

    @Test void ordinaryCancellationSharesTheOutstandingPreemptionRpc() throws Exception {
        try (var f = new Fixture()) {
            assertTrue(f.owner.tryStartSend(f.claim));
            var granted = f.context.beginPreemption(f.reservation, 91L,
                    new org.flexlb.balance.preemption.CancelTarget("127.0.0.1", 9000));
            var registration = ((ContextResult.Granted<PreemptionRegistration>) granted).value();
            var preemptionReply = SchedulerTestSupport.dispatcher(f.owner.runtime).preempt(registration, 500L);
            verify(f.channel, timeout(2000)).cancelAsync(any(), anyInt(), any(), anyLong());
            f.owner.cancel(41L, 0L, CancelReason.CLIENT_CANCELLED);
            f.owner.onDeliveryEvidence(f.claim, new ContextResult.TransportResult(f.claim,
                    DeliveryResult.delivered(), System.currentTimeMillis()));
            f.cancel.complete(EngineGrpcClient.CancelReply.REQUEST_CLEANED);
            assertEquals(EngineGrpcClient.CancelReply.REQUEST_CLEANED, preemptionReply.get(2, TimeUnit.SECONDS));
            f.assertArchivedAfterCleanup();
            verify(f.channel, times(1)).cancelAsync(any(), anyInt(), any(), anyLong());
        }
    }

    @Test void cancellationQueuedBehindPreemptionInitializesItsOwnDeadlineBeforeAdvancing() throws Exception {
        try (var f = new Fixture()) {
            assertTrue(f.owner.tryStartSend(f.claim));
            var original = (java.util.concurrent.ExecutorService) ReflectionTestUtils.getField(f.owner.runtime, "stopWorkers");
            var queued = new java.util.concurrent.ConcurrentLinkedQueue<Runnable>();
            var paused = mock(java.util.concurrent.ExecutorService.class);
            doAnswer(call -> { queued.add(call.getArgument(0)); return null; }).when(paused).execute(any(Runnable.class));
            ReflectionTestUtils.setField(f.owner.runtime, "stopWorkers", paused);
            try {
                var registration = new PreemptionRegistration(f.context, 91L, f.reservation,
                        new org.flexlb.balance.preemption.CancelTarget("127.0.0.1", 9000));
                var dispatcher = SchedulerTestSupport.dispatcher(f.owner.runtime);
                dispatcher.preempt(registration, 500L);
                var record = f.stopRecord();
                f.owner.cancel(41L, 0L, CancelReason.CLIENT_CANCELLED);
                var stopping = f.driver();
                Runnable task;
                while ((task = queued.poll()) != null) { task.run(); }
                assertFalse(stopping.isDone(), "queued cancellation must not see an uninitialized deadline");
                assertNull(SchedulerTestSupport.failure(f.owner));
                assertSame(record, f.stopRecord());
            } finally {
                ReflectionTestUtils.setField(f.owner.runtime, "stopWorkers", original);
                Runnable task;
                while ((task = queued.poll()) != null) { task.run(); }
            }
            f.owner.onDeliveryEvidence(f.claim, new ContextResult.TransportResult(f.claim,
                    DeliveryResult.delivered(), System.currentTimeMillis()));
            f.cancel.complete(EngineGrpcClient.CancelReply.REQUEST_CLEANED);
            f.assertArchivedAfterCleanup();
            verify(f.channel, times(1)).cancelAsync(any(), anyInt(), any(), anyLong());
        }
    }

    @Test void nextPreemptionDoesNotReuseAnEarlierNotFoundDuringCallbackExit() throws Exception {
        try (var f = new Fixture()) {
            assertTrue(f.owner.tryStartSend(f.claim));
            var target = new org.flexlb.balance.preemption.CancelTarget("127.0.0.1", 9000);
            var first = new PreemptionRegistration(f.context, 91L, f.reservation, target);
            var next = new PreemptionRegistration(f.context, 92L, f.reservation, target);
            when(f.channel.cancelAsync(any(), anyInt(), any(), anyLong())).thenReturn(f.cancel,
                    CompletableFuture.completedFuture(EngineGrpcClient.CancelReply.REQUEST_CLEANED));
            var dispatcher = SchedulerTestSupport.dispatcher(f.owner.runtime);
            var result = dispatcher.preempt(first, 500L).thenCompose(reply -> dispatcher.preempt(next, 500L));
            var record = f.stopRecord();
            f.cancel.complete(EngineGrpcClient.CancelReply.NOT_FOUND);
            assertEquals(EngineGrpcClient.CancelReply.REQUEST_CLEANED, result.get(2, TimeUnit.SECONDS));
            assertSame(record, f.stopRecord(), "a new preemption advances the existing delivery record");
            verify(f.channel, times(2)).cancelAsync(any(), anyInt(), any(), anyLong());
            f.owner.cancel(41L, 0L, CancelReason.CLIENT_CANCELLED);
            f.owner.onDeliveryEvidence(f.claim, new ContextResult.TransportResult(f.claim,
                    DeliveryResult.delivered(), System.currentTimeMillis()));
            f.assertArchivedAfterCleanup();
        }
    }

    @Test void failedFactApplicationRetriesTheFactWithoutRepeatingTheRpc() throws Exception {
        try (var f = new Fixture()) {
            doThrow(new IllegalStateException("temporary publication failure")).doCallRealMethod()
                    .when(f.owner).onDeliveryEvidence(eq(f.claim), isA(ContextResult.CleanupAcknowledgement.class));
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.cancel(41L, 0L, CancelReason.CLIENT_CANCELLED);
            f.owner.onDeliveryEvidence(f.claim, new ContextResult.TransportResult(f.claim,
                    DeliveryResult.delivered(), System.currentTimeMillis()));
            f.cancel.complete(EngineGrpcClient.CancelReply.REQUEST_CLEANED);
            f.assertArchivedAfterCleanup();
            verify(f.channel, times(1)).cancelAsync(any(), anyInt(), any(), anyLong());
            assertNull(SchedulerTestSupport.failure(f.owner));
        }
    }

    @Test void concurrentStopEventsAreSerializedWithoutHoldingTheContextLock() throws Exception {
        try (var f = new Fixture()) {
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.cancel(41L, 0L, CancelReason.CLIENT_CANCELLED);
            var record = f.stopRecord();
            var active = new java.util.concurrent.atomic.AtomicInteger();
            var maximum = new java.util.concurrent.atomic.AtomicInteger();
            var done = new java.util.concurrent.CountDownLatch(1_000);
            try (var producers = java.util.concurrent.Executors.newFixedThreadPool(8)) {
                for (int i = 0; i < 1_000; i++) {
                    producers.submit(() -> ReflectionTestUtils.invokeMethod(record, "post", (Runnable) () -> {
                        maximum.accumulateAndGet(active.incrementAndGet(), Math::max);
                        assertFalse(Thread.holdsLock(f.context));
                        Thread.yield();
                        active.decrementAndGet();
                        done.countDown();
                    }));
                }
            }
            assertTrue(done.await(3, TimeUnit.SECONDS));
            assertEquals(1, maximum.get());
            assertSame(record, f.stopRecord());
            verify(f.channel, times(1)).cancelAsync(any(), anyInt(), any(), anyLong());
            f.owner.onDeliveryEvidence(f.claim, new ContextResult.TransportResult(f.claim,
                    DeliveryResult.delivered(), System.currentTimeMillis()));
            f.cancel.complete(EngineGrpcClient.CancelReply.REQUEST_CLEANED);
            f.assertArchivedAfterCleanup();
        }
    }

    @ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(booleans = {false, true})
    void rejectedTimerRetainsTheSameRecordAndAcceptsLateProof(boolean permanent) throws Exception {
        try (var f = new Fixture()) {
            var original = (java.util.concurrent.ScheduledThreadPoolExecutor)
                    ReflectionTestUtils.getField(f.owner.runtime, "cleanupExecutor");
            var rejecting = mock(java.util.concurrent.ScheduledThreadPoolExecutor.class);
            var submissions = new java.util.concurrent.atomic.AtomicInteger();
            doAnswer(call -> {
                if (submissions.incrementAndGet() <= 2 || permanent) {
                    throw new java.util.concurrent.RejectedExecutionException("timer unavailable");
                }
                return original.schedule(call.getArgument(0, Runnable.class), call.getArgument(1, Long.class),
                        call.getArgument(2, TimeUnit.class));
            }).when(rejecting).schedule(any(Runnable.class), anyLong(), any(TimeUnit.class));
            ReflectionTestUtils.setField(f.owner.runtime, "cleanupExecutor", rejecting);
            try {
                assertTrue(f.owner.tryStartSend(f.claim));
                f.owner.cancel(41L, 0L, CancelReason.CLIENT_CANCELLED);
                var record = f.stopRecord();
                SchedulerTestSupport.awaitRequestWork(f.owner.runtime);
                assertEquals(3, submissions.get());
                assertEquals(permanent, f.driver().isCompletedExceptionally());
                assertFalse(f.settlement().isDone());
                assertSame(record, f.stopRecord());
                ReflectionTestUtils.setField(f.owner.runtime, "cleanupExecutor", original);
                f.owner.onDeliveryEvidence(f.claim, new ContextResult.TransportResult(f.claim,
                        DeliveryResult.delivered(), System.currentTimeMillis()));
                f.cancel.complete(EngineGrpcClient.CancelReply.REQUEST_CLEANED);
                f.assertArchivedAfterCleanup();
                verify(f.channel, times(1)).cancelAsync(any(), anyInt(), any(), anyLong());
            } finally { ReflectionTestUtils.setField(f.owner.runtime, "cleanupExecutor", original); }
        }
    }

    @Test void immediateRpcReplyAndSettlementReentryDoNotHoldStopRecordMonitor() throws Exception {
        try (var f = new Fixture()) {
            when(f.channel.cancelAsync(any(), anyInt(), any(), anyLong())).thenReturn(
                    CompletableFuture.completedFuture(EngineGrpcClient.CancelReply.REQUEST_CLEANED));
            assertTrue(f.owner.tryStartSend(f.claim));
            var callbacks = new java.util.concurrent.atomic.AtomicInteger();
            doAnswer(call -> {
                assertFalse(Thread.holdsLock(f.context));
                assertFalse(Thread.holdsLock(f.stopRecord()));
                callbacks.incrementAndGet();
                return call.callRealMethod();
            }).when(f.owner).onDeliveryEvidence(eq(f.claim), isA(ContextResult.CleanupAcknowledgement.class));
            f.owner.cancel(41L, 0L, CancelReason.CLIENT_CANCELLED);
            f.owner.onDeliveryEvidence(f.claim, new ContextResult.TransportResult(f.claim,
                    DeliveryResult.delivered(), System.currentTimeMillis()));
            f.assertArchivedAfterCleanup();
            assertEquals(1, callbacks.get());
            verify(f.channel, times(1)).cancelAsync(any(), anyInt(), any(), anyLong());
        }
    }

    private static final class CleanupLogs implements AutoCloseable {
        private final ch.qos.logback.classic.Logger logger =
                (ch.qos.logback.classic.Logger) org.slf4j.LoggerFactory.getLogger("flexlbLogger");
        private final ch.qos.logback.classic.Level previousLevel = logger.getLevel();
        private final ch.qos.logback.core.read.ListAppender<ch.qos.logback.classic.spi.ILoggingEvent> appender =
                new ch.qos.logback.core.read.ListAppender<>();

        CleanupLogs() {
            logger.setLevel(ch.qos.logback.classic.Level.WARN);
            appender.start();
            logger.addAppender(appender);
        }

        long count(ch.qos.logback.classic.Level level) {
            return appender.list.stream().filter(event -> event.getLevel() == level
                    && event.getFormattedMessage().contains("[delivery-cleanup]")).count();
        }

        public void close() {
            logger.detachAppender(appender);
            appender.stop();
            logger.setLevel(previousLevel);
        }
    }

    private static final class Fixture implements AutoCloseable {
        final EngineGrpcClient channel = mock(EngineGrpcClient.class);
        final CompletableFuture<EngineGrpcClient.CancelReply> cancel = new CompletableFuture<>();
        final AbstractRequestScheduler owner;
        final RequestContext context;
        final PrefillEndpoint prefill = mock(PrefillEndpoint.class);
        final java.util.concurrent.atomic.AtomicBoolean prefillOwned = new java.util.concurrent.atomic.AtomicBoolean(true);
        final java.util.concurrent.atomic.AtomicBoolean decodeOwned = new java.util.concurrent.atomic.AtomicBoolean(true);
        final java.util.concurrent.atomic.AtomicBoolean decodeClearedByState = new java.util.concurrent.atomic.AtomicBoolean();
        final java.util.concurrent.atomic.AtomicBoolean prefillClearedByState = new java.util.concurrent.atomic.AtomicBoolean();
        final DecodeEndpoint decode = RequestProtocolTestSupport.decodeEndpoint();
        final DecodeResources.ReservationHandle reservation = new DecodeResources.ReservationHandle(1L, 41L, 1L);
        final DeliveryClaim claim;
        Fixture() {
            var config = SchedulingTestConfig.batchConfig();
            var service = mock(ConfigService.class);
            when(service.loadBalanceConfig()).thenReturn(config);
            owner = SchedulerTestSupport.create(service, mock(DeliveryMetricsReporter.class), mock(RequestSchedulerReporter.class), mock(RecentCacheKeyTraceReporter.class));
            ReflectionTestUtils.setField(owner.runtime, "engineClient", channel);
            when(channel.cancelAsync(any(), anyInt(), any(), anyLong())).thenReturn(cancel);
            context = RequestProtocolTestSupport.context(config, 41L);
            var future = RequestProtocolTestSupport.register(owner, context);
            var response = new Response();
            response.setSuccess(true);
            var item = org.flexlb.balance.scheduler.SchedulingTestConfig.createRoute(context, response, null, null,
                    prefill, decode, reservation, System.currentTimeMillis());
            when(prefill.release(isNull(), any())).thenAnswer(call -> {
                assertFalse(Thread.holdsLock(context), "resource cleanup must run outside the request monitor");
                assertSame(item, call.getArgument(1), "only this exact Prefill owner may be released");
                assertTrue(prefillOwned.compareAndSet(true, false), "Prefill ownership may be released only once");
                return true;
            });
            when(decode.release(any(), any())).thenAnswer(call -> {
                assertFalse(Thread.holdsLock(context), "Decode cleanup must run outside the request monitor");
                assertEquals(reservation, call.getArgument(0), "only this exact Decode owner may be released");
                assertTrue(decodeOwned.compareAndSet(true, false), "Decode ownership may be released only once");
                return DecodeResources.ResourceChange.of(DecodeResources.ReservationReleaseResult.RELEASED, true);
            });
            RequestProtocolTestSupport.beginRoutePublication(owner, new RequestProtocolTestSupport.Registered(item, future));
            claim = RequestProtocolTestSupport.claimBatch(owner, item, 51L, () -> true);
            assertNotNull(claim);
        }
        CompletableFuture<Void> settlement() { return claim.owner().data().getScheduler().runtime.signals().delivery(claim).toCompletableFuture(); }
        void awaitArchive() throws InterruptedException { RequestProtocolTestSupport.awaitCondition(() -> !owner.requests.isCurrent(context)); }

        // DecodeState removes exact ownership before Endpoint publishes a terminal or retirement fact.
        private void clearDecodeStateOwnership() {
            if (decodeOwned.getAndSet(false)) { decodeClearedByState.set(true); }
        }

        void recordDecodeStatus(DecodeEndpoint source, DecodeResources.DecodeRequestStatus status) {
            context.recordEngineEvidence(decodeEvidence(source, status));
        }

        ContextResult.EngineEvidence decodeEvidence(DecodeEndpoint source, DecodeResources.DecodeRequestStatus status) {
            var current = context.snapshot().route();
            var exact = current != null && current.decodeEp() == source ? current : null;
            var kind = status.kind() == DecodeResources.DecodeRequestStatus.Kind.ACTIVE ? ContextResult.EngineKind.ACTIVE
                    : status.errorCode() == 0L ? ContextResult.EngineKind.COMPLETED : ContextResult.EngineKind.FAILED;
            return new ContextResult.EngineEvidence(exact, status.reservation(), org.flexlb.dao.route.RoleType.DECODE,
                    kind, status.errorCode(), null, System.currentTimeMillis());
        }

        /** Deliberately pause the real notification owner while independently testing resource settlement. */
        Runnable prepareTerminalEffects() {
            recordDecodeStatus(decode, DecodeResources.DecodeRequestStatus.terminal(reservation, 0L));
            var closing = context.beginClose(context.snapshot().pendingEnd(), System.currentTimeMillis());
            assertInstanceOf(ContextResult.Granted.class, closing);
            var token = ((ContextResult.Granted<ContextResult.EndToken>) closing).value();
            return () -> {
                owner.runtime.deadlines().update(context, context.snapshot().deadlines());
                var completed = context.completeClose(token, new ContextResult.CloseProgress(ContextResult.ProgressKind.NOTIFICATIONS,
                        false, false, true, true), System.currentTimeMillis());
                if (completed.after().terminalRecord() != null) { owner.requests.archive(context, completed.after().terminalRecord()); }
                owner.requestCleanup(context);
            };
        }

        void applyDecodeTerminal() {
            clearDecodeStateOwnership();
            owner.onEngineEvidence(context, new ContextResult.EngineEvidence((context.snapshot().route() != null && context.snapshot().route().decodeEp() == decode ? context.snapshot().route() : null), DecodeResources.DecodeRequestStatus.terminal(reservation, 0L).reservation(), org.flexlb.dao.route.RoleType.DECODE, (DecodeResources.DecodeRequestStatus.terminal(reservation, 0L).kind() == DecodeResources.DecodeRequestStatus.Kind.ACTIVE ? ContextResult.EngineKind.ACTIVE : DecodeResources.DecodeRequestStatus.terminal(reservation, 0L).errorCode() == 0L ? ContextResult.EngineKind.COMPLETED : ContextResult.EngineKind.FAILED), DecodeResources.DecodeRequestStatus.terminal(reservation, 0L).errorCode(), null, System.currentTimeMillis()));
        }

        void retirePrefill() {
            // Real Prefill retirement drops generation ownership before publishing this exact fact.
            if (prefillOwned.getAndSet(false)) { prefillClearedByState.set(true); }
            owner.onEngineEvidence(claim.item().ctx(), new ContextResult.EngineEvidence((claim.item().prefillEp() == prefill ? claim.item() : null), null, org.flexlb.dao.route.RoleType.PREFILL, ContextResult.EngineKind.RETIRED, 0L, "Prefill generation retired", System.currentTimeMillis()));
        }

        void retireDecode() {
            clearDecodeStateOwnership();
            when(decode.isRetired()).thenReturn(true);
            owner.onEngineEvidence(context, new ContextResult.EngineEvidence((context.snapshot().route() != null && context.snapshot().route().decodeEp() == decode ? context.snapshot().route() : null), reservation, org.flexlb.dao.route.RoleType.DECODE, ContextResult.EngineKind.RETIRED, 0L, "Decode generation retired", System.currentTimeMillis()));
        }

        void assertDecodeOwnershipReleasedOnce() {
            assertFalse(decodeOwned.get(), "archival requires physical Decode ownership to be released");
            verify(decode, times(decodeClearedByState.get() ? 0 : 1))
                    .release(reservation, DecodeResources.ReleaseReason.REMOTE_CLEANUP);
        }

        CompletableFuture<DeliveryCleanup.Result> driver() {
            return SchedulerTestSupport.dispatcher(owner.runtime).stop(claim);
        }

        Object stopRecord() {
            var cleanup = ReflectionTestUtils.getField(SchedulerTestSupport.dispatcher(owner.runtime), "cleanup");
            var records = (java.util.Map<?, ?>) ReflectionTestUtils.getField(cleanup, "sessions");
            return java.util.Objects.requireNonNull(records.get(claim));
        }

        void assertArchivedAfterCleanup() throws Exception {
            settlement().get(2, TimeUnit.SECONDS);
            synchronized (context) {
                assertEquals(CancelReason.CLIENT_CANCELLED, context.snapshot().delivery().abandonment());
                assertTrue(context.snapshot().delivery().prefillProof());
                assertTrue(context.snapshot().delivery().decodeProof());
            }
            awaitArchive();
            SchedulerTestSupport.awaitRequestWork(owner.runtime);
            assertFalse(context.data().getFuture().join().isSuccess());
            assertFalse(prefillOwned.get(), "archival requires Prefill ownership to be released");
            verify(prefill, times(prefillClearedByState.get() ? 0 : 1)).release(null, claim.item());
            assertDecodeOwnershipReleasedOnce();
        }
        public void close() {
            // End fake RPCs / Worker generations after assertions so accepted callbacks can drain.
            cancel.complete(EngineGrpcClient.CancelReply.FAILED);
            if (owner.requests.isCurrent(context)) {
                var send = context.snapshot().delivery().send();
                if (send == ContextResult.SendState.NOT_STARTED || send == ContextResult.SendState.SENDING) {
                    owner.onDeliveryEvidence(claim, new ContextResult.TransportResult(claim,
                            DeliveryResult.notSent(new IllegalStateException("fixture sender exited")), System.currentTimeMillis()));
                }
                retirePrefill();
                retireDecode();
            }
            SchedulerTestSupport.runtime(owner).requests().stopAccepting();
            ((QueuedRequestScheduler)owner).close();
            SchedulerTestSupport.timer(owner.runtime).close();
            SchedulerTestSupport.closeRequestExecutors(owner.runtime);
        }
    }
}
