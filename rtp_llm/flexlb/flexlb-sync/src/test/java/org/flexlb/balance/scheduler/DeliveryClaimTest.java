package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.DecodeResources;
import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.eviction.EngineCancelChannel;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.service.RecentCacheKeyTraceReporter;
import org.flexlb.service.monitor.DeliveryMetricsReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;
import org.junit.jupiter.params.provider.CsvSource;
import org.springframework.test.util.ReflectionTestUtils;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import static org.junit.jupiter.api.Assertions.*;
import static org.mockito.Mockito.*;

class DeliveryClaimTest {
    @Test void decodeProofAndTerminalEffectsShareOneQueuedCleanupPass() throws Exception {
        try (var f = new Fixture()) {
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.completeDelivery(f.claim, DeliveryResult.delivered());
            f.owner.runtime.continuations().awaitIdle();
            var executor = spy(f.owner.runtime.continuations());
            ReflectionTestUtils.setField(f.owner, "continuations", executor);
            ReflectionTestUtils.setField(f.owner.runtime, "continuations", executor);

            f.applyDecodeTerminal();
            executor.awaitIdle();

            // The real per-context queue runs the status effect, then exactly one cleanup effect.
            verify(executor, times(2)).submit(eq(f.context), any(Runnable.class));
            assertTrue(f.settlement().isDone());
            assertFalse(f.owner.requests.isCurrent(f.context));
            assertFalse(f.prefillOwned.get());
            assertFalse(f.decodeOwned.get());
            verify(f.prefill, times(1)).releaseRequest(f.claim.item);
            verify(f.decode, never()).release(any(), any());
        }
    }

    @Test void terminalBeforeSenderFinishesQueuesCleanupOnlyAfterProofAndShutdownDrainsIt() throws Exception {
        try (var f = new Fixture()) {
            assertTrue(f.owner.tryStartSend(f.claim));
            var executor = spy(f.owner.runtime.continuations());
            ReflectionTestUtils.setField(f.owner, "continuations", executor);
            ReflectionTestUtils.setField(f.owner.runtime, "continuations", executor);

            f.applyDecodeTerminal();
            executor.awaitIdle();
            verify(executor, times(1)).submit(eq(f.context), any(Runnable.class));
            assertFalse(f.settlement().isDone());
            assertTrue(f.owner.requests.isCurrent(f.context));
            assertTrue(f.prefillOwned.get(), "the sender still pins Prefill ownership");
            verify(f.prefill, never()).releaseRequest(any());

            f.owner.runtime.stopAccepting();
            f.owner.completeDelivery(f.claim, DeliveryResult.delivered());
            f.owner.runtime.shutdown();

            verify(executor, times(2)).submit(eq(f.context), any(Runnable.class));
            assertTrue(f.settlement().isDone());
            assertFalse(f.owner.requests.isCurrent(f.context));
            assertFalse(f.prefillOwned.get());
            assertFalse(f.decodeOwned.get());
            verify(f.prefill, times(1)).releaseRequest(f.claim.item);
            verify(f.decode, never()).release(any(), any());
        }
    }

    @Test void rejectedCleanupQueueClaimRemainsRetryableWithoutReleasingResources() throws Exception {
        try (var f = new Fixture()) {
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.completeDelivery(f.claim, DeliveryResult.delivered());
            f.owner.runtime.continuations().awaitIdle();
            f.clearDecodeStateOwnership();
            f.claim.recordDecodeTerminalStatus(f.decode, DecodeResources.DecodeRequestStatus.terminal(f.reservation, 0L));
            f.owner.settleDelivery(f.claim);
            Runnable terminal = f.context.scheduler().acceptDecodeStatus(f.context, f.decode,
                    DecodeResources.DecodeRequestStatus.terminal(f.reservation, 0L), System.currentTimeMillis());
            assertNotNull(terminal);
            var executor = spy(f.owner.runtime.continuations());
            ReflectionTestUtils.setField(f.owner, "continuations", executor);
            ReflectionTestUtils.setField(f.owner.runtime, "continuations", executor);
            var rejected = new IllegalStateException("queue acceptance rejected");
            doAnswer(call -> {
                assertFalse(Thread.holdsLock(f.context), "queue submission must run outside the request monitor");
                f.owner.resumeCleanup(f.context);
                f.owner.enqueueCleanup(f.context);
                assertTrue(f.prefillOwned.get(), "a synchronous event cannot consume an unaccepted queue claim");
                throw rejected;
            }).doCallRealMethod().when(executor).submit(eq(f.context), any(Runnable.class));

            assertSame(rejected, assertThrows(IllegalStateException.class, () -> f.owner.enqueueCleanup(f.context)));
            assertTrue(f.prefillOwned.get());
            assertTrue(f.owner.requests.isCurrent(f.context));
            verify(f.prefill, never()).releaseRequest(any());

            f.owner.enqueueCleanup(f.context);
            executor.awaitIdle();
            verify(executor, times(2)).submit(eq(f.context), any(Runnable.class));
            assertFalse(f.prefillOwned.get(), "a rejected claim must not suppress the accepted retry");
            assertTrue(f.owner.requests.isCurrent(f.context), "terminal effects still own the archive gate");
            terminal.run();
            executor.awaitIdle();
            assertFalse(f.owner.requests.isCurrent(f.context));
            verify(f.prefill, times(1)).releaseRequest(f.claim.item);
            verify(f.decode, never()).release(any(), any());
        }
    }

    @Test void runningCleanupStillAcceptsANotificationAndPreventsEarlyShutdown() throws Exception {
        try (var f = new Fixture()) {
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.completeDelivery(f.claim, DeliveryResult.delivered());
            f.owner.runtime.continuations().awaitIdle();
            var executor = spy(f.owner.runtime.continuations());
            ReflectionTestUtils.setField(f.owner, "continuations", executor);
            ReflectionTestUtils.setField(f.owner.runtime, "continuations", executor);
            var entered = new java.util.concurrent.CountDownLatch(1);
            var release = new java.util.concurrent.CountDownLatch(1);
            doAnswer(call -> {
                assertFalse(Thread.holdsLock(f.context));
                entered.countDown();
                RequestProtocolTestSupport.await(release);
                assertTrue(f.prefillOwned.compareAndSet(true, false));
                return true;
            }).when(f.prefill).releaseRequest(f.claim.item);
            var shutdownFailure = new AtomicReference<Throwable>();
            Thread shutdown = new Thread(() -> {
                try { f.owner.runtime.shutdown(); }
                catch (Throwable failure) { shutdownFailure.set(failure); }
            });
            try {
                f.applyDecodeTerminal();
                assertTrue(entered.await(2, TimeUnit.SECONDS));
                f.owner.enqueueCleanup(f.context);
                verify(executor, times(3)).submit(eq(f.context), any(Runnable.class));
                shutdown.start();
                RequestProtocolTestSupport.awaitCondition(() -> shutdown.getState() == Thread.State.WAITING);
                assertTrue(shutdown.isAlive());
                assertTrue(f.owner.requests.isCurrent(f.context));
            } finally {
                release.countDown();
                shutdown.join(2_000);
            }
            assertFalse(shutdown.isAlive());
            assertNull(shutdownFailure.get());
            assertFalse(f.owner.requests.isCurrent(f.context));
            verify(f.prefill, times(1)).releaseRequest(f.claim.item);
            verify(f.decode, never()).release(any(), any());
        }
    }

    @Test void cancellationBeforeSendProducesNoRemoteCancel() throws Exception {
        try (var f = new Fixture()) {
            f.owner.cancel(41L, 0L, CancelReason.CLIENT_CANCELLED);
            assertFalse(f.owner.tryStartSend(f.claim));
            assertFalse(f.settlement().isDone(), "the sender still owns the prepared delivery");
            verifyNoInteractions(f.channel);
            f.owner.completeDelivery(f.claim, DeliveryResult.notSent(new IllegalStateException("cancelled before send")));
            f.settlement().get(2, TimeUnit.SECONDS);
            f.awaitArchive();
        }
    }

    @Test void earlyRemoteCleanupCannotReleaseTheSenderObligation() throws Exception {
        try (var f = new Fixture()) {
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.cancel(41L, 0L, CancelReason.CLIENT_CANCELLED);
            verify(f.channel).cancel(any(), eq(41L), eq(CancelReason.CLIENT_CANCELLED), anyLong());
            f.cancel.complete(EngineCancelChannel.CancelAck.REQUEST_CLEANED);
            assertFalse(f.settlement().isDone());
            assertTrue(f.owner.requests.isCurrent(f.context));
            SchedulerTestSupport.runtime(f.owner).stopAccepting();

            f.owner.completeDelivery(f.claim, DeliveryResult.delivered());
            f.settlement().get(2, TimeUnit.SECONDS);
            f.awaitArchive();
            assertFalse(f.context.getFuture().join().isSuccess(), "late ACK cannot replace cancellation");
            SchedulerTestSupport.runtime(f.owner).shutdown();

        }
    }

    @Test void prefillFenceWaitsForExactDecodeSettlement() throws Exception {
        try (var f = new Fixture()) {
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.completeDelivery(f.claim, DeliveryResult.uncertain(new IllegalStateException("reply lost")));
            f.owner.onResponseUndeliverable(f.context);
            f.cancel.complete(EngineCancelChannel.CancelAck.REQUEST_FENCED);
            assertFalse(f.settlement().isDone());
            f.claim.recordDecodeTerminalStatus(RequestProtocolTestSupport.decodeEndpoint(), DecodeResources.DecodeRequestStatus.terminal(f.reservation, 0L));
            f.owner.settleDelivery(f.claim);
            assertFalse(f.settlement().isDone(), "another endpoint is not proof");
            f.claim.recordDecodeTerminalStatus(f.decode, DecodeResources.DecodeRequestStatus.terminal(
                    new DecodeResources.ReservationHandle(1L, 41L, 99L), 0L));
            f.owner.settleDelivery(f.claim);
            assertFalse(f.settlement().isDone(), "another reservation is not proof");
            f.applyDecodeTerminal();
            f.settlement().get(2, TimeUnit.SECONDS);
        }
    }

    @Test void failedResponseDeliveryAfterSelectedSuccessStillCancelsEngine() throws Exception {
        try (var f = new Fixture()) {
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.completeDelivery(f.claim, DeliveryResult.delivered());
            assertTrue(f.context.getFuture().get(2, TimeUnit.SECONDS).isSuccess());
            f.owner.onResponseUndeliverable(f.context);
            verify(f.channel).cancel(any(), eq(41L), eq(CancelReason.CLIENT_CANCELLED), anyLong());
            assertTrue(f.context.getFuture().join().isSuccess(), "selected result remains immutable");
            assertFalse(f.settlement().isDone());
            f.cancel.complete(EngineCancelChannel.CancelAck.REQUEST_CLEANED);
            f.settlement().get(2, TimeUnit.SECONDS);
            f.awaitArchive();
        }
    }

    @Test void rejectedCleanupExecutorRecordsFailureWithoutArchiving() throws Exception {
        try (var f = new Fixture()) {
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.runtime.cleanupExecutor().shutdown();
            f.owner.cancel(41L, 0L, CancelReason.SHUTDOWN);
            f.owner.completeDelivery(f.claim, DeliveryResult.delivered());
            assertFalse(f.settlement().isDone(), "executor failure is not remote cleanup proof");
            assertTrue(f.owner.requests.isCurrent(f.context));
            org.junit.jupiter.api.Assertions.assertNotNull(SchedulerTestSupport.failure(f.owner));
        }
    }

    @Test void cleanupTimeoutWithoutProofKeepsTheResourceObligationPending() throws Exception {
        try (var f = new Fixture()) {
            Runnable timeout = f.controlCleanupTimeout();
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.cancel(41L, 0L, CancelReason.CLIENT_CANCELLED);
            f.owner.completeDelivery(f.claim, DeliveryResult.delivered());

            timeout.run();
            f.owner.runtime.continuations().awaitIdle();

            assertAll(
                    () -> assertFalse(f.settlement().isDone(), "cleanup execution failure is not resource settlement"),
                    () -> assertTrue(f.owner.requests.isCurrent(f.context)),
                    () -> assertNull(f.claim.provenReleaseReason()),
                    () -> verify(f.prefill, never()).releaseRequest(any()),
                    () -> verify(f.decode, never()).release(any(), any()));
        }
    }

    @Test void activeDecodeHeartbeatCannotPublishAnUnprocessedNotSentResult() throws Exception {
        try (var f = new Fixture()) {
            // Pause the sender between recording its result and scheduling the failure effects.
            f.claim.recordSendResult(DeliveryResult.notSent(new IllegalStateException("not sent")));
            Runnable heartbeat = f.owner.acceptDecodeStatus(f.context, f.decode,
                    DecodeResources.DecodeRequestStatus.active(f.reservation), System.currentTimeMillis());
            assertNotNull(heartbeat, "the current decision deadline must produce a heartbeat continuation");
            heartbeat.run();
            assertFalse(f.settlement().isDone(), "ACTIVE is not delivery release evidence");
            f.owner.recordSchedulingFailure(f.context, org.flexlb.dao.loadbalance.StrategyErrorType.NO_AVAILABLE_WORKER, "not sent");
            f.owner.settleDelivery(f.claim);
            f.owner.runtime.continuations().awaitIdle();
            f.awaitArchive();
        }
    }

    @Test void notSentQueuesCleanupBeforePublishingTheSettlementBarrier() throws Exception {
        try (var f = new Fixture()) {
            RequestContinuationExecutor original = f.owner.runtime.continuations();
            RequestContinuationExecutor held = mock(RequestContinuationExecutor.class);
            var accepted = new java.util.ArrayList<Runnable>();
            org.mockito.Mockito.doAnswer(call -> {
                assertFalse(f.settlement().isDone(), "cleanup must be accepted before the settlement barrier opens");
                accepted.add(call.getArgument(1));
                return null;
            }).when(held).submit(org.mockito.ArgumentMatchers.same(f.context), any(Runnable.class));
            ReflectionTestUtils.setField(f.owner, "continuations", held);
            try {
                f.owner.completeDelivery(f.claim, DeliveryResult.notSent(new IllegalStateException("not sent")));
                assertFalse(accepted.isEmpty(), "the failure must have a cleanup owner");
                assertTrue(f.settlement().isDone(), "NOT_SENT proves remote resources can be released");
            } finally {
                ReflectionTestUtils.setField(f.owner, "continuations", original);
                for (Runnable work : accepted) { work.run(); }
            }
            f.awaitArchive();
            assertFalse(f.context.getFuture().get(2, TimeUnit.SECONDS).isSuccess());
            assertNull(f.claim.cleanupEvidence().reason(), "NOT_SENT needs no remote abandonment protocol");
            assertFalse(f.prefillOwned.get());
            assertFalse(f.decodeOwned.get());
            verify(f.decode).release(f.reservation, DecodeResources.ReleaseReason.NOT_SENT);
            verify(f.channel, never()).cancel(any(), anyLong(), any(), anyLong());
        }
    }

    @Test void lateCleanedAckAfterTimeoutArchivesOnlyTheOriginalRequest() throws Exception {
        try (var f = new Fixture()) {
            Runnable timeout = f.controlCleanupTimeout();
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.cancel(41L, 0L, CancelReason.CLIENT_CANCELLED);
            f.owner.completeDelivery(f.claim, DeliveryResult.delivered());
            Response selected = f.context.getFuture().get(2, TimeUnit.SECONDS);

            timeout.run();
            f.cancel.complete(EngineCancelChannel.CancelAck.REQUEST_CLEANED);
            f.assertArchivedAfterCleanup();
            assertSame(selected, f.context.getFuture().join(), "cleanup cannot replace the selected response");

            var terminal = f.owner.requests.findTerminal(41L);
            assertTrue(f.owner.requests.removeExactTerminal(terminal, Long.MAX_VALUE));
            var replacement = RequestProtocolTestSupport.context(f.context.getConfig(), 41L);
            var replacementFuture = RequestProtocolTestSupport.register(f.owner, replacement);
            try {
                f.owner.acceptDeliveryCleanup(f.claim, EngineCancelChannel.CancelAck.REQUEST_CLEANED);
                f.applyDecodeTerminal();
                f.owner.onPrefillGenerationRetired(f.prefill, f.claim.item);
                f.owner.runtime.continuations().awaitIdle();

                assertSame(replacement, f.owner.requests.findActive(41L));
                assertFalse(replacementFuture.isDone(), "old cleanup facts cannot finish a reused request identity");
                verify(f.prefill, times(1)).releaseRequest(f.claim.item);
                f.assertDecodeOwnershipReleasedOnce();
            } finally {
                replacementFuture.completeExceptionally(new IllegalStateException("test cleanup"));
            }
        }
    }

    @Test void endpointRetirementAfterCleanupTimeoutStillArchivesTheRequest() throws Exception {
        try (var f = new Fixture()) {
            Runnable timeout = f.controlCleanupTimeout();
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.cancel(41L, 0L, CancelReason.CLIENT_CANCELLED);
            f.owner.completeDelivery(f.claim, DeliveryResult.delivered());

            timeout.run();
            f.owner.onPrefillGenerationRetired(f.prefill, f.claim.item);
            f.retireDecode();
            f.owner.runtime.continuations().awaitIdle();

            assertTrue(f.claim.cleanupEvidence().remoteSettled(), "both exact generations have retired");
            f.assertArchivedAfterCleanup();
        }
    }

    @Test void lateFenceAfterTimeoutWaitsForTheExactDecodeTerminal() throws Exception {
        try (var f = new Fixture()) {
            Runnable timeout = f.controlCleanupTimeout();
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.cancel(41L, 0L, CancelReason.CLIENT_CANCELLED);
            f.owner.completeDelivery(f.claim, DeliveryResult.delivered());

            timeout.run();
            f.cancel.complete(EngineCancelChannel.CancelAck.REQUEST_FENCED);
            f.owner.onDecodeStatus(f.context, RequestProtocolTestSupport.decodeEndpoint(), DecodeResources.DecodeRequestStatus.terminal(f.reservation, 0L));
            f.owner.onDecodeStatus(f.context, f.decode, DecodeResources.DecodeRequestStatus.terminal(
                    new DecodeResources.ReservationHandle(1L, 41L, 99L), 0L));
            f.owner.runtime.continuations().awaitIdle();
            assertFalse(f.settlement().isDone(), "Prefill fencing and foreign Decode facts do not settle the request");
            assertTrue(f.owner.requests.isCurrent(f.context));
            verify(f.decode, never()).release(any(), any());

            f.applyDecodeTerminal();
            f.assertArchivedAfterCleanup();
        }
    }

    @Test void lateCleanedAckAfterTimeoutStillWaitsForSenderExit() throws Exception {
        try (var f = new Fixture()) {
            Runnable timeout = f.controlCleanupTimeout();
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.cancel(41L, 0L, CancelReason.CLIENT_CANCELLED);

            timeout.run();
            f.cancel.complete(EngineCancelChannel.CancelAck.REQUEST_CLEANED);
            f.owner.runtime.continuations().awaitIdle();
            assertFalse(f.settlement().isDone(), "remote cleanup cannot settle an active sender");
            assertTrue(f.owner.requests.isCurrent(f.context));
            verify(f.prefill, never()).releaseRequest(any());
            verify(f.decode, never()).release(any(), any());

            f.owner.completeDelivery(f.claim, DeliveryResult.delivered());
            f.assertArchivedAfterCleanup();
        }
    }

    @ParameterizedTest
    @EnumSource(value = EngineCancelChannel.CancelAck.class,
            names = {"ACCEPTED", "NOT_FOUND", "UNSUPPORTED", "FAILED"})
    void acknowledgementWithoutCleanupProofCannotReleaseAnUncertainSend(EngineCancelChannel.CancelAck ack) throws Exception {
        try (var f = new Fixture()) {
            f.controlCleanupTimeout();
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.completeDelivery(f.claim, DeliveryResult.uncertain(new IllegalStateException("reply lost")));
            f.owner.onResponseUndeliverable(f.context);
            f.cancel.complete(ack);
            f.owner.runtime.continuations().awaitIdle();
            assertFalse(f.owner.tryStartSend(f.claim), "uncertain delivery must not be sent again");
            assertFalse(f.settlement().isDone());
            assertTrue(f.owner.requests.isCurrent(f.context));
            assertNull(f.claim.provenReleaseReason());
            verify(f.prefill, never()).releaseRequest(any());
            verify(f.decode, never()).release(any(), any());

            f.owner.acceptDeliveryCleanup(f.claim, EngineCancelChannel.CancelAck.REQUEST_CLEANED);
            f.assertArchivedAfterCleanup();
        }
    }

    @ParameterizedTest
    @CsvSource({"true,true", "true,false", "false,true", "false,false"})
    void fenceDecodeProofAndSenderExitSettleInAnyOrder(boolean decodeFirst, boolean senderFirst) throws Exception {
        try (var f = new Fixture()) {
            f.controlCleanupTimeout();
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.cancel(41L, 0L, CancelReason.CLIENT_CANCELLED);
            if (senderFirst) { f.owner.completeDelivery(f.claim, DeliveryResult.uncertain(new IllegalStateException("reply lost"))); }
            Runnable decodeProof = f::applyDecodeTerminal;
            if (decodeFirst) { decodeProof.run(); }
            else { f.cancel.complete(EngineCancelChannel.CancelAck.REQUEST_FENCED); }
            f.owner.runtime.continuations().awaitIdle();
            assertFalse(f.settlement().isDone(), "one side of the cleanup proof cannot settle delivery");
            verify(f.prefill, never()).releaseRequest(any());

            if (decodeFirst) { f.cancel.complete(EngineCancelChannel.CancelAck.REQUEST_FENCED); }
            else { decodeProof.run(); }
            f.owner.runtime.continuations().awaitIdle();
            if (!senderFirst) {
                assertFalse(f.settlement().isDone(), "complete remote proof still needs sender exit");
                f.owner.completeDelivery(f.claim, DeliveryResult.uncertain(new IllegalStateException("reply lost")));
            }
            f.assertArchivedAfterCleanup();
            f.owner.acceptDeliveryCleanup(f.claim, EngineCancelChannel.CancelAck.REQUEST_CLEANED);
            decodeProof.run();
            f.owner.runtime.continuations().awaitIdle();
            verify(f.prefill, times(1)).releaseRequest(f.claim.item);
            f.assertDecodeOwnershipReleasedOnce();
        }
    }

    @Test void repeatedCancellationPreservesFirstReasonAndStartsOneCleanup() throws Exception {
        try (var f = new Fixture()) {
            assertTrue(f.owner.tryStartSend(f.claim));
            assertFalse(f.owner.tryStartSend(f.claim), "send qualification is consumed once");
            f.owner.cancel(41L, 0L, CancelReason.CLIENT_CANCELLED);
            f.owner.cancel(41L, 0L, CancelReason.SHUTDOWN);
            f.owner.onResponseUndeliverable(f.context);
            f.owner.runtime.continuations().awaitIdle();
            verify(f.channel, times(1)).cancel(any(), eq(41L), eq(CancelReason.CLIENT_CANCELLED), anyLong());
            f.owner.completeDelivery(f.claim, DeliveryResult.uncertain(new IllegalStateException("reply lost")));
            f.cancel.complete(EngineCancelChannel.CancelAck.REQUEST_CLEANED);
            f.assertArchivedAfterCleanup();
        }
    }

    @ParameterizedTest
    @org.junit.jupiter.params.provider.MethodSource("cleanupEventOrders")
    void everyCleanupEventPermutationRequiresAllProofAndReleasesOnce(String outcome, String order) throws Exception {
        try (var f = new Fixture()) {
            f.controlCleanupTimeout();
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.cancel(41L, 0L, CancelReason.CLIENT_CANCELLED);
            for (int index = 0; index < order.length(); index++) {
                switch (order.charAt(index)) {
                    case 'S' -> f.owner.completeDelivery(f.claim, switch (outcome) {
                        case "DELIVERED" -> DeliveryResult.delivered();
                        case "REJECTED" -> DeliveryResult.prefillRejected(new IllegalStateException("rejected"));
                        case "UNKNOWN" -> DeliveryResult.uncertain(new IllegalStateException("reply lost"));
                        default -> throw new AssertionError(outcome);
                    });
                    case 'P' -> f.cancel.complete(EngineCancelChannel.CancelAck.REQUEST_FENCED);
                    case 'D' -> f.applyDecodeTerminal();
                    default -> throw new AssertionError(order);
                }
                f.owner.runtime.continuations().awaitIdle();
                if (index < order.length() - 1) {
                    assertFalse(f.settlement().isDone(), "partial evidence: " + order.substring(0, index + 1));
                    assertTrue(f.owner.requests.isCurrent(f.context));
                    verify(f.prefill, never()).releaseRequest(any());
                }
            }
            f.assertArchivedAfterCleanup();
            assertFalse(f.owner.tryStartSend(f.claim));
            assertThrows(IllegalStateException.class, () -> f.owner.completeDelivery(f.claim, DeliveryResult.delivered()));
            f.owner.acceptDeliveryCleanup(f.claim, EngineCancelChannel.CancelAck.REQUEST_CLEANED);
            f.applyDecodeTerminal();
            f.owner.runtime.continuations().awaitIdle();
            verify(f.prefill, times(1)).releaseRequest(f.claim.item);
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
            f.controlCleanupTimeout();
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.cancel(41L, 0L, CancelReason.CLIENT_CANCELLED);
            f.owner.completeDelivery(f.claim, DeliveryResult.uncertain(new IllegalStateException("reply lost")));
            f.cancel.complete(EngineCancelChannel.CancelAck.REQUEST_FENCED);
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
            f.claim.recordDecodeTerminalStatus(source, requestStatus);
            f.owner.settleDelivery(f.claim);
            f.owner.runtime.continuations().awaitIdle();
            assertFalse(f.settlement().isDone());
            assertNull(f.claim.provenReleaseReason());
            assertTrue(f.decodeOwned.get(), "foreign facts cannot release the exact Decode owner");
            verify(f.prefill, never()).releaseRequest(any());
            verify(f.decode, never()).release(any(), any());
            f.applyDecodeTerminal();
            f.assertArchivedAfterCleanup();
        }
    }

    @ParameterizedTest
    @org.junit.jupiter.params.provider.ValueSource(ints = {1, 2, 3, 4, 5, 6, 7, 8})
    void cleanupCanSucceedAtEveryBoundedRetryAndCancelsItsDeadline(int successfulAttempt) throws Exception {
        try (var f = new Fixture()) {
            ControlledCleanup timer = f.controlCleanup();
            var replies = new java.util.concurrent.CopyOnWriteArrayList<CompletableFuture<EngineCancelChannel.CancelAck>>();
            when(f.channel.cancel(any(), anyLong(), any(), anyLong())).thenAnswer(call -> {
                assertFalse(Thread.holdsLock(f.context));
                var reply = new CompletableFuture<EngineCancelChannel.CancelAck>();
                replies.add(reply);
                return reply;
            });
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.cancel(41L, 0L, CancelReason.CLIENT_CANCELLED);
            f.owner.completeDelivery(f.claim, DeliveryResult.uncertain(new IllegalStateException("reply lost")));
            f.owner.runtime.continuations().awaitIdle();
            for (int attempt = 1; attempt <= successfulAttempt; attempt++) {
                assertEquals(attempt, replies.size());
                replies.get(attempt - 1).complete(attempt == successfulAttempt
                        ? EngineCancelChannel.CancelAck.REQUEST_CLEANED : EngineCancelChannel.CancelAck.NOT_FOUND);
                f.owner.runtime.continuations().awaitIdle();
                if (attempt < successfulAttempt) {
                    assertFalse(f.settlement().isDone());
                    verify(f.prefill, never()).releaseRequest(any());
                    timer.runRetry(Math.min(400L, 25L << (attempt - 1)));
                }
            }
            f.assertArchivedAfterCleanup();
            verify(f.channel, times(successfulAttempt)).cancel(any(), eq(41L), eq(CancelReason.CLIENT_CANCELLED), eq(500L));
            verify(timer.deadline).cancel(false);
            assertTrue(timer.retries.isEmpty());
        }
    }

    @Test void timedOutAcknowledgementStillSuppliesLateCleanupProof() throws Exception {
        try (var f = new Fixture()) {
            ControlledCleanup timer = f.controlCleanup();
            var replies = new java.util.concurrent.CopyOnWriteArrayList<CompletableFuture<EngineCancelChannel.CancelAck>>();
            when(f.channel.cancel(any(), anyLong(), any(), anyLong())).thenAnswer(call -> {
                var reply = new CompletableFuture<EngineCancelChannel.CancelAck>();
                replies.add(reply);
                return reply;
            });
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.cancel(41L, 0L, CancelReason.CLIENT_CANCELLED);
            f.owner.completeDelivery(f.claim, DeliveryResult.uncertain(new IllegalStateException("reply lost")));
            f.owner.runtime.continuations().awaitIdle();
            assertEquals(1, replies.size());

            timer.runAckTimeout();
            f.owner.runtime.continuations().awaitIdle();
            assertFalse(f.settlement().isDone());
            timer.runRetry(25L);
            assertEquals(2, replies.size());

            replies.getFirst().complete(EngineCancelChannel.CancelAck.REQUEST_CLEANED);
            f.assertArchivedAfterCleanup();
            verify(f.channel, times(2)).cancel(any(), eq(41L), any(), eq(500L));
            assertTrue(timer.retries.isEmpty());
        }
    }

    @Test void overallTimeoutStopsAnAlreadyScheduledRetry() throws Exception {
        try (var f = new Fixture()) {
            ControlledCleanup timer = f.controlCleanup();
            when(f.channel.cancel(any(), anyLong(), any(), anyLong())).thenReturn(
                    CompletableFuture.completedFuture(EngineCancelChannel.CancelAck.NOT_FOUND));
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.cancel(41L, 0L, CancelReason.CLIENT_CANCELLED);
            f.owner.completeDelivery(f.claim, DeliveryResult.uncertain(new IllegalStateException("reply lost")));
            f.owner.runtime.continuations().awaitIdle();
            verify(f.channel, times(1)).cancel(any(), eq(41L), any(), eq(500L));

            timer.runDeadline();
            timer.runRetry(25L); // A callback already taken from the timer queue may still run.
            f.owner.runtime.continuations().awaitIdle();
            verify(f.channel, times(1)).cancel(any(), eq(41L), any(), eq(500L));
            assertFalse(f.settlement().isDone());
            org.junit.jupiter.api.Assertions.assertNotNull(SchedulerTestSupport.failure(f.owner));
        }
    }

    @Test void retryExhaustionStillAcceptsAnEarlierTimedOutAttemptProof() throws Exception {
        try (var f = new Fixture()) {
            ControlledCleanup timer = f.controlCleanup();
            var first = new CompletableFuture<EngineCancelChannel.CancelAck>();
            var calls = new java.util.concurrent.atomic.AtomicInteger();
            when(f.channel.cancel(any(), anyLong(), any(), anyLong())).thenAnswer(call ->
                    calls.incrementAndGet() == 1 ? first
                            : CompletableFuture.completedFuture(EngineCancelChannel.CancelAck.NOT_FOUND));
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.cancel(41L, 0L, CancelReason.CLIENT_CANCELLED);
            f.owner.completeDelivery(f.claim, DeliveryResult.uncertain(new IllegalStateException("reply lost")));
            f.owner.runtime.continuations().awaitIdle();
            timer.runAckTimeout();
            f.owner.runtime.continuations().awaitIdle();
            for (int attempt = 1; attempt < 8; attempt++) {
                timer.runRetry(Math.min(400L, 25L << (attempt - 1)));
                f.owner.runtime.continuations().awaitIdle();
            }
            assertEquals(8, calls.get());
            org.junit.jupiter.api.Assertions.assertNotNull(SchedulerTestSupport.failure(f.owner));
            assertFalse(f.settlement().isDone());
            assertTrue(timer.retries.isEmpty());

            first.complete(EngineCancelChannel.CancelAck.REQUEST_CLEANED);
            f.assertArchivedAfterCleanup();
            assertEquals(8, calls.get(), "late proof cannot restart exhausted retries");
        }
    }

    @ParameterizedTest
    @EnumSource(value = EngineCancelChannel.CancelAck.class,
            names = {"ACCEPTED", "NOT_FOUND", "UNSUPPORTED", "FAILED"})
    void retryExhaustionCannotInventResourceRelease(EngineCancelChannel.CancelAck ack) throws Exception {
        try (var f = new Fixture()) {
            ControlledCleanup timer = f.controlCleanup();
            when(f.channel.cancel(any(), anyLong(), any(), anyLong())).thenAnswer(call -> CompletableFuture.completedFuture(ack));
            assertTrue(f.owner.tryStartSend(f.claim));
            f.owner.cancel(41L, 0L, CancelReason.CLIENT_CANCELLED);
            f.owner.completeDelivery(f.claim, DeliveryResult.uncertain(new IllegalStateException("reply lost")));
            f.owner.runtime.continuations().awaitIdle();
            for (int attempt = 1; attempt < 8; attempt++) {
                verify(f.channel, times(attempt)).cancel(any(), eq(41L), any(), anyLong());
                timer.runRetry(Math.min(400L, 25L << (attempt - 1)));
                f.owner.runtime.continuations().awaitIdle();
            }
            verify(f.channel, times(8)).cancel(any(), eq(41L), any(), anyLong());
            assertTrue(timer.retries.isEmpty(), "retry budget must end without scheduling a ninth attempt");
            assertTrue(f.owner.requests.isCurrent(f.context));
            assertNull(f.claim.provenReleaseReason());
            verify(f.prefill, never()).releaseRequest(any());
            verify(f.decode, never()).release(any(), any());
        }
    }

    private static final class ControlledCleanup {
        final java.util.ArrayDeque<java.util.Map.Entry<Long, Runnable>> retries = new java.util.ArrayDeque<>();
        final java.util.ArrayDeque<Runnable> acknowledgements = new java.util.ArrayDeque<>();
        final ScheduledFuture<?> deadline = mock(ScheduledFuture.class);
        Runnable deadlineAction;
        final ScheduledThreadPoolExecutor executor = spy(new ScheduledThreadPoolExecutor(1));

        ControlledCleanup() {
            doAnswer(call -> {
                long delay = call.getArgument(1);
                if (delay == DeliveryCleanupTask.CLEANUP_TIMEOUT_MS) {
                    deadlineAction = call.getArgument(0);
                    return deadline;
                }
                if (delay < 500L) { retries.addLast(java.util.Map.entry(delay, call.getArgument(0))); }
                if (delay == 500L) { acknowledgements.addLast(call.getArgument(0)); }
                return mock(ScheduledFuture.class);
            }).when(executor).schedule(any(Runnable.class), anyLong(), eq(TimeUnit.MILLISECONDS));
        }

        void runRetry(long expectedDelay) {
            var retry = retries.pollFirst();
            assertNotNull(retry, "cleanup must schedule a retry");
            assertEquals(expectedDelay, retry.getKey().longValue());
            retry.getValue().run();
        }

        void runAckTimeout() {
            Runnable timeout = acknowledgements.pollFirst();
            assertNotNull(timeout, "cancel attempt must have an acknowledgement deadline");
            timeout.run();
        }

        void runDeadline() {
            assertNotNull(deadlineAction, "cleanup must have an overall deadline");
            deadlineAction.run();
        }
    }

    private static final class Fixture implements AutoCloseable {
        final EngineCancelChannel channel = mock(EngineCancelChannel.class);
        final CompletableFuture<EngineCancelChannel.CancelAck> cancel = new CompletableFuture<>();
        final AbstractRequestScheduler owner;
        final RequestContext context;
        final PrefillEndpoint prefill = mock(PrefillEndpoint.class);
        final java.util.concurrent.atomic.AtomicBoolean prefillOwned = new java.util.concurrent.atomic.AtomicBoolean(true);
        final java.util.concurrent.atomic.AtomicBoolean decodeOwned = new java.util.concurrent.atomic.AtomicBoolean(true);
        final java.util.concurrent.atomic.AtomicBoolean decodeClearedByState = new java.util.concurrent.atomic.AtomicBoolean();
        final DecodeEndpoint decode = RequestProtocolTestSupport.decodeEndpoint();
        final DecodeResources.ReservationHandle reservation = new DecodeResources.ReservationHandle(1L, 41L, 1L);
        final RequestContext.DeliveryClaim claim;
        Fixture() {
            var config = SchedulingTestConfig.batchConfig();
            var service = mock(ConfigService.class);
            when(service.loadBalanceConfig()).thenReturn(config);
            owner = SchedulerTestSupport.create(service, mock(DeliveryMetricsReporter.class), mock(RequestSchedulerReporter.class), mock(RecentCacheKeyTraceReporter.class));
            ReflectionTestUtils.setField(owner.runtime, "cancelChannel", channel);
            when(channel.cancel(any(), anyLong(), any(), anyLong())).thenReturn(cancel);
            context = RequestProtocolTestSupport.context(config, 41L);
            var future = RequestProtocolTestSupport.register(owner, context);
            var response = new Response();
            response.setSuccess(true);
            var item = org.flexlb.balance.scheduler.SchedulingTestConfig.createRoute(SchedulingTestConfig.freezeInputs(context), response, null, null,
                    prefill, decode, reservation, System.currentTimeMillis());
            when(prefill.releaseRequest(any())).thenAnswer(call -> {
                assertFalse(Thread.holdsLock(context), "resource cleanup must run outside the request monitor");
                assertSame(item, call.getArgument(0), "only this exact Prefill owner may be released");
                assertTrue(prefillOwned.compareAndSet(true, false), "Prefill ownership may be released only once");
                return true;
            });
            when(decode.release(any(), any())).thenAnswer(call -> {
                assertFalse(Thread.holdsLock(context), "Decode cleanup must run outside the request monitor");
                assertEquals(reservation, call.getArgument(0), "only this exact Decode owner may be released");
                assertTrue(decodeOwned.compareAndSet(true, false), "Decode ownership may be released only once");
                return DecodeResources.ReservationReleaseResult.RELEASED;
            });
            RequestProtocolTestSupport.bindRoute(owner, new RequestProtocolTestSupport.Registered(item, future));
            claim = RequestProtocolTestSupport.claimBatch(owner, item, 51L, () -> true);
            assertNotNull(claim);
        }
        CompletableFuture<Void> settlement() { return claim.settlement().toCompletableFuture(); }
        void awaitArchive() throws InterruptedException { RequestProtocolTestSupport.awaitCondition(() -> !owner.requests.isCurrent(context)); }

        // DecodeState removes exact ownership before Endpoint publishes a terminal or retirement fact.
        private void clearDecodeStateOwnership() {
            if (decodeOwned.getAndSet(false)) { decodeClearedByState.set(true); }
        }

        void applyDecodeTerminal() {
            clearDecodeStateOwnership();
            owner.onDecodeStatus(context, decode, DecodeResources.DecodeRequestStatus.terminal(reservation, 0L));
        }

        void retireDecode() {
            clearDecodeStateOwnership();
            when(decode.isRetired()).thenReturn(true);
            owner.onDecodeGenerationRetired(context, decode, reservation);
        }

        void assertDecodeOwnershipReleasedOnce() {
            assertFalse(decodeOwned.get(), "archival requires physical Decode ownership to be released");
            verify(decode, times(decodeClearedByState.get() ? 0 : 1))
                    .release(reservation, DecodeResources.ReleaseReason.REMOTE_CLEANUP);
        }

        ControlledCleanup controlCleanup() {
            var controlled = new ControlledCleanup();
            owner.runtime.cleanupExecutor().shutdown();
            ReflectionTestUtils.setField(owner.runtime, "cleanupExecutor", controlled.executor);
            return controlled;
        }

        Runnable controlCleanupTimeout() {
            AtomicReference<Runnable> deadline = new AtomicReference<>();
            var timer = spy(new ScheduledThreadPoolExecutor(1));
            // Capture real scheduled callbacks; no wall-clock timer runs in these regressions.
            doAnswer(call -> {
                if (call.<Long>getArgument(1) == DeliveryCleanupTask.CLEANUP_TIMEOUT_MS) {
                    deadline.set(call.getArgument(0));
                }
                return mock(ScheduledFuture.class);
            }).when(timer).schedule(any(Runnable.class), anyLong(), eq(TimeUnit.MILLISECONDS));
            owner.runtime.cleanupExecutor().shutdown();
            ReflectionTestUtils.setField(owner.runtime, "cleanupExecutor", timer);
            return () -> {
                assertNotNull(deadline.get(), "the cleanup task must install its overall deadline");
                deadline.get().run();
            };
        }

        void assertArchivedAfterCleanup() throws Exception {
            settlement().get(2, TimeUnit.SECONDS);
            synchronized (context) {
                assertEquals(CancelReason.CLIENT_CANCELLED, ReflectionTestUtils.getField(claim, "abandonmentReason"));
                assertEquals(true, ReflectionTestUtils.getField(claim, "prefillReleaseProven"));
                assertEquals(true, ReflectionTestUtils.getField(claim, "decodeReleaseProven"));
            }
            awaitArchive();
            owner.runtime.continuations().awaitIdle();
            assertFalse(context.getFuture().join().isSuccess());
            assertFalse(prefillOwned.get(), "archival requires Prefill ownership to be released");
            verify(prefill, times(1)).releaseRequest(claim.item);
            assertDecodeOwnershipReleasedOnce();
        }
        public void close() {
            SchedulerTestSupport.runtime(owner).stopAccepting();
            ((QueuedRequestScheduler)owner).close();
            owner.runtime.timer().close();
            owner.runtime.closeRequestExecutors();
        }
    }
}
