package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.scheduler.RequestSlot.AdmissionHandle;
import org.flexlb.balance.scheduler.RequestSlot.DeliveryClaim;
import org.flexlb.config.ConfigService;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.service.monitor.RequestSchedulerReporter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import static org.flexlb.balance.scheduler.RequestLifecycleTestSupport.await;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.spy;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/** Frontend result selection precedes unlocked publication and cannot undo TTL cleanup. */
class RequestCompletionPublicationRaceTest {

    @Test
    @Timeout(15)
    void inactivityWinsWhileAcknowledgementReportingAndTerminalCleanupAreBothPaused() throws Exception {
        var config = spy(SchedulingTestConfig.batchConfig());
        long timeoutMs = TimeUnit.HOURS.toMillis(1L);
        config.getRequestLifecycle().getRequest().setTimeoutMs(timeoutMs);
        var runtime = spy(config.getInternalRuntime());
        when(runtime.getBatchDispatchCompletionThreads()).thenReturn(1);
        when(config.getInternalRuntime()).thenReturn(runtime);
        ConfigService service = mock(ConfigService.class);
        when(service.loadBalanceConfig()).thenReturn(config);
        BatchSchedulerReporter reporter = mock(BatchSchedulerReporter.class);
        RequestRegistry registry = new RequestRegistry(service, reporter,
                mock(RequestSchedulerReporter.class));
        ExecutorService operations = Executors.newFixedThreadPool(2);
        CountDownLatch reportingEntered = new CountDownLatch(1);
        CountDownLatch resumeReporting = new CountDownLatch(1);
        CountDownLatch cleanupEntered = new CountDownLatch(1);
        CountDownLatch resumeCleanup = new CountDownLatch(1);
        try {
            BalanceContext context = RequestLifecycleTestSupport.context(config, 501L);
            CompletableFuture<Response> future = registry.register(context);
            RequestSlot slot = registry.requestSlot(501L);
            PrefillEndpoint prefill = mock(PrefillEndpoint.class);
            when(prefill.getIp()).thenReturn("prefill");
            DecodeEndpoint decode = mock(DecodeEndpoint.class);
            var reservation = new DecodeEndpoint.ReservationHandle(1L, 501L, 1L);
            ScheduledRequest item = new ScheduledRequest(context, future, new Response(), null, null,
                    prefill, decode, reservation, slot.createdAtMs());
            RequestLifecycleTestSupport.bind(registry,
                    new RequestLifecycleTestSupport.Registered(item, future));
            DeliveryClaim claim = RequestLifecycleTestSupport.claimBatch(
                    registry, item, 601L, () -> true);
            assertNotNull(claim);

            doAnswer(invocation -> {
                assertFalse(Thread.holdsLock(slot));
                reportingEntered.countDown();
                await(resumeReporting);
                return null;
            }).when(reporter).reportDispatchAckTimeMs(anyString(), anyString(), anyLong());
            doAnswer(invocation -> {
                assertFalse(Thread.holdsLock(slot));
                cleanupEntered.countDown();
                await(resumeCleanup);
                return DecodeEndpoint.ReservationReleaseResult.RELEASED;
            }).when(decode).release(reservation, DecodeEndpoint.ReleaseReason.EXPIRED);
            CompletableFuture<Void> callback = future.thenAccept(response ->
                    assertFalse(Thread.holdsLock(slot), "frontend callbacks must not hold the slot lock"));

            Future<?> acknowledgement = operations.submit(() -> claim.complete(DeliveryResult.delivered()));
            assertTrue(reportingEntered.await(2L, TimeUnit.SECONDS));
            assertEquals(RequestState.Phase.ACKNOWLEDGED, slot.snapshot().state());
            assertFalse(future.isDone());

            Future<?> expiry = operations.submit(() ->
                    registry.expireInactiveRequest(slot, slot.createdAtMs() + timeoutMs));
            assertTrue(cleanupEntered.await(2L, TimeUnit.SECONDS));
            resumeReporting.countDown();
            acknowledgement.get(2L, TimeUnit.SECONDS);

            // A second response on the single publisher worker proves the old ACK's
            // queued publication has run while the TTL response is still withheld.
            var barrier = registry.register(RequestLifecycleTestSupport.context(config, 502L));
            registry.cancelRequest(502L, 0L, CancelReason.CLIENT_CANCELLED);
            assertFalse(barrier.get(2L, TimeUnit.SECONDS).isSuccess());
            assertFalse(future.isDone(), "an obsolete success permit cannot win after TTL claims cleanup");

            resumeCleanup.countDown();
            expiry.get(2L, TimeUnit.SECONDS);
            Response expired = future.get(2L, TimeUnit.SECONDS);
            callback.get(2L, TimeUnit.SECONDS);
            assertFalse(expired.isSuccess());
            assertEquals(StrategyErrorType.BATCH_SLO_EXPIRED.getErrorCode(), expired.getCode());
            assertEquals(RequestState.Phase.TIMED_OUT, slot.snapshot().state());
            verify(decode).release(reservation, DecodeEndpoint.ReleaseReason.EXPIRED);
            verify(prefill).expireCommittedItem(item);
        } finally {
            resumeReporting.countDown();
            resumeCleanup.countDown();
            operations.shutdownNow();
            operations.awaitTermination(5L, TimeUnit.SECONDS);
            if (registry.closeAdmissionAndAwaitMutations()) {
                registry.closeOutstandingAndTerminalize();
                registry.closeExpiration();
                registry.closePublisher();
            }
        }
    }

    @Test
    void deliverySelectedBeforeExpiryKeepsItsResponseEvenBeforeTheFutureIsCompleted() {
        Fixture fixture = fixture();
        Response success = new Response();
        success.setSuccess(true);
        RequestSlot.SelectedPublication publishSuccess = fixture.slot().selectResponse(fixture.delivery().publication(), success);
        assertFalse(fixture.slot().future().isDone());
        CompletableFuture<Void> callback = fixture.slot().future().thenAccept(response ->
                assertFalse(Thread.holdsLock(fixture.slot())));
        synchronized (fixture.slot()) {
            RequestLifecycleTestSupport.recordCancellation(fixture.slot(), CancelReason.DEADLINE_EXCEEDED, "request inactive");
            TerminalAction terminal = fixture.slot().finishRequest(null,
                    TerminalOutcome.timeout("request inactive"), new Response(), true);
            assertNotNull(terminal);
            assertNull(terminal.publication(), "an already selected delivery owns the frontend result");
            assertEquals(RequestState.Phase.TIMED_OUT,
                    fixture.slot().finishTermination(terminal).terminal().state());
        }
        assertTrue(publishSuccess.complete());
        assertSame(success, fixture.slot().future().join());
        callback.join();
    }

    @ParameterizedTest
    @EnumSource(TerminalForm.class)
    void terminalSelectionInvalidatesAnUnpublishedAcknowledgementForEveryCompletionKind(TerminalForm form) {
        Fixture fixture = fixture();
        Response failure = new Response();
        failure.setSuccess(false);
        TerminalAction terminal;
        synchronized (fixture.slot()) {
            // ACKNOWLEDGED records the Engine fact, not a selected frontend result.
            terminal = fixture.slot().finishRequest(null,
                    TerminalOutcome.fail("worker failed before response publication"), failure, failure != null);
            assertNotNull(terminal.publication());
            fixture.slot().finishTermination(terminal);
        }
        Response success = new Response();
        success.setSuccess(true);
        assertFalse(fixture.slot().selectResponse(fixture.delivery().publication(), success).complete());
        assertFalse(fixture.slot().future().isDone());
        CompletableFuture<Void> callback = fixture.slot().future().handle((response, error) -> {
            assertFalse(Thread.holdsLock(fixture.slot()));
            return null;
        });
        RequestSlot.SelectedPublication publication = switch (form) {
            case RESPONSE -> fixture.slot().selectResponse(terminal.publication(), failure);
            case FAILURE -> fixture.slot().selectFailure(terminal.publication(), new IllegalStateException("worker failed"));
            case CANCELLATION -> fixture.slot().selectCancellation(terminal.publication(), false);
        };
        assertTrue(publication.complete());
        callback.join();
        assertTrue(fixture.slot().future().isDone());
        if (form == TerminalForm.RESPONSE) {
            assertSame(failure, fixture.slot().future().join());
        } else {
            assertTrue(fixture.slot().future().isCompletedExceptionally());
            assertEquals(form == TerminalForm.CANCELLATION, fixture.slot().future().isCancelled());
        }
    }

    @ParameterizedTest
    @EnumSource(TerminalForm.class)
    void externalFutureOperationUnderSlotLockLeavesRequestUnchanged(TerminalForm form) {
        RequestSlot slot = new RequestSlot(mock(RequestCompletionPublisher.class), 702L,
                null, null, null, null);
        synchronized (slot) {
            assertThrows(IllegalStateException.class, () -> {
                switch (form) {
                    case RESPONSE -> slot.future().complete(new Response());
                    case FAILURE -> slot.future().completeExceptionally(new IllegalStateException("failure"));
                    case CANCELLATION -> slot.future().cancel(false);
                }
            });
            assertTrue(slot.isOpen());
            assertEquals(RequestState.Phase.QUEUED, slot.snapshot().state());
            assertFalse(slot.future().isDone());
        }
    }

    private static Fixture fixture() {
        RequestCompletionPublisher publisher = mock(RequestCompletionPublisher.class);
        RequestSlot slot = new RequestSlot(publisher, 701L, null, null, null, null);
        when(publisher.tryReservePublication(eq(slot), any())).thenAnswer(invocation ->
                new RequestSlot.PublicationPermit(publisher, slot, invocation.getArgument(1)));
        var config = SchedulingTestConfig.batchConfig();
        BalanceContext context = RequestLifecycleTestSupport.context(config, slot.requestId());
        ScheduledRequest item = new ScheduledRequest(context, slot.future(), new Response(), null, null,
                null, null, null, slot.createdAtMs());
        RequestSlot.DeliveryPublication delivery;
        synchronized (slot) {
            AdmissionHandle admission = slot.tryBeginAdmissionHandle();
            assertNotNull(admission);
            assertTrue(slot.tryBindItemForPublication(item));
            slot.completeAdmissionHandle(admission);
            RequestLifecycleTestSupport.startBatchDelivery(slot, 801L);
            delivery = RequestLifecycleTestSupport.acknowledge(slot, 801L).delivery();
            assertNotNull(delivery);
        }
        return new Fixture(slot, delivery);
    }

    private enum TerminalForm { RESPONSE, FAILURE, CANCELLATION }

    private record Fixture(RequestSlot slot, RequestSlot.DeliveryPublication delivery) { }
}
