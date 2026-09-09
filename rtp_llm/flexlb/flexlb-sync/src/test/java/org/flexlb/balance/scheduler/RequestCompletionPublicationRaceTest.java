package org.flexlb.balance.scheduler;

import org.flexlb.balance.delivery.DeliveryResult;
import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.PrefillEndpoint;
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
import java.util.function.BooleanSupplier;

import static org.flexlb.balance.scheduler.RequestLifecycleTestSupport.await;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertSame;
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
            RequestRegistry.DeliveryClaim claim = RequestLifecycleTestSupport.claimBatch(
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
                return true;
            }).when(decode).expireReservationExact(reservation);
            CompletableFuture<Void> callback = future.thenAccept(response ->
                    assertFalse(Thread.holdsLock(slot), "frontend callbacks must not hold the slot lock"));

            Future<?> acknowledgement = operations.submit(() -> registry.complete(claim, DeliveryResult.delivered()));
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
            verify(decode).expireReservationExact(reservation);
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
        BooleanSupplier publishSuccess = fixture.delivery().publication().claimDeliveryResponse(success);
        assertFalse(fixture.slot().future().isDone());
        CompletableFuture<Void> callback = fixture.slot().future().thenAccept(response ->
                assertFalse(Thread.holdsLock(fixture.slot())));
        synchronized (fixture.slot()) {
            fixture.slot().markCancellationRequested(CancelReason.DEADLINE_EXCEEDED, "request inactive");
            TerminalAction terminal = fixture.slot().beginTerminalizing(true, false, false, null,
                    owner -> owner.timeout("request inactive"), new Response());
            assertNotNull(terminal);
            assertNull(terminal.publication(), "an already selected delivery owns the frontend result");
            assertEquals(RequestState.Phase.TIMED_OUT,
                    fixture.slot().finishTombstone(terminal).terminal().state());
        }
        assertTrue(publishSuccess.getAsBoolean());
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
            terminal = fixture.slot().beginTerminalizing(true, false, false, null,
                    owner -> owner.fail("worker failed before response publication"), failure);
            assertNotNull(terminal.publication());
            fixture.slot().finishTombstone(terminal);
        }
        Response success = new Response();
        success.setSuccess(true);
        assertFalse(fixture.delivery().publication().claimDeliveryResponse(success).getAsBoolean());
        assertFalse(fixture.slot().future().isDone());
        CompletableFuture<Void> callback = fixture.slot().future().handle((response, error) -> {
            assertFalse(Thread.holdsLock(fixture.slot()));
            return null;
        });
        BooleanSupplier publication = switch (form) {
            case RESPONSE -> terminal.publication().claimTerminalResponse(failure);
            case FAILURE -> terminal.publication().claimFailure(new IllegalStateException("worker failed"));
            case CANCELLATION -> terminal.publication().claimCancellation(false);
        };
        assertTrue(publication.getAsBoolean());
        callback.join();
        assertTrue(fixture.slot().future().isDone());
        if (form == TerminalForm.RESPONSE) {
            assertSame(failure, fixture.slot().future().join());
        } else {
            assertTrue(fixture.slot().future().isCompletedExceptionally());
            assertEquals(form == TerminalForm.CANCELLATION, fixture.slot().future().isCancelled());
        }
    }

    private static Fixture fixture() {
        RequestCompletionPublisher publisher = mock(RequestCompletionPublisher.class);
        RequestSlot slot = new RequestSlot(publisher, 701L);
        when(publisher.tryReservePublication(eq(slot), any())).thenAnswer(invocation ->
                new RequestSlot.PublicationPermit(publisher, slot, invocation.getArgument(1)));
        var config = SchedulingTestConfig.batchConfig();
        BalanceContext context = RequestLifecycleTestSupport.context(config, slot.requestId());
        ScheduledRequest item = new ScheduledRequest(context, slot.future(), new Response(), null, null,
                null, null, null, slot.createdAtMs());
        RequestSlot.DeliveryConfirmation delivery;
        synchronized (slot) {
            AdmissionMutation admission = slot.tryBeginAdmissionMutation((owner, response) -> { }, owner -> { });
            assertNotNull(admission);
            assertTrue(slot.tryBindItemForPublication(item));
            slot.completeAdmissionMutation(admission);
            slot.startBatchEnqueue(801L);
            delivery = slot.confirmDeliveryForPublication(item, DeliveryClaimKind.BATCH_ENQUEUE, 801L);
            assertNotNull(delivery);
        }
        return new Fixture(slot, delivery);
    }

    private enum TerminalForm { RESPONSE, FAILURE, CANCELLATION }

    private record Fixture(RequestSlot slot, RequestSlot.DeliveryConfirmation delivery) { }
}
