package org.flexlb.balance.scheduler.priority;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.scheduler.BatchItem;
import org.flexlb.balance.scheduler.PrefillQueueManager;
import org.flexlb.balance.scheduler.SchedulingTestConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentMatchers;
import org.mockito.Mockito;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ScheduledFuture;
import java.util.concurrent.ScheduledThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Post-delivery ownership tests for {@link AdmissionLease}.
 *
 * <p>The lease owns only admission backpressure after delivery succeeds. It
 * must never infer Engine absence and free ledgers itself; the registrar's
 * request-scoped Engine fence owns that reconciliation.</p>
 */
class AdmissionLeasePostSuccessTest {

    private ScheduledThreadPoolExecutor timeoutExecutor;
    private AdmissionLease.SoftTimeoutScheduler timeoutScheduler;

    @BeforeEach
    void setUpTimeoutScheduler() {
        timeoutExecutor = new ScheduledThreadPoolExecutor(1);
        timeoutExecutor.setRemoveOnCancelPolicy(true);
        timeoutExecutor.setExecuteExistingDelayedTasksAfterShutdownPolicy(false);
        timeoutScheduler = (lease, task, delay, unit) ->
                timeoutExecutor.schedule(task, delay, unit);
    }

    @AfterEach
    void shutdownTimeoutScheduler() {
        timeoutExecutor.shutdownNow();
    }

    @Test
    void softTimeoutDelegatesEngineFenceWithoutReleasingAnyLedger() {
        InflightRegistrar registrar = mock(InflightRegistrar.class);
        DecodeEndpoint decode = mock(DecodeEndpoint.class);
        PrefillQueueManager queue = mock(PrefillQueueManager.class);
        CompletableFuture<Response> future = new CompletableFuture<>();
        BatchItem item = batchItem("3001", future, decode);
        when(registrar.fenceAfterDeliveryTimeout(item, "post_delivery_soft_timeout"))
                .thenReturn(InflightRegistrar.PostDeliveryFenceResult.STARTED);

        AdmissionLease lease = new AdmissionLease(
                item, decode, queue, registrar, 20, null, timeoutScheduler);
        lease.bindTo(future);
        future.complete(successResponse());

        verify(registrar, Mockito.timeout(1_000).times(1))
                .fenceAfterDeliveryTimeout(item, "post_delivery_soft_timeout");
        verify(queue, never()).tryRemove(ArgumentMatchers.anyString(), anyString());
        verify(decode, never()).release(ArgumentMatchers.anyString());
        verify(registrar, never()).unregisterInflight(any());
        verify(registrar, never()).finishYieldedById(ArgumentMatchers.anyString(), anyString());
    }

    @Test
    void schedulerEngineOwnedResultClosesDiagnosticStateWithoutCleanup() {
        InflightRegistrar registrar = mock(InflightRegistrar.class);
        DecodeEndpoint decode = mock(DecodeEndpoint.class);
        PrefillQueueManager queue = mock(PrefillQueueManager.class);
        BatchItem item = batchItem("3002", new CompletableFuture<>(), decode);
        when(registrar.fenceAfterDeliveryTimeout(item, "post_delivery_soft_timeout"))
                .thenReturn(InflightRegistrar.PostDeliveryFenceResult.ENGINE_OWNED);
        AtomicInteger active = new AtomicInteger(1);
        AdmissionLease lease = new AdmissionLease(
                item, decode, queue, registrar, 0,
                active::decrementAndGet, null);

        lease.markDeliverySucceeded();
        lease.reconcileAfterDeliveryTimeout();

        assertEquals(3, lease.leaseState());
        assertEquals(0, active.get());
        verify(queue, never()).tryRemove(ArgumentMatchers.anyString(), anyString());
        verify(decode, never()).release(ArgumentMatchers.anyString());
        verify(registrar, never()).unregisterInflight(any());
    }

    @Test
    void successfulDeliveryWithoutDecodeClosesAdmissionImmediately() {
        InflightRegistrar registrar = mock(InflightRegistrar.class);
        CompletableFuture<Response> future = new CompletableFuture<>();
        BatchItem item = batchItem("3009", future, null);
        AtomicInteger activeAdmissions = new AtomicInteger(1);
        AtomicInteger scheduledTimeouts = new AtomicInteger();
        AdmissionLease.SoftTimeoutScheduler scheduler =
                (lease, task, delay, unit) -> {
                    scheduledTimeouts.incrementAndGet();
                    return timeoutExecutor.schedule(task, delay, unit);
                };
        AdmissionLease lease = new AdmissionLease(
                item, null, null, registrar, 60_000,
                activeAdmissions::decrementAndGet, scheduler);
        lease.bindTo(future);

        future.complete(successResponse());
        lease.markDeliverySucceeded();
        lease.markDecodeAccepted();

        assertEquals(3, lease.leaseState());
        assertEquals(0, activeAdmissions.get());
        assertEquals(0, scheduledTimeouts.get());
        verify(registrar, never()).fenceAfterDeliveryTimeout(any(), anyString());
        verify(registrar, never()).unregisterInflight(any());
    }

    @Test
    void deliveryTimeoutReconciliationIsIdempotentAndNotifiesBackpressureExactlyOnce() {
        InflightRegistrar registrar = mock(InflightRegistrar.class);
        DecodeEndpoint decode = mock(DecodeEndpoint.class);
        BatchItem item = batchItem("3003", new CompletableFuture<>(), decode);
        when(registrar.fenceAfterDeliveryTimeout(item, "post_delivery_soft_timeout"))
                .thenReturn(InflightRegistrar.PostDeliveryFenceResult.STARTED);
        AtomicInteger active = new AtomicInteger(1);
        AdmissionLease lease = new AdmissionLease(
                item, decode, null, registrar, 0,
                active::decrementAndGet, null);
        lease.markDeliverySucceeded();

        lease.reconcileAfterDeliveryTimeout();
        lease.reconcileAfterDeliveryTimeout();
        lease.reconcileAfterDeliveryTimeout();

        assertEquals(0, active.get());
        verify(registrar, times(1))
                .fenceAfterDeliveryTimeout(item, "post_delivery_soft_timeout");
        verify(decode, never()).release(ArgumentMatchers.anyString());
    }

    @Test
    void fenceExceptionStillClosesBackpressureSlotWithoutUnsafeFallback() {
        InflightRegistrar registrar = mock(InflightRegistrar.class);
        DecodeEndpoint decode = mock(DecodeEndpoint.class);
        PrefillQueueManager queue = mock(PrefillQueueManager.class);
        BatchItem item = batchItem("3004", new CompletableFuture<>(), decode);
        doThrow(new IllegalStateException("control plane failed"))
                .when(registrar)
                .fenceAfterDeliveryTimeout(item, "post_delivery_soft_timeout");
        AtomicInteger active = new AtomicInteger(1);
        AdmissionLease lease = new AdmissionLease(
                item, decode, queue, registrar, 0,
                active::decrementAndGet, null);
        lease.markDeliverySucceeded();

        lease.reconcileAfterDeliveryTimeout();

        assertEquals(0, active.get());
        verify(queue, never()).tryRemove(ArgumentMatchers.anyString(), anyString());
        verify(decode, never()).release(ArgumentMatchers.anyString());
        verify(registrar, never()).unregisterInflight(any());
    }

    @Test
    void canceledSoftTimeoutIsRemovedFromOwningExecutorQueue() throws Exception {
        assertTrue(timeoutExecutor.getRemoveOnCancelPolicy());
        InflightRegistrar registrar = mock(InflightRegistrar.class);
        DecodeEndpoint decode = mock(DecodeEndpoint.class);
        BatchItem item = batchItem("3005", new CompletableFuture<>(), decode);
        when(registrar.fenceAfterDeliveryTimeout(item, "post_delivery_soft_timeout"))
                .thenReturn(InflightRegistrar.PostDeliveryFenceResult.STARTED);
        int before = timeoutExecutor.getQueue().size();
        AdmissionLease lease = new AdmissionLease(
                item, decode, null, registrar, TimeUnit.MINUTES.toMillis(1),
                null, timeoutScheduler);

        lease.markDeliverySucceeded();
        awaitQueueSize(before + 1);
        lease.reconcileAfterDeliveryTimeout();
        awaitQueueSize(before);

        assertEquals(before, timeoutExecutor.getQueue().size());
    }

    @Test
    void decodeAcceptanceCancelsTimeoutWithoutReleasingResources() {
        InflightRegistrar registrar = mock(InflightRegistrar.class);
        DecodeEndpoint decode = mock(DecodeEndpoint.class);
        PrefillQueueManager queue = mock(PrefillQueueManager.class);
        BatchItem item = batchItem("3006", new CompletableFuture<>(), decode);
        AtomicInteger active = new AtomicInteger(1);
        AdmissionLease lease = new AdmissionLease(
                item, decode, queue, registrar, 60_000,
                active::decrementAndGet, timeoutScheduler);
        lease.markDeliverySucceeded();

        lease.markDecodeAccepted();

        assertEquals(3, lease.leaseState());
        assertEquals(0, active.get());
        verify(registrar, never()).fenceAfterDeliveryTimeout(any(), anyString());
        verify(queue, never()).tryRemove(ArgumentMatchers.anyString(), anyString());
        verify(decode, never()).release(ArgumentMatchers.anyString());
    }

    @Test
    void authoritativeSettlementClosesDeliveryWaitWithoutEndpointCleanup()
            throws Exception {
        InflightRegistrar registrar = mock(InflightRegistrar.class);
        DecodeEndpoint decode = mock(DecodeEndpoint.class);
        PrefillQueueManager queue = mock(PrefillQueueManager.class);
        BatchItem item = batchItem("3010", new CompletableFuture<>(), decode);
        AtomicInteger activeAdmissions = new AtomicInteger(1);
        int baseline = timeoutExecutor.getQueue().size();
        AdmissionLease lease = new AdmissionLease(
                item, decode, queue, registrar, TimeUnit.MINUTES.toMillis(1),
                activeAdmissions::decrementAndGet, timeoutScheduler);
        lease.markDeliverySucceeded();
        awaitQueueSize(baseline + 1);

        lease.markRequestSettled();
        lease.markRequestSettled();
        lease.reconcileAfterDeliveryTimeout();
        awaitQueueSize(baseline);

        assertEquals(2, lease.leaseState());
        assertEquals(0, activeAdmissions.get());
        assertEquals(baseline, timeoutExecutor.getQueue().size());
        verify(registrar, never()).fenceAfterDeliveryTimeout(any(), anyString());
        verify(registrar, never()).unregisterInflight(any());
        verify(queue, never()).tryRemove(ArgumentMatchers.anyString(), anyString());
        verify(decode, never()).release(ArgumentMatchers.anyString());
    }

    @Test
    void decodeAcceptanceBetweenScheduleAndHandlePublicationCancelsRetainedTask()
            throws Exception {
        InflightRegistrar registrar = mock(InflightRegistrar.class);
        DecodeEndpoint decode = mock(DecodeEndpoint.class);
        BatchItem item = batchItem("3008", new CompletableFuture<>(), decode);
        int baseline = timeoutExecutor.getQueue().size();
        CountDownLatch scheduled = new CountDownLatch(1);
        CountDownLatch allowScheduleReturn = new CountDownLatch(1);
        AtomicReference<ScheduledFuture<?>> taskRef = new AtomicReference<>();
        AdmissionLease.SoftTimeoutScheduler blockedReturn =
                (scheduledLease, task, delay, unit) -> {
                    ScheduledFuture<?> timeout = timeoutScheduler.schedule(
                            scheduledLease, task, delay, unit);
                    taskRef.set(timeout);
                    scheduled.countDown();
                    try {
                        if (!allowScheduleReturn.await(1, TimeUnit.SECONDS)) {
                            throw new IllegalStateException("schedule return barrier timed out");
                        }
                    } catch (InterruptedException interrupted) {
                        Thread.currentThread().interrupt();
                        throw new IllegalStateException(interrupted);
                    }
                    return timeout;
                };
        AdmissionLease lease = new AdmissionLease(
                item, decode, null, registrar, TimeUnit.MINUTES.toMillis(1),
                null, blockedReturn);

        Thread delivery = Thread.ofVirtual().start(lease::markDeliverySucceeded);
        assertTrue(scheduled.await(1, TimeUnit.SECONDS));
        awaitQueueSize(baseline + 1);
        lease.markDecodeAccepted();
        allowScheduleReturn.countDown();
        delivery.join(TimeUnit.SECONDS.toMillis(1));
        awaitQueueSize(baseline);

        assertEquals(3, lease.leaseState());
        assertTrue(taskRef.get().isCancelled());
        assertEquals(baseline, timeoutExecutor.getQueue().size());
        verify(registrar, never()).fenceAfterDeliveryTimeout(any(), anyString());
    }

    @Test
    void failedDeliveryStillUsesPreHandoffDirectCleanup() {
        InflightRegistrar registrar = mock(InflightRegistrar.class);
        DecodeEndpoint decode = mock(DecodeEndpoint.class);
        PrefillQueueManager queue = mock(PrefillQueueManager.class);
        CompletableFuture<Response> future = new CompletableFuture<>();
        BatchItem item = batchItem("3007", future, decode);
        AdmissionLease lease = new AdmissionLease(
                item, decode, queue, registrar, 20, null, timeoutScheduler);
        lease.bindTo(future);

        future.complete(Response.error(StrategyErrorType.NO_AVAILABLE_WORKER));

        verify(queue, times(1)).tryRemove("3007", "LEASE_RELEASE");
        verify(decode, times(1)).release("3007");
        verify(registrar, times(1)).unregisterInflight(item);
        verify(registrar, never()).fenceAfterDeliveryTimeout(any(), anyString());
    }

    private void awaitQueueSize(int expected) throws InterruptedException {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(1);
        while (timeoutExecutor.getQueue().size() != expected
                && System.nanoTime() < deadline) {
            Thread.sleep(1);
        }
        assertEquals(expected, timeoutExecutor.getQueue().size());
    }

    private static BatchItem batchItem(String requestId,
                                       CompletableFuture<Response> future,
                                       DecodeEndpoint decodeEndpoint) {
        BalanceContext context = new BalanceContext();
        context.setConfig(SchedulingTestConfig.batchConfig());
        Request request = new Request();
        request.setRequestId(requestId);
        context.setRequest(request);

        ServerStatus prefill = new ServerStatus();
        prefill.setRequestId(requestId);
        ServerStatus decode = null;
        if (decodeEndpoint != null) {
            decode = new ServerStatus();
            decode.setRequestId(requestId);
        }
        return new BatchItem(context, future, new Response(), prefill, decode,
                null, decodeEndpoint, System.currentTimeMillis());
    }

    private static Response successResponse() {
        Response response = new Response();
        response.setSuccess(true);
        return response;
    }
}
