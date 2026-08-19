package org.flexlb.balance.scheduler.priority;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.scheduler.BatchItem;
import org.flexlb.balance.scheduler.PrefillQueueManager;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.junit.jupiter.api.Test;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.timeout;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Post-Enqueue ownership tests for {@link AdmissionLease}.
 *
 * <p>Once Enqueue succeeds, local cleanup is no longer safe: a missing Decode
 * observation may mean delayed WorkerStatus rather than failed dispatch. The
 * timeout therefore asks the scheduler to reconcile against the Engine while
 * retaining queue, Decode and inflight accounting. Only authoritative Decode
 * acceptance or a terminal reconciliation fence retires the active lease.
 */
class AdmissionLeasePostSuccessTest {

    private static final long SOFT_TIMEOUT_MS = 30L;
    private static final long VERIFY_TIMEOUT_MS = 500L;

    @Test
    void softTimeout_requestsReconciliation_andRetainsResourcesUntilTerminalFence() {
        AtomicInteger activeCount = new AtomicInteger(1);
        InflightRegistrar registrar = mock(InflightRegistrar.class);
        DecodeEndpoint decodeEp = mock(DecodeEndpoint.class);
        PrefillQueueManager prefillQueue = mock(PrefillQueueManager.class);
        CompletableFuture<Response> future = new CompletableFuture<>();
        BatchItem item = batchItem(3001L, future);

        when(registrar.requestPostHandoverReconciliation(
                item, "post_success_soft_timeout")).thenReturn(true);
        AdmissionLease lease = new AdmissionLease(item, decodeEp, prefillQueue,
                registrar, SOFT_TIMEOUT_MS, activeCount::decrementAndGet);
        lease.bindTo(future);

        future.complete(successResponse());

        verify(registrar, timeout(VERIFY_TIMEOUT_MS).times(1))
                .requestPostHandoverReconciliation(item, "post_success_soft_timeout");
        assertEquals(4, lease.leaseState()); // RECONCILING
        assertEquals(1, activeCount.get());
        assertResourcesRetained(item, decodeEp, prefillQueue, registrar);

        lease.completeSchedulerSettlement();

        assertEquals(2, lease.leaseState()); // CLOSED_CLEANUP
        assertEquals(0, activeCount.get());
        assertResourcesRetained(item, decodeEp, prefillQueue, registrar);
    }

    @Test
    void forceCloseAfterHandover_isIdempotent_andCompletionNotifiesOnce() {
        AtomicInteger activeCount = new AtomicInteger(1);
        InflightRegistrar registrar = mock(InflightRegistrar.class);
        DecodeEndpoint decodeEp = mock(DecodeEndpoint.class);
        PrefillQueueManager prefillQueue = mock(PrefillQueueManager.class);
        BatchItem item = batchItem(3002L, new CompletableFuture<>());

        when(registrar.requestPostHandoverReconciliation(
                item, "post_success_soft_timeout")).thenReturn(true);
        AdmissionLease lease = new AdmissionLease(item, decodeEp, prefillQueue,
                registrar, 0, activeCount::decrementAndGet);
        lease.handoverToEngine();

        lease.forceCloseAfterHandover();
        lease.forceCloseAfterHandover();
        lease.completeSchedulerSettlement();
        lease.completeSchedulerSettlement();

        verify(registrar, times(1)).requestPostHandoverReconciliation(
                item, "post_success_soft_timeout");
        assertEquals(2, lease.leaseState());
        assertEquals(0, activeCount.get());
        assertResourcesRetained(item, decodeEp, prefillQueue, registrar);
    }

    @Test
    void authoritativeSchedulerTerminalClosesLeaseFromHandoverWait() {
        AtomicInteger activeCount = new AtomicInteger(1);
        InflightRegistrar registrar = mock(InflightRegistrar.class);
        DecodeEndpoint decodeEp = mock(DecodeEndpoint.class);
        PrefillQueueManager prefillQueue = mock(PrefillQueueManager.class);
        BatchItem item = batchItem(3007L, new CompletableFuture<>());
        AdmissionLease lease = new AdmissionLease(item, decodeEp, prefillQueue,
                registrar, 0, activeCount::decrementAndGet);

        lease.handoverToEngine();
        lease.completeSchedulerSettlement();
        lease.completeSchedulerSettlement();

        assertEquals(2, lease.leaseState()); // CLOSED_CLEANUP
        assertEquals(0, activeCount.get());
        assertResourcesRetained(item, decodeEp, prefillQueue, registrar);
    }

    @Test
    void alreadyGoneAtReconciliationRequest_completesLease_withoutLocalCleanup() {
        AtomicInteger activeCount = new AtomicInteger(1);
        InflightRegistrar registrar = mock(InflightRegistrar.class);
        DecodeEndpoint decodeEp = mock(DecodeEndpoint.class);
        PrefillQueueManager prefillQueue = mock(PrefillQueueManager.class);
        BatchItem item = batchItem(3003L, new CompletableFuture<>());

        when(registrar.requestPostHandoverReconciliation(
                item, "post_success_soft_timeout")).thenReturn(false);
        AdmissionLease lease = new AdmissionLease(item, decodeEp, prefillQueue,
                registrar, 0, activeCount::decrementAndGet);
        lease.handoverToEngine();

        lease.forceCloseAfterHandover();

        assertEquals(2, lease.leaseState());
        assertEquals(0, activeCount.get());
        assertResourcesRetained(item, decodeEp, prefillQueue, registrar);
    }

    @Test
    void decodeAcceptanceDuringReconciliation_closesEngineOwned() {
        AtomicInteger activeCount = new AtomicInteger(1);
        InflightRegistrar registrar = mock(InflightRegistrar.class);
        DecodeEndpoint decodeEp = mock(DecodeEndpoint.class);
        PrefillQueueManager prefillQueue = mock(PrefillQueueManager.class);
        BatchItem item = batchItem(3004L, new CompletableFuture<>());

        when(registrar.requestPostHandoverReconciliation(
                item, "post_success_soft_timeout")).thenReturn(true);
        AdmissionLease lease = new AdmissionLease(item, decodeEp, prefillQueue,
                registrar, 0, activeCount::decrementAndGet);
        lease.handoverToEngine();
        lease.forceCloseAfterHandover();

        lease.markDecodeAccepted();
        lease.completeSchedulerSettlement();

        assertEquals(3, lease.leaseState()); // CLOSED_ENGINE_OWNED
        assertEquals(0, activeCount.get());
        assertResourcesRetained(item, decodeEp, prefillQueue, registrar);
    }

    @Test
    void decodeAcceptedAtTimeout_closesEngineOwned_withoutReconciliation() {
        AtomicInteger activeCount = new AtomicInteger(1);
        InflightRegistrar registrar = mock(InflightRegistrar.class);
        DecodeEndpoint decodeEp = mock(DecodeEndpoint.class);
        PrefillQueueManager prefillQueue = mock(PrefillQueueManager.class);
        CompletableFuture<Response> future = new CompletableFuture<>();
        BatchItem item = batchItem(3005L, future);

        when(decodeEp.isConfirmedTracked(3005L)).thenReturn(true);
        AdmissionLease lease = new AdmissionLease(item, decodeEp, prefillQueue,
                registrar, SOFT_TIMEOUT_MS, activeCount::decrementAndGet);
        lease.bindTo(future);

        future.complete(successResponse());

        verify(decodeEp, timeout(VERIFY_TIMEOUT_MS).atLeastOnce())
                .isConfirmedTracked(3005L);
        assertEquals(3, lease.leaseState());
        assertEquals(0, activeCount.get());
        verify(registrar, never()).requestPostHandoverReconciliation(any(), any());
        assertResourcesRetained(item, decodeEp, prefillQueue, registrar);
    }

    @Test
    void decodeAcceptedBeforeHandover_remainsEngineOwned() {
        AtomicInteger activeCount = new AtomicInteger(1);
        InflightRegistrar registrar = mock(InflightRegistrar.class);
        DecodeEndpoint decodeEp = mock(DecodeEndpoint.class);
        PrefillQueueManager prefillQueue = mock(PrefillQueueManager.class);
        BatchItem item = batchItem(3006L, new CompletableFuture<>());

        AdmissionLease lease = new AdmissionLease(item, decodeEp, prefillQueue,
                registrar, SOFT_TIMEOUT_MS, activeCount::decrementAndGet);

        lease.markDecodeAccepted();
        lease.handoverToEngine();

        assertEquals(3, lease.leaseState());
        assertEquals(0, activeCount.get());
        verify(registrar, never()).requestPostHandoverReconciliation(any(), any());
        assertResourcesRetained(item, decodeEp, prefillQueue, registrar);
    }

    @Test
    void forceClose_doubleChecksDecodeAcceptance_beforeRequestingReconciliation() {
        AtomicInteger activeCount = new AtomicInteger(1);
        InflightRegistrar registrar = mock(InflightRegistrar.class);
        DecodeEndpoint decodeEp = mock(DecodeEndpoint.class);
        PrefillQueueManager prefillQueue = mock(PrefillQueueManager.class);
        BatchItem item = batchItem(3007L, new CompletableFuture<>());

        when(decodeEp.isConfirmedTracked(3007L)).thenReturn(true);
        AdmissionLease lease = new AdmissionLease(item, decodeEp, prefillQueue,
                registrar, 0, activeCount::decrementAndGet);
        lease.handoverToEngine();

        lease.forceCloseAfterHandover();

        assertEquals(3, lease.leaseState());
        assertEquals(0, activeCount.get());
        verify(registrar, never()).requestPostHandoverReconciliation(any(), any());
        assertResourcesRetained(item, decodeEp, prefillQueue, registrar);
    }

    private static void assertResourcesRetained(BatchItem item,
                                                DecodeEndpoint decodeEp,
                                                PrefillQueueManager prefillQueue,
                                                InflightRegistrar registrar) {
        verify(prefillQueue, never()).tryRemove(anyLong(), any());
        verify(decodeEp, never()).release(anyLong());
        verify(registrar, never()).unregisterInflight(item);
        verify(registrar, never()).finishYieldedById(anyLong(), any());
    }

    private static BatchItem batchItem(long requestId,
                                       CompletableFuture<Response> future) {
        BalanceContext ctx = new BalanceContext();
        Request request = new Request();
        request.setRequestId(requestId);
        ctx.setRequest(request);

        ServerStatus decode = new ServerStatus();
        decode.setRequestId(requestId);
        return new BatchItem(ctx, future, new Response(), new ServerStatus(),
                decode, null, null, System.currentTimeMillis());
    }

    private static Response successResponse() {
        Response response = new Response();
        response.setSuccess(true);
        return response;
    }
}
