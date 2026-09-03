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
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentMatchers;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * PR-D §2.1: unit tests for {@link AdmissionLease} — the CAS-guarded
 * single ownership boundary between the admission scheduler and the
 * dispatch/completion pipeline.
 *
 * <p>Core invariants under test:
 * <ol>
 *   <li>close() is exactly-once — a second call is a no-op;</li>
 *   <li>markDeliverySucceeded() is exactly-once — a second call is a no-op;</li>
 *   <li>close() and markDeliverySucceeded() are mutually exclusive — whichever
 *       runs first seals the lease, the other is a no-op;</li>
 *   <li>bindTo(future) on success → markDeliverySucceeded (no resource release);</li>
 *   <li>bindTo(future) on failure → close (all resources released);</li>
 *   <li>bindTo(future) on exceptional completion → close;</li>
 *   <li>null decodeEp / null prefillQueue are safe (skip the corresponding
 *       cleanup step, still execute the remaining steps).</li>
 *   <li>onCloseCallback is decremented exactly once on every terminal path:
 *       close(), reconcileAfterDeliveryTimeout(), markDecodeAccepted();</li>
 *   <li>onCloseCallback is called even if releaseResources() throws
 *       (try-finally guarantee).</li>
 * </ol>
 */
class AdmissionLeaseTest {

    // ==================== close() exactly-once ====================

    @Test
    void close_is_exactly_once() {
        InflightRegistrar registrar = mock(InflightRegistrar.class);
        BatchItem item = batchItem("1001", new CompletableFuture<>());
        AdmissionLease lease = new AdmissionLease(item, null, null, registrar,
                0, null, null);

        lease.close();
        lease.close();

        verify(registrar, times(1)).unregisterInflight(item);
    }

    // ==================== markDeliverySucceeded() exactly-once ====================

    @Test
    void markDeliverySucceeded_is_exactly_once() {
        InflightRegistrar registrar = mock(InflightRegistrar.class);
        BatchItem item = batchItem("1002", new CompletableFuture<>());
        AdmissionLease lease = new AdmissionLease(item, null, null, registrar,
                0, null, null);

        lease.markDeliverySucceeded();
        lease.markDeliverySucceeded();

        verify(registrar, never()).unregisterInflight(any());
    }

    // ==================== mutual exclusivity ====================

    @Test
    void close_then_delivery_is_noop() {
        InflightRegistrar registrar = mock(InflightRegistrar.class);
        BatchItem item = batchItem("1003", new CompletableFuture<>());
        AdmissionLease lease = new AdmissionLease(item, null, null, registrar,
                0, null, null);

        lease.close();
        lease.markDeliverySucceeded();

        verify(registrar, times(1)).unregisterInflight(item);
    }

    @Test
    void prefill_only_delivery_closes_engine_owned_and_close_is_noop() {
        InflightRegistrar registrar = mock(InflightRegistrar.class);
        BatchItem item = batchItem("1004", new CompletableFuture<>());
        AdmissionLease lease = new AdmissionLease(item, null, null, registrar,
                0, null, null);

        lease.markDeliverySucceeded();
        lease.close();

        // No Decode observation can arrive for a Prefill-only plan. Successful
        // delivery is therefore terminal and must release the admission permit
        // without scheduling reconciliation or releasing engine resources.
        verify(registrar, never()).unregisterInflight(any());
        assertEquals(3, lease.leaseState()); // CLOSED_ENGINE_OWNED
    }

    // ==================== bindTo: success → delivery ====================

    @Test
    void bindTo_success_completes_delivery_not_close() {
        InflightRegistrar registrar = mock(InflightRegistrar.class);
        CompletableFuture<Response> future = new CompletableFuture<>();
        BatchItem item = batchItem("1005", future);
        AdmissionLease lease = new AdmissionLease(item, null, null, registrar,
                0, null, null);
        lease.bindTo(future);

        future.complete(successResponse("1005"));

        // delivery seals the lease — close (unregisterInflight) must not run
        verify(registrar, never()).unregisterInflight(any());
    }

    // ==================== bindTo: failure → close ====================

    @Test
    void bindTo_failure_completes_close() {
        InflightRegistrar registrar = mock(InflightRegistrar.class);
        CompletableFuture<Response> future = new CompletableFuture<>();
        BatchItem item = batchItem("1006", future);
        AdmissionLease lease = new AdmissionLease(item, null, null, registrar,
                0, null, null);
        lease.bindTo(future);

        future.complete(failedResponse());

        verify(registrar, times(1)).unregisterInflight(item);
    }

    @Test
    void bindTo_exceptional_completes_close() {
        InflightRegistrar registrar = mock(InflightRegistrar.class);
        CompletableFuture<Response> future = new CompletableFuture<>();
        BatchItem item = batchItem("1007", future);
        AdmissionLease lease = new AdmissionLease(item, null, null, registrar,
                0, null, null);
        lease.bindTo(future);

        future.completeExceptionally(new RuntimeException("dispatch error"));

        verify(registrar, times(1)).unregisterInflight(item);
    }

    // ==================== close releases all resources ====================

    @Test
    void close_calls_tryRemove_release_and_unregister() {
        InflightRegistrar registrar = mock(InflightRegistrar.class);
        DecodeEndpoint decodeEp = mock(DecodeEndpoint.class);
        PrefillQueueManager prefillQueue = mock(PrefillQueueManager.class);
        CompletableFuture<Response> future = new CompletableFuture<>();
        BatchItem item = batchItemWithDecode("1008", future, "1008");
        AdmissionLease lease = new AdmissionLease(item, decodeEp, prefillQueue, registrar,
                0, null, null);

        lease.close();

        verify(prefillQueue, times(1)).tryRemove("1008", "LEASE_RELEASE");
        verify(decodeEp, times(1)).release("1008");
        verify(registrar, times(1)).unregisterInflight(item);
    }

    // ==================== null decodeEp / null prefillQueue safe ====================

    @Test
    void close_with_null_decodeEp_skips_release_but_still_unregisters() {
        InflightRegistrar registrar = mock(InflightRegistrar.class);
        PrefillQueueManager prefillQueue = mock(PrefillQueueManager.class);
        CompletableFuture<Response> future = new CompletableFuture<>();
        BatchItem item = batchItemWithDecode("1009", future, "1009");
        AdmissionLease lease = new AdmissionLease(item, null, prefillQueue, registrar,
                0, null, null);

        lease.close();

        verify(prefillQueue, times(1)).tryRemove("1009", "LEASE_RELEASE");
        verify(registrar, times(1)).unregisterInflight(item);
    }

    @Test
    void close_with_null_prefillQueue_skips_tryRemove_but_still_releases_and_unregisters() {
        InflightRegistrar registrar = mock(InflightRegistrar.class);
        DecodeEndpoint decodeEp = mock(DecodeEndpoint.class);
        CompletableFuture<Response> future = new CompletableFuture<>();
        BatchItem item = batchItemWithDecode("1010", future, "1010");
        AdmissionLease lease = new AdmissionLease(item, decodeEp, null, registrar,
                0, null, null);

        lease.close();

        verify(decodeEp, times(1)).release("1010");
        verify(registrar, times(1)).unregisterInflight(item);
    }

    @Test
    void close_with_null_decode_in_item_skips_release() {
        InflightRegistrar registrar = mock(InflightRegistrar.class);
        DecodeEndpoint decodeEp = mock(DecodeEndpoint.class);
        CompletableFuture<Response> future = new CompletableFuture<>();
        // item with decode = null
        BatchItem item = batchItem("1011", future);
        AdmissionLease lease = new AdmissionLease(item, decodeEp, null, registrar,
                0, null, null);

        lease.close();

        verify(decodeEp, never()).release(ArgumentMatchers.anyString());
        verify(registrar, times(1)).unregisterInflight(item);
    }

    // ==================== onCloseCallback counter guarantee ====================

    /**
     * onCloseCallback must decrement on the reconcileAfterDeliveryTimeout path:
     * create (increment) → delivery → reconcile (decrement).
     */
    @Test
    void onCloseCallback_decrements_on_reconcileAfterDeliveryTimeout() {
        AtomicInteger activeCount = new AtomicInteger(0);
        InflightRegistrar registrar = mock(InflightRegistrar.class);
        DecodeEndpoint decodeEp = mock(DecodeEndpoint.class);
        PrefillQueueManager prefillQueue = mock(PrefillQueueManager.class);
        BatchItem item = batchItemWithDecode("2001", new CompletableFuture<>(), "2001");

        when(registrar.fenceAfterDeliveryTimeout(item, "post_delivery_soft_timeout"))
                .thenReturn(InflightRegistrar.PostDeliveryFenceResult.STARTED);

        activeCount.incrementAndGet();
        AdmissionLease lease = new AdmissionLease(item, decodeEp, prefillQueue, registrar,
                0, activeCount::decrementAndGet, null);
        lease.markDeliverySucceeded();
        assertEquals(1, lease.leaseState()); // DELIVERY_WAIT
        assertEquals(1, activeCount.get());

        lease.reconcileAfterDeliveryTimeout();

        assertEquals(2, lease.leaseState()); // CLOSED
        assertEquals(0, activeCount.get()); // counter decremented
        // Scheduler owns the Engine fence; the lease cannot release ledgers.
        verify(registrar, times(1))
                .fenceAfterDeliveryTimeout(item, "post_delivery_soft_timeout");
        verify(registrar, never()).unregisterInflight(item);
        verify(decodeEp, never()).release(ArgumentMatchers.anyString());
    }

    /**
     * onCloseCallback must decrement on the markDecodeAccepted path:
     * create (increment) → delivery → decodeAccept (decrement).
     */
    @Test
    void onCloseCallback_decrements_on_markDecodeAccepted() {
        AtomicInteger activeCount = new AtomicInteger(0);
        InflightRegistrar registrar = mock(InflightRegistrar.class);
        DecodeEndpoint decodeEp = mock(DecodeEndpoint.class);
        PrefillQueueManager prefillQueue = mock(PrefillQueueManager.class);
        BatchItem item = batchItemWithDecode("2002", new CompletableFuture<>(), "2002");

        activeCount.incrementAndGet();
        AdmissionLease lease = new AdmissionLease(item, decodeEp, prefillQueue, registrar,
                0, activeCount::decrementAndGet, null);
        lease.markDeliverySucceeded();
        assertEquals(1, activeCount.get());

        lease.markDecodeAccepted();

        assertEquals(3, lease.leaseState()); // CLOSED_ENGINE_OWNED
        assertEquals(0, activeCount.get()); // counter decremented
        // Resources NOT released (engine owns them)
        verify(registrar, never()).unregisterInflight(any());
        verify(registrar, never()).finishYieldedById(ArgumentMatchers.anyString(), anyString());
    }

    /**
     * CAS mutex: close() (UNSET→CLOSED) then reconcileAfterDeliveryTimeout()
     * (DELIVERY_WAIT→CLOSED) — the second CAS fails, onCloseCallback is
     * called exactly once.
     */
    @Test
    void onCloseCallback_only_decrement_once_close_then_reconcile() {
        AtomicInteger activeCount = new AtomicInteger(0);
        InflightRegistrar registrar = mock(InflightRegistrar.class);
        DecodeEndpoint decodeEp = mock(DecodeEndpoint.class);
        PrefillQueueManager prefillQueue = mock(PrefillQueueManager.class);
        BatchItem item = batchItemWithDecode("2003", new CompletableFuture<>(), "2003");

        activeCount.incrementAndGet();
        AdmissionLease lease = new AdmissionLease(item, decodeEp, prefillQueue, registrar,
                0, activeCount::decrementAndGet, null);

        // close() succeeds: UNSET→CLOSED
        lease.close();
        assertEquals(0, activeCount.get()); // counter decremented

        // reconcileAfterDeliveryTimeout() CAS fails (state is CLOSED, not DELIVERY_WAIT)
        lease.reconcileAfterDeliveryTimeout();
        assertEquals(0, activeCount.get()); // NOT decremented again

        // Resources released exactly once (by close())
        verify(registrar, times(1)).unregisterInflight(item);
        verify(registrar, never()).finishYieldedById(ArgumentMatchers.anyString(), anyString());
    }

    /**
     * try-finally guarantee: if releaseResources() throws, onCloseCallback
     * must still be called (counter decremented). Without the try-finally fix,
     * the exception would skip notifyCloseCallback() and leak the counter.
     */
    @Test
    void onCloseCallback_called_even_if_releaseResources_throws() {
        AtomicInteger activeCount = new AtomicInteger(0);
        InflightRegistrar registrar = mock(InflightRegistrar.class);
        PrefillQueueManager prefillQueue = mock(PrefillQueueManager.class);
        BatchItem item = batchItemWithDecode("2004", new CompletableFuture<>(), "2004");

        // Make releaseResources() throw by having prefillQueue.tryRemove throw
        doThrow(new RuntimeException("simulated queue error"))
                .when(prefillQueue).tryRemove("2004", "LEASE_RELEASE");

        activeCount.incrementAndGet();
        AdmissionLease lease = new AdmissionLease(item, null, prefillQueue, registrar,
                0, activeCount::decrementAndGet, null);

        // close() should catch the exception and still call notifyCloseCallback
        lease.close();

        assertEquals(2, lease.leaseState()); // CLOSED — CAS succeeded
        assertEquals(0, activeCount.get()); // counter decremented despite exception
        verify(prefillQueue, times(1)).tryRemove("2004", "LEASE_RELEASE");
    }

    /**
     * try-finally guarantee for reconcileAfterDeliveryTimeout: if the registrar
     * fence delegation throws, onCloseCallback must still be called and the
     * lease must not fall back to an unsafe local release.
     */
    @Test
    void onCloseCallback_called_even_if_engineFenceDelegation_throws() {
        AtomicInteger activeCount = new AtomicInteger(0);
        InflightRegistrar registrar = mock(InflightRegistrar.class);
        DecodeEndpoint decodeEp = mock(DecodeEndpoint.class);
        PrefillQueueManager prefillQueue = mock(PrefillQueueManager.class);
        BatchItem item = batchItemWithDecode("2005", new CompletableFuture<>(), "2005");

        doThrow(new RuntimeException("simulated cancel error"))
                .when(registrar)
                .fenceAfterDeliveryTimeout(item, "post_delivery_soft_timeout");

        activeCount.incrementAndGet();
        AdmissionLease lease = new AdmissionLease(item, decodeEp, prefillQueue, registrar,
                0, activeCount::decrementAndGet, null);
        lease.markDeliverySucceeded();

        // reconcileAfterDeliveryTimeout should catch the exception and still call notifyCloseCallback
        lease.reconcileAfterDeliveryTimeout();

        assertEquals(2, lease.leaseState()); // CLOSED
        assertEquals(0, activeCount.get()); // counter decremented despite exception
        verify(registrar, times(1))
                .fenceAfterDeliveryTimeout(item, "post_delivery_soft_timeout");
        verify(registrar, never()).unregisterInflight(item);
        verify(decodeEp, never()).release(ArgumentMatchers.anyString());
    }

    // ==================== helpers ====================

    private static BatchItem batchItem(String requestId, CompletableFuture<Response> future) {
        return batchItemWithDecode(requestId, future, "0");
    }

    private static BatchItem batchItemWithDecode(String requestId,
                                                  CompletableFuture<Response> future,
                                                  String decodeRequestId) {
        BalanceContext ctx = new BalanceContext();
        ctx.setConfig(SchedulingTestConfig.batchConfig());
        Request request = new Request();
        request.setRequestId(requestId);
        ctx.setRequest(request);

        ServerStatus decode = null;
        if (!"0".equals(decodeRequestId)) {
            decode = new ServerStatus();
            decode.setRequestId(decodeRequestId);
        }

        return new BatchItem(ctx, future, new Response(),
                new ServerStatus(), decode, null, null, System.currentTimeMillis());
    }

    private static Response successResponse(String requestId) {
        Response response = new Response();
        response.setSuccess(true);
        return response;
    }

    private static Response failedResponse() {
        return Response.error(StrategyErrorType.NO_AVAILABLE_WORKER);
    }
}
