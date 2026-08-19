package org.flexlb.balance.scheduler.priority;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.scheduler.BatchItem;
import org.flexlb.balance.scheduler.PrefillQueueManager;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.flexlb.dao.loadbalance.StrategyErrorType;
import org.junit.jupiter.api.Test;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
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
 *   <li>handoverToEngine() is exactly-once — a second call is a no-op;</li>
 *   <li>close() and handoverToEngine() are mutually exclusive — whichever
 *       runs first seals the lease, the other is a no-op;</li>
 *   <li>bindTo(future) on success → handoverToEngine (no resource release);</li>
 *   <li>bindTo(future) on failure → close (all resources released);</li>
 *   <li>bindTo(future) on exceptional completion → close;</li>
 *   <li>null decodeEp / null prefillQueue are safe (skip the corresponding
 *       cleanup step, still execute the remaining steps).</li>
 *   <li>onCloseCallback is decremented exactly once on every terminal path:
 *       close(), forceCloseAfterHandover(), markDecodeAccepted();</li>
 *   <li>onCloseCallback is called even if releaseResources() throws
 *       (try-finally guarantee).</li>
 * </ol>
 */
class AdmissionLeaseTest {

    // ==================== close() exactly-once ====================

    @Test
    void close_is_exactly_once() {
        InflightRegistrar registrar = mock(InflightRegistrar.class);
        BatchItem item = batchItem(1001L, new CompletableFuture<>());
        AdmissionLease lease = new AdmissionLease(item, null, null, registrar);

        lease.close();
        lease.close();

        verify(registrar, times(1)).unregisterInflight(item);
    }

    // ==================== handoverToEngine() exactly-once ====================

    @Test
    void handoverToEngine_is_exactly_once() {
        InflightRegistrar registrar = mock(InflightRegistrar.class);
        BatchItem item = batchItem(1002L, new CompletableFuture<>());
        AdmissionLease lease = new AdmissionLease(item, null, null, registrar);

        lease.handoverToEngine();
        lease.handoverToEngine();

        verify(registrar, never()).unregisterInflight(any());
    }

    // ==================== mutual exclusivity ====================

    @Test
    void close_then_handover_is_noop() {
        InflightRegistrar registrar = mock(InflightRegistrar.class);
        BatchItem item = batchItem(1003L, new CompletableFuture<>());
        AdmissionLease lease = new AdmissionLease(item, null, null, registrar);

        lease.close();
        lease.handoverToEngine();

        verify(registrar, times(1)).unregisterInflight(item);
    }

    /**
     * close() from HANDED_OVER is now a no-op (Warning 2 fix). Post-handover
     * cleanup routes through forceCloseAfterHandover() or markDecodeAccepted()
     * only.
     */
    @Test
    void handover_then_close_is_noop() {
        InflightRegistrar registrar = mock(InflightRegistrar.class);
        BatchItem item = batchItem(1004L, new CompletableFuture<>());
        AdmissionLease lease = new AdmissionLease(item, null, null, registrar);

        lease.handoverToEngine();
        lease.close();

        // close() from HANDED_OVER is a no-op — no resource release
        verify(registrar, never()).unregisterInflight(any());
        assertEquals(1, lease.leaseState()); // still HANDED_OVER
    }

    // ==================== bindTo: success → handover ====================

    @Test
    void bindTo_success_completes_handover_not_close() {
        InflightRegistrar registrar = mock(InflightRegistrar.class);
        CompletableFuture<Response> future = new CompletableFuture<>();
        BatchItem item = batchItem(1005L, future);
        AdmissionLease lease = new AdmissionLease(item, null, null, registrar);
        lease.bindTo(future);

        future.complete(successResponse(1005L));

        // handover seals the lease — close (unregisterInflight) must not run
        verify(registrar, never()).unregisterInflight(any());
    }

    // ==================== bindTo: failure → close ====================

    @Test
    void bindTo_failure_completes_close() {
        InflightRegistrar registrar = mock(InflightRegistrar.class);
        CompletableFuture<Response> future = new CompletableFuture<>();
        BatchItem item = batchItem(1006L, future);
        AdmissionLease lease = new AdmissionLease(item, null, null, registrar);
        lease.bindTo(future);

        future.complete(failedResponse());

        verify(registrar, times(1)).unregisterInflight(item);
    }

    @Test
    void bindTo_exceptional_completes_close() {
        InflightRegistrar registrar = mock(InflightRegistrar.class);
        CompletableFuture<Response> future = new CompletableFuture<>();
        BatchItem item = batchItem(1007L, future);
        AdmissionLease lease = new AdmissionLease(item, null, null, registrar);
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
        BatchItem item = batchItemWithDecode(1008L, future, 1008L);
        AdmissionLease lease = new AdmissionLease(item, decodeEp, prefillQueue, registrar);

        lease.close();

        verify(prefillQueue, times(1)).tryRemove(1008L, "LEASE_RELEASE");
        verify(decodeEp, times(1)).release(1008L);
        verify(registrar, times(1)).unregisterInflight(item);
    }

    // ==================== null decodeEp / null prefillQueue safe ====================

    @Test
    void close_with_null_decodeEp_skips_release_but_still_unregisters() {
        InflightRegistrar registrar = mock(InflightRegistrar.class);
        PrefillQueueManager prefillQueue = mock(PrefillQueueManager.class);
        CompletableFuture<Response> future = new CompletableFuture<>();
        BatchItem item = batchItemWithDecode(1009L, future, 1009L);
        AdmissionLease lease = new AdmissionLease(item, null, prefillQueue, registrar);

        lease.close();

        verify(prefillQueue, times(1)).tryRemove(1009L, "LEASE_RELEASE");
        verify(registrar, times(1)).unregisterInflight(item);
    }

    @Test
    void close_with_null_prefillQueue_skips_tryRemove_but_still_releases_and_unregisters() {
        InflightRegistrar registrar = mock(InflightRegistrar.class);
        DecodeEndpoint decodeEp = mock(DecodeEndpoint.class);
        CompletableFuture<Response> future = new CompletableFuture<>();
        BatchItem item = batchItemWithDecode(1010L, future, 1010L);
        AdmissionLease lease = new AdmissionLease(item, decodeEp, null, registrar);

        lease.close();

        verify(decodeEp, times(1)).release(1010L);
        verify(registrar, times(1)).unregisterInflight(item);
    }

    @Test
    void close_with_null_decode_in_item_skips_release() {
        InflightRegistrar registrar = mock(InflightRegistrar.class);
        DecodeEndpoint decodeEp = mock(DecodeEndpoint.class);
        CompletableFuture<Response> future = new CompletableFuture<>();
        // item with decode = null
        BatchItem item = batchItem(1011L, future);
        AdmissionLease lease = new AdmissionLease(item, decodeEp, null, registrar);

        lease.close();

        verify(decodeEp, never()).release(anyLong());
        verify(registrar, times(1)).unregisterInflight(item);
    }

    // ==================== onCloseCallback counter guarantee ====================

    /**
     * onCloseCallback must decrement on the forceCloseAfterHandover path:
     * create (increment) → handover → forceClose (decrement).
     */
    @Test
    void onCloseCallback_decrements_on_forceCloseAfterHandover() {
        AtomicInteger activeCount = new AtomicInteger(0);
        InflightRegistrar registrar = mock(InflightRegistrar.class);
        DecodeEndpoint decodeEp = mock(DecodeEndpoint.class);
        PrefillQueueManager prefillQueue = mock(PrefillQueueManager.class);
        BatchItem item = batchItemWithDecode(2001L, new CompletableFuture<>(), 2001L);

        when(decodeEp.isConfirmedTracked(2001L)).thenReturn(false);

        activeCount.incrementAndGet();
        AdmissionLease lease = new AdmissionLease(item, decodeEp, prefillQueue, registrar,
                0, activeCount::decrementAndGet);
        lease.handoverToEngine();
        assertEquals(1, lease.leaseState()); // HANDED_OVER
        assertEquals(1, activeCount.get());

        lease.forceCloseAfterHandover();

        assertEquals(2, lease.leaseState()); // CLOSED_CLEANUP
        assertEquals(0, activeCount.get()); // counter decremented
        // Resources released on force-close (decode not accepted)
        verify(registrar, times(1)).unregisterInflight(item);
        verify(registrar, times(1)).finishYieldedById(2001L, "post_success_soft_timeout");
    }

    /**
     * onCloseCallback must decrement on the markDecodeAccepted path:
     * create (increment) → handover → decodeAccept (decrement).
     */
    @Test
    void onCloseCallback_decrements_on_markDecodeAccepted() {
        AtomicInteger activeCount = new AtomicInteger(0);
        InflightRegistrar registrar = mock(InflightRegistrar.class);
        DecodeEndpoint decodeEp = mock(DecodeEndpoint.class);
        PrefillQueueManager prefillQueue = mock(PrefillQueueManager.class);
        BatchItem item = batchItemWithDecode(2002L, new CompletableFuture<>(), 2002L);

        activeCount.incrementAndGet();
        AdmissionLease lease = new AdmissionLease(item, decodeEp, prefillQueue, registrar,
                0, activeCount::decrementAndGet);
        lease.handoverToEngine();
        assertEquals(1, activeCount.get());

        lease.markDecodeAccepted();

        assertEquals(3, lease.leaseState()); // CLOSED_ENGINE_OWNED
        assertEquals(0, activeCount.get()); // counter decremented
        // Resources NOT released (engine owns them)
        verify(registrar, never()).unregisterInflight(any());
        verify(registrar, never()).finishYieldedById(anyLong(), anyString());
    }

    /**
     * CAS mutex: close() (UNSET→CLOSED) then forceCloseAfterHandover()
     * (HANDED_OVER→CLOSED) — the second CAS fails, onCloseCallback is
     * called exactly once.
     */
    @Test
    void onCloseCallback_only_decrement_once_close_then_forceClose() {
        AtomicInteger activeCount = new AtomicInteger(0);
        InflightRegistrar registrar = mock(InflightRegistrar.class);
        DecodeEndpoint decodeEp = mock(DecodeEndpoint.class);
        PrefillQueueManager prefillQueue = mock(PrefillQueueManager.class);
        BatchItem item = batchItemWithDecode(2003L, new CompletableFuture<>(), 2003L);

        activeCount.incrementAndGet();
        AdmissionLease lease = new AdmissionLease(item, decodeEp, prefillQueue, registrar,
                0, activeCount::decrementAndGet);

        // close() succeeds: UNSET→CLOSED
        lease.close();
        assertEquals(0, activeCount.get()); // counter decremented

        // forceCloseAfterHandover() CAS fails (state is CLOSED, not HANDED_OVER)
        lease.forceCloseAfterHandover();
        assertEquals(0, activeCount.get()); // NOT decremented again

        // Resources released exactly once (by close())
        verify(registrar, times(1)).unregisterInflight(item);
        verify(registrar, never()).finishYieldedById(anyLong(), anyString());
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
        BatchItem item = batchItemWithDecode(2004L, new CompletableFuture<>(), 2004L);

        // Make releaseResources() throw by having prefillQueue.tryRemove throw
        doThrow(new RuntimeException("simulated queue error"))
                .when(prefillQueue).tryRemove(2004L, "LEASE_RELEASE");

        activeCount.incrementAndGet();
        AdmissionLease lease = new AdmissionLease(item, null, prefillQueue, registrar,
                0, activeCount::decrementAndGet);

        // close() should catch the exception and still call notifyCloseCallback
        lease.close();

        assertEquals(2, lease.leaseState()); // CLOSED_CLEANUP — CAS succeeded
        assertEquals(0, activeCount.get()); // counter decremented despite exception
        verify(prefillQueue, times(1)).tryRemove(2004L, "LEASE_RELEASE");
    }

    /**
     * try-finally guarantee for forceCloseAfterHandover: if
     * finishYieldedById throws, onCloseCallback must still be called.
     */
    @Test
    void onCloseCallback_called_even_if_finishYieldedById_throws() {
        AtomicInteger activeCount = new AtomicInteger(0);
        InflightRegistrar registrar = mock(InflightRegistrar.class);
        DecodeEndpoint decodeEp = mock(DecodeEndpoint.class);
        PrefillQueueManager prefillQueue = mock(PrefillQueueManager.class);
        BatchItem item = batchItemWithDecode(2005L, new CompletableFuture<>(), 2005L);

        when(decodeEp.isConfirmedTracked(2005L)).thenReturn(false);
        // Make finishYieldedById throw
        doThrow(new RuntimeException("simulated cancel error"))
                .when(registrar).finishYieldedById(2005L, "post_success_soft_timeout");

        activeCount.incrementAndGet();
        AdmissionLease lease = new AdmissionLease(item, decodeEp, prefillQueue, registrar,
                0, activeCount::decrementAndGet);
        lease.handoverToEngine();

        // forceCloseAfterHandover should catch the exception and still call notifyCloseCallback
        lease.forceCloseAfterHandover();

        assertEquals(2, lease.leaseState()); // CLOSED_CLEANUP
        assertEquals(0, activeCount.get()); // counter decremented despite exception
        verify(registrar, times(1)).finishYieldedById(2005L, "post_success_soft_timeout");
    }

    // ==================== helpers ====================

    private static BatchItem batchItem(long requestId, CompletableFuture<Response> future) {
        return batchItemWithDecode(requestId, future, 0L);
    }

    private static BatchItem batchItemWithDecode(long requestId,
                                                  CompletableFuture<Response> future,
                                                  long decodeRequestId) {
        BalanceContext ctx = new BalanceContext();
        Request request = new Request();
        request.setRequestId(requestId);
        ctx.setRequest(request);

        ServerStatus decode = null;
        if (decodeRequestId != 0) {
            decode = new ServerStatus();
            decode.setRequestId(decodeRequestId);
        }

        return new BatchItem(ctx, future, new Response(),
                new ServerStatus(), decode, null, null, System.currentTimeMillis());
    }

    private static Response successResponse(long requestId) {
        Response response = new Response();
        response.setSuccess(true);
        return response;
    }

    private static Response failedResponse() {
        return Response.error(StrategyErrorType.NO_AVAILABLE_WORKER);
    }
}
