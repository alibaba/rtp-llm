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
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;

/**
 * PR-D §2.2: timeout-release idempotency专项单测 — verifies the CAS
 * boundary between {@code orTimeout} (entry timeout on submit) and the
 * dispatch-pipeline terminal paths.
 *
 * <p>Design §2.2: "已 dispatch 的请求撞上超时属于竞争窗口：
 * close() 三步幂等无害". The {@link AdmissionLease#settled} CAS guarantees
 * that exactly one of {@code markDeliverySucceeded()} / {@code close()} runs,
 * even when the timeout fires concurrently with the dispatch pipeline.
 *
 * <p>Scenarios:
 * <ol>
 *   <li>orTimeout fires → close() runs → all three resources released;</li>
 *   <li>orTimeout fires + dispatch failure (double terminal) → close()
 *       runs exactly once (CAS);</li>
 *   <li>Dispatch success + orTimeout fires later → markDeliverySucceeded
 *       already sealed → close() is a no-op (no resource release);</li>
 *   <li>Dispatch failure + orTimeout fires later → close() already sealed
 *       → second close() is a no-op (resources released exactly once).</li>
 * </ol>
 */
class TimeoutReleaseIdempotencyTest {

    @Test
    void orTimeout_fires_releases_all_resources_exactly_once() {
        InflightRegistrar registrar = mock(InflightRegistrar.class);
        DecodeEndpoint decodeEp = mock(DecodeEndpoint.class);
        PrefillQueueManager prefillQueue = mock(PrefillQueueManager.class);
        CompletableFuture<Response> future = new CompletableFuture<>();
        BatchItem item = batchItemWithDecode(2001L, future, 2001L);

        AdmissionLease lease = new AdmissionLease(item, decodeEp, prefillQueue, registrar);
        lease.bindTo(future);

        // Simulate orTimeout firing
        future.completeExceptionally(new TimeoutException("admission timeout"));

        awaitCallback(future);

        verify(prefillQueue, times(1)).tryRemove(2001L, "LEASE_RELEASE");
        verify(decodeEp, times(1)).release(2001L);
        verify(registrar, times(1)).unregisterInflight(item);
    }

    @Test
    void orTimeout_and_dispatch_failure_both_fire_close_runs_once() {
        InflightRegistrar registrar = mock(InflightRegistrar.class);
        DecodeEndpoint decodeEp = mock(DecodeEndpoint.class);
        PrefillQueueManager prefillQueue = mock(PrefillQueueManager.class);
        CompletableFuture<Response> future = new CompletableFuture<>();
        BatchItem item = batchItemWithDecode(2002L, future, 2002L);

        AdmissionLease lease = new AdmissionLease(item, decodeEp, prefillQueue, registrar);
        lease.bindTo(future);

        // Race: orTimeout fires first
        future.completeExceptionally(new TimeoutException("admission timeout"));
        // Dispatch pipeline also tries to fail the future (no-op — already completed)
        future.complete(Response.error(StrategyErrorType.SCHEDULER_PLAN_CONFLICT));

        awaitCallback(future);

        // CAS: only one close() ran
        verify(prefillQueue, times(1)).tryRemove(2002L, "LEASE_RELEASE");
        verify(decodeEp, times(1)).release(2002L);
        verify(registrar, times(1)).unregisterInflight(item);
    }

    /**
     * close() from DELIVERY_PENDING is now a no-op (Warning 2 fix). After dispatch
     * success triggers markDeliverySucceeded (0→1), a subsequent close() does NOT
     * transition 1→2. Post-delivery cleanup routes through
     * reconcileAfterDeliveryTimeout() or markDecodeAccepted() only.
     */
    @Test
    void dispatch_success_then_close_is_noop_from_handed_over() {
        InflightRegistrar registrar = mock(InflightRegistrar.class);
        DecodeEndpoint decodeEp = mock(DecodeEndpoint.class);
        PrefillQueueManager prefillQueue = mock(PrefillQueueManager.class);
        CompletableFuture<Response> future = new CompletableFuture<>();
        BatchItem item = batchItemWithDecode(2003L, future, 2003L);

        AdmissionLease lease = new AdmissionLease(item, decodeEp, prefillQueue, registrar);
        lease.bindTo(future);

        // Dispatch succeeds first → markDeliverySucceeded (CAS 0→1)
        Response success = new Response();
        success.setSuccess(true);
        future.complete(success);
        awaitCallback(future);

        // close() from DELIVERY_PENDING is now a no-op (CAS fails)
        lease.close();

        // No resources released — close is a no-op from DELIVERY_PENDING
        verify(prefillQueue, never()).tryRemove(anyLong(), anyString());
        verify(decodeEp, never()).release(anyLong());
        verify(registrar, never()).unregisterInflight(any());
    }

    @Test
    void dispatch_failure_then_orTimeout_close_seals_second_close_is_noop() {
        InflightRegistrar registrar = mock(InflightRegistrar.class);
        DecodeEndpoint decodeEp = mock(DecodeEndpoint.class);
        PrefillQueueManager prefillQueue = mock(PrefillQueueManager.class);
        CompletableFuture<Response> future = new CompletableFuture<>();
        BatchItem item = batchItemWithDecode(2004L, future, 2004L);

        AdmissionLease lease = new AdmissionLease(item, decodeEp, prefillQueue, registrar);
        lease.bindTo(future);

        // Dispatch fails first → close() runs
        future.complete(Response.error(StrategyErrorType.NO_AVAILABLE_WORKER));
        awaitCallback(future);

        // orTimeout fires later — simulate by calling close() directly
        lease.close();

        // CAS: close ran exactly once (from bindTo's err branch)
        verify(prefillQueue, times(1)).tryRemove(2004L, "LEASE_RELEASE");
        verify(decodeEp, times(1)).release(2004L);
        verify(registrar, times(1)).unregisterInflight(item);
    }

    @Test
    void real_orTimeout_completes_future_exceptionally_and_triggers_close() throws Exception {
        InflightRegistrar registrar = mock(InflightRegistrar.class);
        DecodeEndpoint decodeEp = mock(DecodeEndpoint.class);
        PrefillQueueManager prefillQueue = mock(PrefillQueueManager.class);
        CompletableFuture<Response> future = new CompletableFuture<>();
        BatchItem item = batchItemWithDecode(2005L, future, 2005L);

        AdmissionLease lease = new AdmissionLease(item, decodeEp, prefillQueue, registrar);
        lease.bindTo(future);

        // Attach a real orTimeout (1ms) and wait for it to fire
        future.orTimeout(1, TimeUnit.MILLISECONDS);
        // Wait for the callback chain to settle
        Thread.sleep(100);

        assertTrue(future.isDone(), "future must be completed by orTimeout");
        assertTrue(future.isCompletedExceptionally(),
                "orTimeout must complete exceptionally");

        verify(prefillQueue, times(1)).tryRemove(2005L, "LEASE_RELEASE");
        verify(decodeEp, times(1)).release(2005L);
        verify(registrar, times(1)).unregisterInflight(item);
    }

    // ==================== helpers ====================

    /**
     * Wait for the whenComplete callback registered by bindTo to fire.
     * CompletableFuture.whenComplete callbacks run synchronously on the
     * completing thread, so by the time complete()/completeExceptionally()
     * returns, the callback has already executed. For orTimeout (which
     * completes on a separate thread), we need a brief wait.
     */
    private static void awaitCallback(CompletableFuture<?> future) {
        if (!future.isDone()) {
            try { future.get(1, TimeUnit.SECONDS); } catch (Exception ignored) { }
        }
        // Small sleep for async callback propagation
        try { Thread.sleep(10); } catch (InterruptedException ignored) { }
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
}
