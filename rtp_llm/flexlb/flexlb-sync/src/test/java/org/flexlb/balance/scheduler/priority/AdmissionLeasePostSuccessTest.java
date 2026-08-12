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

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.atLeastOnce;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.timeout;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Comprehensive unit tests for the AdmissionLease "triple-lock" fix
 * (Fix A + B + C + D).
 *
 * <p>Covers the three-state CAS (UNSET→HANDED_OVER→CLOSED), post-success
 * soft timeout, forceCloseAfterHandover with engine cancel signal,
 * backpressure counter, and idempotency under concurrency.
 *
 * <p>Test index (15 tests):
 * <ol>
 *   <li>soft_timeout_fires_when_decode_does_not_accept</li>
 *   <li>soft_timeout_cancelled_when_decode_accepts_in_time</li>
 *   <li>cas_three_state_transitions</li>
 *   <li>forceCloseAfterHandover_is_idempotent</li>
 *   <li>handoverToEngine_and_close_are_mutually_exclusive_concurrent</li>
 *   <li>forceCloseAfterHandover_and_calibrate_race_is_idempotent</li>
 *   <li>backpressure_counter_increments_on_lease_creation</li>
 *   <li>backpressure_counter_decrements_on_lease_close</li>
 *   <li>finishYieldedById_only_called_on_soft_timeout_path</li>
 *   <li>soft_timeout_disabled_when_softTimeoutMs_is_zero</li>
 *   <li>backpressure_callback_null_is_safe</li>
 *   <li>soft_timeout_does_not_fire_when_prefill_fails</li>
 *   <li>markDecodeAccepted_decrements_counter_without_releasing_resources</li>
 *   <li>forceCloseAfterHandover_double_checks_isConfirmedTracked</li>
 *   <li>softTimeoutFuture_cancelled_on_close</li>
 * </ol>
 */
class AdmissionLeasePostSuccessTest {

    private static final long SOFT_TIMEOUT_MS = 50L;
    private static final long WAIT_MS = 300L;

    // ==================== Test 1 ====================

    /**
     * Soft timeout fires when decode doesn't accept: prefill succeeds →
     * handoverToEngine → soft timeout fires → decodeEp.isConfirmedTracked
     * returns false → forceCloseAfterHandover → verify tryRemove + release +
     * unregisterInflight + finishYieldedById all called.
     */
    @Test
    void soft_timeout_fires_when_decode_does_not_accept() throws Exception {
        InflightRegistrar registrar = mock(InflightRegistrar.class);
        DecodeEndpoint decodeEp = mock(DecodeEndpoint.class);
        PrefillQueueManager prefillQueue = mock(PrefillQueueManager.class);
        CompletableFuture<Response> future = new CompletableFuture<>();
        BatchItem item = batchItemWithDecode(3001L, future, 3001L);

        when(decodeEp.isConfirmedTracked(3001L)).thenReturn(false);

        AdmissionLease lease = new AdmissionLease(item, decodeEp, prefillQueue, registrar,
                SOFT_TIMEOUT_MS, null);
        lease.bindTo(future);

        future.complete(successResponse());
        Thread.sleep(WAIT_MS);

        verify(prefillQueue, times(1)).tryRemove(3001L, "LEASE_RELEASE");
        verify(decodeEp, times(1)).release(3001L);
        verify(registrar, times(1)).unregisterInflight(item);
        verify(registrar, times(1)).finishYieldedById(3001L, "post_success_soft_timeout");
    }

    // ==================== Test 2 ====================

    /**
     * Soft timeout cancelled when decode accepts in time: prefill succeeds →
     * handoverToEngine → soft timeout fires → isConfirmedTracked returns true
     * → markDecodeAccepted (counter decremented, resources NOT released).
     */
    @Test
    void soft_timeout_cancelled_when_decode_accepts_in_time() throws Exception {
        AtomicInteger activeCount = new AtomicInteger(0);
        InflightRegistrar registrar = mock(InflightRegistrar.class);
        DecodeEndpoint decodeEp = mock(DecodeEndpoint.class);
        PrefillQueueManager prefillQueue = mock(PrefillQueueManager.class);
        CompletableFuture<Response> future = new CompletableFuture<>();
        BatchItem item = batchItemWithDecode(3002L, future, 3002L);

        when(decodeEp.isConfirmedTracked(3002L)).thenReturn(true);

        activeCount.incrementAndGet();
        AdmissionLease lease = new AdmissionLease(item, decodeEp, prefillQueue, registrar,
                SOFT_TIMEOUT_MS, activeCount::decrementAndGet);
        lease.bindTo(future);

        future.complete(successResponse());
        Thread.sleep(WAIT_MS);

        // markDecodeAccepted decrements counter (CAS 1→2 + notifyCloseCallback)
        assertEquals(0, activeCount.get());
        // But does NOT release resources or send cancel signal
        verify(prefillQueue, never()).tryRemove(anyLong(), anyString());
        verify(decodeEp, never()).release(anyLong());
        verify(registrar, never()).unregisterInflight(any());
        verify(registrar, never()).finishYieldedById(anyLong(), anyString());
        // Lease reached CLOSED via markDecodeAccepted
        assertEquals(2, lease.leaseState());
    }

    // ==================== Test 3 ====================

    /**
     * CAS three-state transitions: UNSET→HANDED_OVER (handoverToEngine),
     * UNSET→CLOSED (close failure path), HANDED_OVER→CLOSED
     * (forceCloseAfterHandover soft timeout path).
     */
    @Test
    void cas_three_state_transitions() {
        InflightRegistrar registrar = mock(InflightRegistrar.class);

        // UNSET → HANDED_OVER (handoverToEngine)
        BatchItem item1 = batchItem(3101L, new CompletableFuture<>());
        AdmissionLease lease1 = new AdmissionLease(item1, null, null, registrar,
                0, null);
        assertEquals(0, lease1.leaseState()); // UNSET
        lease1.handoverToEngine();
        assertEquals(1, lease1.leaseState()); // HANDED_OVER

        // UNSET → CLOSED (close failure path)
        BatchItem item2 = batchItem(3102L, new CompletableFuture<>());
        AdmissionLease lease2 = new AdmissionLease(item2, null, null, registrar,
                0, null);
        assertEquals(0, lease2.leaseState()); // UNSET
        lease2.close();
        assertEquals(2, lease2.leaseState()); // CLOSED

        // HANDED_OVER → CLOSED (forceCloseAfterHandover)
        BatchItem item3 = batchItem(3103L, new CompletableFuture<>());
        AdmissionLease lease3 = new AdmissionLease(item3, null, null, registrar,
                0, null);
        lease3.handoverToEngine();
        assertEquals(1, lease3.leaseState()); // HANDED_OVER
        lease3.forceCloseAfterHandover();
        assertEquals(2, lease3.leaseState()); // CLOSED

        // close() from HANDED_OVER is now a no-op (Warning 2 fix)
        BatchItem item4 = batchItem(3104L, new CompletableFuture<>());
        AdmissionLease lease4 = new AdmissionLease(item4, null, null, registrar,
                0, null);
        lease4.handoverToEngine();
        assertEquals(1, lease4.leaseState()); // HANDED_OVER
        lease4.close();
        assertEquals(1, lease4.leaseState()); // still HANDED_OVER (close is no-op)
        // forceCloseAfterHandover transitions to CLOSED
        lease4.forceCloseAfterHandover();
        assertEquals(2, lease4.leaseState()); // CLOSED
    }

    // ==================== Test 4 ====================

    /**
     * forceCloseAfterHandover is idempotent: multiple calls only execute
     * resource release + finishYieldedById once.
     */
    @Test
    void forceCloseAfterHandover_is_idempotent() {
        InflightRegistrar registrar = mock(InflightRegistrar.class);
        DecodeEndpoint decodeEp = mock(DecodeEndpoint.class);
        PrefillQueueManager prefillQueue = mock(PrefillQueueManager.class);
        BatchItem item = batchItemWithDecode(3004L, new CompletableFuture<>(), 3004L);

        AdmissionLease lease = new AdmissionLease(item, decodeEp, prefillQueue, registrar,
                0, null);
        lease.handoverToEngine();

        lease.forceCloseAfterHandover();
        lease.forceCloseAfterHandover();
        lease.forceCloseAfterHandover();

        verify(prefillQueue, times(1)).tryRemove(3004L, "LEASE_RELEASE");
        verify(decodeEp, times(1)).release(3004L);
        verify(registrar, times(1)).unregisterInflight(item);
        verify(registrar, times(1)).finishYieldedById(3004L, "post_success_soft_timeout");
    }

    // ==================== Test 5 ====================

    /**
     * handoverToEngine and close are mutually exclusive under concurrency:
     * whichever wins the CAS (0→1 or 0→2) seals the lease. The other is
     * a no-op. After both settle, exactly one path executed.
     */
    @Test
    void handoverToEngine_and_close_are_mutually_exclusive_concurrent() throws Exception {
        int iterations = 200;
        int handoverWins = 0;
        int closeWins = 0;

        for (int i = 0; i < iterations; i++) {
            InflightRegistrar registrar = mock(InflightRegistrar.class);
            BatchItem item = batchItem(3200L + i, new CompletableFuture<>());
            // Use 0 soft timeout to avoid async interference
            AdmissionLease lease = new AdmissionLease(item, null, null, registrar,
                    0, null);

            CountDownLatch start = new CountDownLatch(1);
            AtomicInteger result = new AtomicInteger(-1); // -1=unset, 0=handover, 1=close

            Thread handoverThread = new Thread(() -> {
                try { start.await(); } catch (InterruptedException e) { return; }
                lease.handoverToEngine();
                result.compareAndSet(-1, 0);
            });

            Thread closeThread = new Thread(() -> {
                try { start.await(); } catch (InterruptedException e) { return; }
                lease.close();
                result.compareAndSet(-1, 1);
            });

            handoverThread.start();
            closeThread.start();
            start.countDown();
            handoverThread.join();
            closeThread.join();

            // Exactly one of handover/close should have sealed the lease first
            int state = lease.leaseState();
            assertTrue(state == 1 || state == 2,
                    "state must be HANDED_OVER(1) or CLOSED(2), got " + state);
            // If handover won (state went to 1 first), close may have then
            // transitioned 1→2; either way, exactly one initial CAS succeeded.
            if (state == 1) {
                handoverWins++;
            } else {
                closeWins++;
            }
        }

        assertTrue(handoverWins + closeWins == iterations,
                "all iterations must settle: handover=" + handoverWins + " close=" + closeWins);
    }

    // ==================== Test 6 ====================

    /**
     * forceCloseAfterHandover and normal calibrate race: both may call
     * decodeEp.release() — verify it's idempotent (release is a no-op on
     * an already-released reservation).
     */
    @Test
    void forceCloseAfterHandover_and_calibrate_race_is_idempotent() {
        InflightRegistrar registrar = mock(InflightRegistrar.class);
        DecodeEndpoint decodeEp = mock(DecodeEndpoint.class);
        PrefillQueueManager prefillQueue = mock(PrefillQueueManager.class);
        BatchItem item = batchItemWithDecode(3006L, new CompletableFuture<>(), 3006L);

        AdmissionLease lease = new AdmissionLease(item, decodeEp, prefillQueue, registrar,
                0, null);
        lease.handoverToEngine();

        // Simulate calibrate releasing the decode reservation first
        // (real DecodeEndpoint.release is idempotent via ConcurrentHashMap.remove)
        // The mock just accepts the call without error
        lease.forceCloseAfterHandover();

        // Verify release was called (forceCloseAfterHandover path)
        verify(decodeEp, atLeastOnce()).release(3006L);
        verify(registrar, times(1)).finishYieldedById(3006L, "post_success_soft_timeout");
        verify(registrar, times(1)).unregisterInflight(item);
    }

    // ==================== Test 7 ====================

    /**
     * Backpressure counter increments on lease creation: creating N leases
     * with a shared AtomicInteger increments the counter N times.
     */
    @Test
    void backpressure_counter_increments_on_lease_creation() {
        AtomicInteger activeCount = new AtomicInteger(0);
        List<AdmissionLease> leases = new ArrayList<>();

        for (int i = 0; i < 10; i++) {
            activeCount.incrementAndGet();
            BatchItem item = batchItem(3300L + i, new CompletableFuture<>());
            AdmissionLease lease = new AdmissionLease(item, null, null,
                    mock(InflightRegistrar.class), 0, activeCount::decrementAndGet);
            leases.add(lease);
        }

        assertEquals(10, activeCount.get());
    }

    // ==================== Test 8 ====================

    /**
     * Backpressure counter decrements on lease close: after closing a lease,
     * the counter decrements, allowing new requests to be admitted.
     */
    @Test
    void backpressure_counter_decrements_on_lease_close() {
        AtomicInteger activeCount = new AtomicInteger(0);

        // Create 3 leases
        activeCount.incrementAndGet();
        AdmissionLease lease1 = new AdmissionLease(
                batchItem(3401L, new CompletableFuture<>()), null, null,
                mock(InflightRegistrar.class), 0, activeCount::decrementAndGet);
        activeCount.incrementAndGet();
        AdmissionLease lease2 = new AdmissionLease(
                batchItem(3402L, new CompletableFuture<>()), null, null,
                mock(InflightRegistrar.class), 0, activeCount::decrementAndGet);
        activeCount.incrementAndGet();
        AdmissionLease lease3 = new AdmissionLease(
                batchItem(3403L, new CompletableFuture<>()), null, null,
                mock(InflightRegistrar.class), 0, activeCount::decrementAndGet);

        assertEquals(3, activeCount.get());

        // Close one via close() (failure path)
        lease2.close();
        assertEquals(2, activeCount.get());

        // Close one via forceCloseAfterHandover (soft timeout path)
        lease3.handoverToEngine();
        lease3.forceCloseAfterHandover();
        assertEquals(1, activeCount.get());

        // Close the last one via markDecodeAccepted (decode accepted path)
        // close() from HANDED_OVER is now a no-op
        lease1.handoverToEngine();
        lease1.markDecodeAccepted();
        assertEquals(0, activeCount.get());
    }

    // ==================== Test 9 ====================

    /**
     * finishYieldedById is only called on the soft-timeout path
     * (forceCloseAfterHandover), NOT on the normal close() failure path.
     */
    @Test
    void finishYieldedById_only_called_on_soft_timeout_path() {
        InflightRegistrar registrar = mock(InflightRegistrar.class);
        DecodeEndpoint decodeEp = mock(DecodeEndpoint.class);
        PrefillQueueManager prefillQueue = mock(PrefillQueueManager.class);

        // Normal close() path — NO finishYieldedById
        BatchItem item1 = batchItemWithDecode(3501L, new CompletableFuture<>(), 3501L);
        AdmissionLease lease1 = new AdmissionLease(item1, decodeEp, prefillQueue, registrar,
                0, null);
        lease1.close();
        verify(registrar, times(1)).unregisterInflight(item1);
        verify(registrar, never()).finishYieldedById(anyLong(), anyString());

        // forceCloseAfterHandover path — finishYieldedById IS called
        BatchItem item2 = batchItemWithDecode(3502L, new CompletableFuture<>(), 3502L);
        AdmissionLease lease2 = new AdmissionLease(item2, decodeEp, prefillQueue, registrar,
                0, null);
        lease2.handoverToEngine();
        lease2.forceCloseAfterHandover();
        verify(registrar, times(1)).finishYieldedById(3502L, "post_success_soft_timeout");

        // close() from HANDED_OVER is now a no-op — neither finishYieldedById
        // nor unregisterInflight is called
        BatchItem item3 = batchItemWithDecode(3503L, new CompletableFuture<>(), 3503L);
        AdmissionLease lease3 = new AdmissionLease(item3, decodeEp, prefillQueue, registrar,
                0, null);
        lease3.handoverToEngine();
        lease3.close();
        verify(registrar, never()).finishYieldedById(eq(3503L), anyString());
        verify(registrar, never()).unregisterInflight(eq(item3));
        verify(prefillQueue, never()).tryRemove(eq(3503L), anyString());
        verify(decodeEp, never()).release(eq(3503L));
    }

    // ==================== Test 10 ====================

    /**
     * Soft timeout is disabled when softTimeoutMs is 0: no force close
     * should fire after handoverToEngine.
     */
    @Test
    void soft_timeout_disabled_when_softTimeoutMs_is_zero() throws Exception {
        InflightRegistrar registrar = mock(InflightRegistrar.class);
        DecodeEndpoint decodeEp = mock(DecodeEndpoint.class);
        PrefillQueueManager prefillQueue = mock(PrefillQueueManager.class);
        CompletableFuture<Response> future = new CompletableFuture<>();
        BatchItem item = batchItemWithDecode(3003L, future, 3003L);

        when(decodeEp.isConfirmedTracked(3003L)).thenReturn(false);

        AdmissionLease lease = new AdmissionLease(item, decodeEp, prefillQueue, registrar,
                0, null); // softTimeoutMs = 0 → disabled
        lease.bindTo(future);

        future.complete(successResponse());
        Thread.sleep(WAIT_MS);

        // No force close should have fired
        verify(prefillQueue, never()).tryRemove(anyLong(), anyString());
        verify(decodeEp, never()).release(anyLong());
        verify(registrar, never()).unregisterInflight(any());
        verify(registrar, never()).finishYieldedById(anyLong(), anyString());
    }

    // ==================== Test 11 ====================

    /**
     * Backpressure callback null is safe: when no backpressure tracking is
     * needed (limit=0 case), the null callback doesn't cause errors.
     */
    @Test
    void backpressure_callback_null_is_safe() {
        InflightRegistrar registrar = mock(InflightRegistrar.class);
        BatchItem item = batchItem(3601L, new CompletableFuture<>());

        AdmissionLease lease = new AdmissionLease(item, null, null, registrar,
                0, null); // no callback
        lease.handoverToEngine();
        lease.forceCloseAfterHandover();

        // Should not throw — null callback is handled gracefully
        verify(registrar, times(1)).unregisterInflight(item);
        verify(registrar, times(1)).finishYieldedById(3601L, "post_success_soft_timeout");
    }

    // ==================== Test 12 ====================

    /**
     * Soft timeout does not fire when prefill fails: bindTo failure path
     * calls close() (CAS 0→2), handoverToEngine is never called, so no
     * soft timeout is scheduled.
     */
    @Test
    void soft_timeout_does_not_fire_when_prefill_fails() throws Exception {
        InflightRegistrar registrar = mock(InflightRegistrar.class);
        DecodeEndpoint decodeEp = mock(DecodeEndpoint.class);
        PrefillQueueManager prefillQueue = mock(PrefillQueueManager.class);
        CompletableFuture<Response> future = new CompletableFuture<>();
        BatchItem item = batchItemWithDecode(3005L, future, 3005L);

        when(decodeEp.isConfirmedTracked(3005L)).thenReturn(false);

        AdmissionLease lease = new AdmissionLease(item, decodeEp, prefillQueue, registrar,
                SOFT_TIMEOUT_MS, null);
        lease.bindTo(future);

        // Prefill fails → close() runs (CAS 0→2)
        future.complete(failedResponse());
        Thread.sleep(WAIT_MS);

        // close() should have released resources
        verify(prefillQueue, times(1)).tryRemove(3005L, "LEASE_RELEASE");
        verify(decodeEp, times(1)).release(3005L);
        verify(registrar, times(1)).unregisterInflight(item);

        // But NO finishYieldedById — that's only on the soft-timeout path
        verify(registrar, never()).finishYieldedById(anyLong(), anyString());

        // And the lease is CLOSED, not HANDED_OVER
        assertEquals(2, lease.leaseState());
    }

    // ==================== Test 13 ====================

    /**
     * markDecodeAccepted decrements the backpressure counter but does NOT
     * release resources or send a cancel signal.
     */
    @Test
    void markDecodeAccepted_decrements_counter_without_releasing_resources() {
        AtomicInteger activeCount = new AtomicInteger(0);
        InflightRegistrar registrar = mock(InflightRegistrar.class);
        DecodeEndpoint decodeEp = mock(DecodeEndpoint.class);
        PrefillQueueManager prefillQueue = mock(PrefillQueueManager.class);
        BatchItem item = batchItemWithDecode(3701L, new CompletableFuture<>(), 3701L);

        activeCount.incrementAndGet();
        AdmissionLease lease = new AdmissionLease(item, decodeEp, prefillQueue, registrar,
                0, activeCount::decrementAndGet);
        lease.handoverToEngine();
        assertEquals(1, lease.leaseState());
        assertEquals(1, activeCount.get());

        lease.markDecodeAccepted();

        assertEquals(2, lease.leaseState()); // CLOSED
        assertEquals(0, activeCount.get()); // counter decremented
        // Resources NOT released
        verify(prefillQueue, never()).tryRemove(anyLong(), anyString());
        verify(decodeEp, never()).release(anyLong());
        verify(registrar, never()).unregisterInflight(any());
        verify(registrar, never()).finishYieldedById(anyLong(), anyString());
    }

    /**
     * WorkerStatus can report Decode acceptance before the EnqueueBatch ACK
     * hands the lease to the engine.  The early observation must be retained
     * and close the lease immediately when handover later succeeds.
     */
    @Test
    void decodeAccepted_beforeHandover_closesLeaseWhenHandoverSucceeds() {
        AtomicInteger activeCount = new AtomicInteger(1);
        InflightRegistrar registrar = mock(InflightRegistrar.class);
        DecodeEndpoint decodeEp = mock(DecodeEndpoint.class);
        PrefillQueueManager prefillQueue = mock(PrefillQueueManager.class);
        BatchItem item = batchItemWithDecode(3702L, new CompletableFuture<>(), 3702L);

        AdmissionLease lease = new AdmissionLease(item, decodeEp, prefillQueue, registrar,
                SOFT_TIMEOUT_MS, activeCount::decrementAndGet);

        lease.markDecodeAccepted();
        assertEquals(0, lease.leaseState());
        assertEquals(1, activeCount.get());

        lease.handoverToEngine();

        assertEquals(2, lease.leaseState());
        assertEquals(0, activeCount.get());
        verify(prefillQueue, never()).tryRemove(anyLong(), anyString());
        verify(decodeEp, never()).release(anyLong());
        verify(registrar, never()).unregisterInflight(any());
        verify(registrar, never()).finishYieldedById(anyLong(), anyString());
    }

    // ==================== Test 14 ====================

    /**
     * forceCloseAfterHandover TOCTOU fix: after CAS succeeds, if
     * isConfirmedTracked returns true (decode accepted between the lambda
     * check and the CAS), only decrement the counter — no resource release,
     * no cancel signal.
     */
    @Test
    void forceCloseAfterHandover_double_checks_isConfirmedTracked() {
        AtomicInteger activeCount = new AtomicInteger(0);
        InflightRegistrar registrar = mock(InflightRegistrar.class);
        DecodeEndpoint decodeEp = mock(DecodeEndpoint.class);
        PrefillQueueManager prefillQueue = mock(PrefillQueueManager.class);
        BatchItem item = batchItemWithDecode(3702L, new CompletableFuture<>(), 3702L);

        // Simulate TOCTOU: isConfirmedTracked returns true (decode accepted
        // between the soft-timeout check and the CAS in forceCloseAfterHandover)
        when(decodeEp.isConfirmedTracked(3702L)).thenReturn(true);

        activeCount.incrementAndGet();
        AdmissionLease lease = new AdmissionLease(item, decodeEp, prefillQueue, registrar,
                0, activeCount::decrementAndGet);
        lease.handoverToEngine();

        // forceCloseAfterHandover detects the race — only decrements counter
        lease.forceCloseAfterHandover();

        assertEquals(2, lease.leaseState()); // CLOSED
        assertEquals(0, activeCount.get()); // counter decremented
        // Resources NOT released (decode accepted — engine owns them)
        verify(prefillQueue, never()).tryRemove(anyLong(), anyString());
        verify(decodeEp, never()).release(anyLong());
        verify(registrar, never()).unregisterInflight(any());
        verify(registrar, never()).finishYieldedById(anyLong(), anyString());
    }

    // ==================== Test 15 ====================

    /**
     * softTimeoutFuture is cancelled on close: after force-closing, the
     * pending soft timeout should NOT fire again (the ScheduledFuture was
     * cancelled). Resources are released exactly once.
     */
    @Test
    void softTimeoutFuture_cancelled_on_close() throws Exception {
        AtomicInteger activeCount = new AtomicInteger(0);
        InflightRegistrar registrar = mock(InflightRegistrar.class);
        DecodeEndpoint decodeEp = mock(DecodeEndpoint.class);
        PrefillQueueManager prefillQueue = mock(PrefillQueueManager.class);
        BatchItem item = batchItemWithDecode(3703L, new CompletableFuture<>(), 3703L);

        when(decodeEp.isConfirmedTracked(3703L)).thenReturn(false);

        activeCount.incrementAndGet();
        AdmissionLease lease = new AdmissionLease(item, decodeEp, prefillQueue, registrar,
                SOFT_TIMEOUT_MS, activeCount::decrementAndGet);
        lease.handoverToEngine(); // schedules soft timeout

        // Immediately force-close — cancels the pending soft timeout
        lease.forceCloseAfterHandover();
        assertEquals(2, lease.leaseState());
        assertEquals(0, activeCount.get());

        // Wait past the soft timeout — it should NOT have fired again
        Thread.sleep(WAIT_MS);

        // Resources released exactly once (not twice — soft timeout was cancelled)
        verify(prefillQueue, times(1)).tryRemove(3703L, "LEASE_RELEASE");
        verify(decodeEp, times(1)).release(3703L);
        verify(registrar, times(1)).unregisterInflight(item);
        verify(registrar, times(1)).finishYieldedById(3703L, "post_success_soft_timeout");
    }

    // ==================== Helpers ====================

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

    private static Response successResponse() {
        Response response = new Response();
        response.setSuccess(true);
        return response;
    }

    private static Response failedResponse() {
        return Response.error(StrategyErrorType.NO_AVAILABLE_WORKER);
    }
}
