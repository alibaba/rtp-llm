package org.flexlb.balance.dp;

import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.TimeUnit;
import java.util.stream.IntStream;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

class InflightBatchRegistryTest {

    private InflightBatchRegistry reg;
    private ServerStatus prefill;
    private ServerStatus decode;

    @BeforeEach
    void setUp() {
        reg = new InflightBatchRegistry();
        prefill = ss("10.0.0.1", 8080, 9080);
        decode = ss("10.0.0.2", 8081, 9081);
    }

    @Test
    void register_then_lookup_byRequest_and_byBatch() {
        PrefillBatch batch = makeBatch(4, prefill, decode);
        long batchId = 100L;
        reg.register(batchId, batch);

        assertEquals(1, reg.sizeBatches());
        assertEquals(4, reg.sizeRequests());

        // every requestId resolves back to its batch + decode address
        for (PendingRequest r : batch.requests()) {
            InflightBatchRegistry.RequestEntry entry = reg.lookupByRequest(r.requestId());
            assertNotNull(entry);
            assertEquals(batchId, entry.batchId());
            assertEquals(prefill, entry.prefill());
            assertEquals(decode, entry.decode());
        }
        // batch-level lookup also works
        InflightBatchRegistry.BatchEntry batchEntry = reg.lookupByBatch(batchId);
        assertNotNull(batchEntry);
        assertEquals(prefill, batchEntry.prefill());
        assertEquals(4, batchEntry.requestIds().size());
    }

    @Test
    void removeRequest_keeps_batch_entry_until_last_request_gone() {
        PrefillBatch batch = makeBatch(4, prefill, decode);
        reg.register(1L, batch);

        // Remove the first 3 → batch should still exist (one request still alive)
        for (int i = 0; i < 3; i++) {
            reg.removeRequest(batch.requests().get(i).requestId());
        }
        assertNotNull(reg.lookupByBatch(1L), "batch must remain while at least one request is still pending");
        assertEquals(1, reg.sizeRequests());

        // Remove the last one → batch should be evicted as well
        reg.removeRequest(batch.requests().get(3).requestId());
        assertNull(reg.lookupByBatch(1L));
        assertEquals(0, reg.sizeBatches());
        assertEquals(0, reg.sizeRequests());
    }

    @Test
    void remove_batch_clears_all_request_entries() {
        PrefillBatch batch = makeBatch(3, prefill, decode);
        reg.register(7L, batch);
        assertEquals(3, reg.sizeRequests());

        reg.remove(7L);

        assertEquals(0, reg.sizeBatches());
        assertEquals(0, reg.sizeRequests());
        for (PendingRequest r : batch.requests()) {
            assertNull(reg.lookupByRequest(r.requestId()));
        }
    }

    @Test
    void lookup_unknown_returns_null() {
        assertNull(reg.lookupByRequest(99L));
        assertNull(reg.lookupByBatch(99L));
    }

    @Test
    void evict_stale_drops_old_entries_and_increments_counter() throws Exception {
        // Inject a "long-ago" batch
        PrefillBatch batch = makeBatch(2, prefill, decode);
        reg.register(42L, batch);

        // Reflectively backdate createdAtMs to simulate staleness
        long longAgoMs = System.currentTimeMillis() - InflightBatchRegistry.STALE_THRESHOLD_MS - 1000;
        forceCreatedAt(reg, 42L, longAgoMs);

        long evictedBefore = reg.getEvictedCount();
        reg.evictStale();

        assertEquals(0, reg.sizeBatches(), "batch older than STALE_THRESHOLD_MS must be evicted");
        assertEquals(0, reg.sizeRequests());
        assertEquals(evictedBefore + 1, reg.getEvictedCount(),
                "metric counter must increment by the number of evicted batches");
    }

    @Test
    void evict_stale_keeps_fresh_entries() {
        PrefillBatch batch = makeBatch(2, prefill, decode);
        reg.register(99L, batch);
        reg.evictStale();
        assertEquals(1, reg.sizeBatches(), "fresh entries must not be evicted");
        assertEquals(2, reg.sizeRequests());
        assertEquals(0, reg.getEvictedCount(), "metric counter must stay 0 when nothing was evicted");
    }

    @Test
    void stale_threshold_is_well_above_typical_request_lifetime() {
        // Guard against accidentally bumping the threshold back down to a value
        // small enough to be mistaken for an "expected completion time". 10 minutes
        // is the V1-α design floor — long enough that explicit cleanup (Cancel /
        // completion notification) handles every healthy request.
        assertTrue(InflightBatchRegistry.STALE_THRESHOLD_MS >= 10L * 60_000L,
                "STALE_THRESHOLD_MS must remain a memory-leak safety net, not a normal cleanup mechanism");
    }

    // ============== Per-request state machine ==============

    @Test
    void newly_registered_request_is_PENDING_ACK() {
        PrefillBatch batch = makeBatch(4, prefill, decode);
        reg.register(1L, batch);
        for (PendingRequest r : batch.requests()) {
            assertEquals(InflightBatchRegistry.RequestState.PENDING_ACK, reg.getState(r.requestId()));
        }
    }

    @Test
    void markActive_succeeds_from_PENDING_ACK() {
        PrefillBatch batch = makeBatch(2, prefill, decode);
        reg.register(1L, batch);
        long reqId = batch.requests().get(0).requestId();

        assertTrue(reg.markActive(reqId));
        assertEquals(InflightBatchRegistry.RequestState.ACTIVE, reg.getState(reqId));
    }

    @Test
    void markActive_fails_after_cancel_tombstone() {
        // The whole point of the tombstone: if Cancel beats the Enqueue ack,
        // markActive (called from handleAck) must NOT undo the cancellation.
        PrefillBatch batch = makeBatch(2, prefill, decode);
        reg.register(1L, batch);
        long reqId = batch.requests().get(0).requestId();

        InflightBatchRegistry.RequestState prev = reg.markCancelled(reqId);
        assertEquals(InflightBatchRegistry.RequestState.PENDING_ACK, prev);
        assertEquals(InflightBatchRegistry.RequestState.CANCELLED, reg.getState(reqId));

        assertFalse(reg.markActive(reqId),
                "markActive must NOT silently overwrite a CANCELLED tombstone — "
                + "this is what guarantees handleAck sees the cancel-before-ack race");
        assertEquals(InflightBatchRegistry.RequestState.CANCELLED, reg.getState(reqId));
    }

    @Test
    void markCancelled_returns_previous_state() {
        PrefillBatch batch = makeBatch(2, prefill, decode);
        reg.register(1L, batch);
        long reqId = batch.requests().get(0).requestId();

        // First cancel: was PENDING_ACK
        assertEquals(InflightBatchRegistry.RequestState.PENDING_ACK, reg.markCancelled(reqId));
        // Re-entrant cancel: was CANCELLED
        assertEquals(InflightBatchRegistry.RequestState.CANCELLED, reg.markCancelled(reqId));
    }

    @Test
    void markCancelled_after_active_returns_active() {
        PrefillBatch batch = makeBatch(2, prefill, decode);
        reg.register(1L, batch);
        long reqId = batch.requests().get(0).requestId();

        assertTrue(reg.markActive(reqId));
        assertEquals(InflightBatchRegistry.RequestState.ACTIVE, reg.markCancelled(reqId),
                "previous state was ACTIVE — caller knows to do the standard cascade Cancel");
    }

    @Test
    void getState_unknown_request_returns_null() {
        assertNull(reg.getState(999_999L));
    }

    @Test
    void state_is_cleared_when_request_removed() {
        PrefillBatch batch = makeBatch(2, prefill, decode);
        reg.register(1L, batch);
        long reqId = batch.requests().get(0).requestId();

        reg.removeRequest(reqId);
        assertNull(reg.getState(reqId), "state must be cleared so retired ids don't leak memory");
    }

    @Test
    void state_is_cleared_when_batch_removed() {
        PrefillBatch batch = makeBatch(3, prefill, decode);
        reg.register(7L, batch);
        reg.remove(7L);
        for (PendingRequest r : batch.requests()) {
            assertNull(reg.getState(r.requestId()));
        }
    }

    @Test
    void concurrent_register_and_remove_no_leaks() throws Exception {
        int batches = 100;
        int parallel = 4;
        java.util.concurrent.ExecutorService pool = java.util.concurrent.Executors.newFixedThreadPool(parallel);
        java.util.concurrent.CountDownLatch start = new java.util.concurrent.CountDownLatch(1);

        for (int b = 0; b < batches; b++) {
            final long batchId = b;
            pool.submit(() -> {
                try { start.await(); } catch (InterruptedException e) { Thread.currentThread().interrupt(); }
                PrefillBatch batch = makeBatch(4, prefill, decode);
                reg.register(batchId, batch);
                for (PendingRequest r : batch.requests()) {
                    reg.removeRequest(r.requestId());
                }
            });
        }
        start.countDown();
        pool.shutdown();
        assertTrue(pool.awaitTermination(5, TimeUnit.SECONDS));

        assertEquals(0, reg.sizeBatches(), "all batches should be fully removed");
        assertEquals(0, reg.sizeRequests());
    }

    // ============== helpers ==============

    private static ServerStatus ss(String ip, int httpPort, int grpcPort) {
        ServerStatus s = new ServerStatus();
        s.setServerIp(ip);
        s.setHttpPort(httpPort);
        s.setGrpcPort(grpcPort);
        return s;
    }

    private static final java.util.concurrent.atomic.AtomicLong REQ_ID_GEN = new java.util.concurrent.atomic.AtomicLong(0);

    private static PrefillBatch makeBatch(int size, ServerStatus prefill, ServerStatus decode) {
        List<PendingRequest> reqs = IntStream.range(0, size)
                .mapToObj(i -> {
                    org.flexlb.dao.BalanceContext ctx = new org.flexlb.dao.BalanceContext();
                    org.flexlb.dao.loadbalance.Request req = new org.flexlb.dao.loadbalance.Request();
                    req.setRequestId(REQ_ID_GEN.incrementAndGet());
                    ctx.setRequest(req);
                    return PendingRequest.of(ctx, prefill, decode, new CompletableFuture<Response>());
                })
                .toList();
        return new PrefillBatch(prefill, new ArrayList<>(reqs), 4);
    }

    /** Reflectively force createdAtMs on the registered batch + its requests for eviction tests. */
    @SuppressWarnings("unchecked")
    private static void forceCreatedAt(InflightBatchRegistry reg, long batchId, long createdAtMs)
            throws ReflectiveOperationException {
        Field byBatchField = InflightBatchRegistry.class.getDeclaredField("byBatch");
        byBatchField.setAccessible(true);
        var byBatch = (java.util.Map<Long, InflightBatchRegistry.BatchEntry>) byBatchField.get(reg);
        InflightBatchRegistry.BatchEntry old = byBatch.get(batchId);
        InflightBatchRegistry.BatchEntry replaced = new InflightBatchRegistry.BatchEntry(
                old.batchId(), old.prefill(), old.requestIds(), createdAtMs);
        byBatch.put(batchId, replaced);

        Field byRequestField = InflightBatchRegistry.class.getDeclaredField("byRequest");
        byRequestField.setAccessible(true);
        var byRequest = (java.util.Map<Long, InflightBatchRegistry.RequestEntry>) byRequestField.get(reg);
        for (Long requestId : old.requestIds()) {
            InflightBatchRegistry.RequestEntry oldR = byRequest.get(requestId);
            if (oldR != null) {
                byRequest.put(requestId, new InflightBatchRegistry.RequestEntry(
                        oldR.requestId(), oldR.batchId(), oldR.prefill(), oldR.decode(), createdAtMs));
            }
        }
    }
}
