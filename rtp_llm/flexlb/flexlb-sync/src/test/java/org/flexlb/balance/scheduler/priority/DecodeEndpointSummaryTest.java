package org.flexlb.balance.scheduler.priority;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.endpoint.RequestInflight;
import org.flexlb.dao.master.TaskInfo;
import org.flexlb.dao.master.WorkerStatus;
import org.flexlb.dao.master.WorkerStatusResponse;
import org.flexlb.enums.TaskPhase;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * O(1) snapshot redesign tests for {@link DecodeEndpointSummary}: the
 * aggregate-only capture must always agree with a full recomputation from the
 * per-entry state, including after concurrent reserve/release churn — the
 * aggregates are maintained inside the same mutation scope, so no drift
 * window may exist. The lazy {@link DecodeEndpointSummary#toFullSnapshot()}
 * upgrade must be field-for-field equivalent to a direct
 * {@link DecodeEndpointSnapshot#capture}.
 */
class DecodeEndpointSummaryTest {

    private WorkerStatus status;
    private DecodeEndpoint endpoint;

    @BeforeEach
    void setUp() {
        status = new WorkerStatus();
        status.setIp("10.0.0.1");
        status.setPort(8080);
        status.setGrpcPort(8081);
        endpoint = new DecodeEndpoint(status);
    }

    // ==================== summary == full snapshot aggregates ====================

    @Test
    void capture_matchesFullSnapshotAggregates_withReservedAndConfirmed() {
        updateStatus(null, 10_000);
        endpoint.reserve(1L, 500, 600, 30, 1_000);
        endpoint.reserve(2L, 300, 350, 40, 2_000);
        endpoint.markQueuedPhase(2L);

        // Confirm a third request via the engine status report path.
        endpoint.reserve(3L, 200, 220, 50, 3_000);
        TaskInfo running = new TaskInfo();
        running.setRequestId(3L);
        running.setPhase(TaskPhase.KV_ALLOCATED);
        updateStatus(Map.of("3", running), 9_000);

        DecodeEndpointSummary summary = DecodeEndpointSummary.capture(endpoint, 5);
        DecodeEndpointSnapshot full = DecodeEndpointSnapshot.capture(endpoint, 5);

        assertEquals(full.endpointId(), summary.endpointId());
        assertEquals(full.admissionVersion(), summary.admissionVersion());
        assertEquals(full.realKvAvailable(), summary.realKvAvailable());
        assertEquals(full.realKvTotal(), summary.realKvTotal());
        assertEquals(full.totalLoad(), summary.totalLoad());
        assertEquals(full.engineLoad(), summary.engineLoad());
        assertEquals(full.concurrencyLimit(), summary.concurrencyLimit());
        assertEquals(full.hardKvReserved(), summary.hardKvReserved());
        assertEquals(full.expectedKvReserved(), summary.expectedKvReserved());

        // Lazy upgrade sees the identical per-entry state.
        DecodeEndpointSnapshot upgraded = summary.toFullSnapshot();
        assertEquals(full.admissionVersion(), upgraded.admissionVersion());
        assertEquals(full.reserved().size(), upgraded.reserved().size());
        assertEquals(full.running().size(), upgraded.running().size());
        assertEquals(full.hardKvReserved(), upgraded.hardKvReserved());
    }

    // ==================== concurrent churn: aggregates never drift ====================

    @Test
    void concurrentReserveRelease_aggregatesMatchFullRecomputation() throws Exception {
        updateStatus(null, 1_000_000);
        int threads = 8;
        int idsPerThread = 200;
        CountDownLatch start = new CountDownLatch(1);
        CountDownLatch done = new CountDownLatch(threads);
        List<Thread> workers = new ArrayList<>();
        for (int t = 0; t < threads; t++) {
            long base = (t + 1) * 10_000L;
            workers.add(new Thread(() -> {
                try {
                    start.await();
                    for (long id = base; id < base + idsPerThread; id++) {
                        endpoint.reserve(id, 100 + id % 7, 110 + id % 7, 30, 1_000);
                        if (id % 2 == 0) {
                            endpoint.release(id);
                        }
                    }
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                } finally {
                    done.countDown();
                }
            }));
        }
        workers.forEach(Thread::start);
        start.countDown();
        assertTrue(done.await(30, TimeUnit.SECONDS), "workers did not finish");

        DecodeEndpointSummary summary = DecodeEndpointSummary.capture(endpoint, 0);

        // Full recomputation from the per-entry reserved view.
        Map<Long, RequestInflight> reserved = endpoint.reservedView();
        long recomputedHardKv = 0;
        long recomputedExpectedKv = 0;
        for (RequestInflight entry : reserved.values()) {
            recomputedHardKv += entry.releasableKvTokens();
            recomputedExpectedKv += entry.expectedKvTokens();
        }

        assertEquals(threads * idsPerThread / 2, reserved.size());
        assertEquals(reserved.size(), summary.totalLoad());
        assertEquals(recomputedHardKv, summary.hardKvReserved());
        assertEquals(recomputedExpectedKv, summary.expectedKvReserved());
        assertEquals(endpoint.admissionVersion(), summary.admissionVersion());
        assertEquals(endpoint.realKvAvailable(), summary.realKvAvailable());

        // And the full snapshot agrees with the summary on every aggregate.
        DecodeEndpointSnapshot full = DecodeEndpointSnapshot.capture(endpoint, 0);
        assertEquals(full.totalLoad(), summary.totalLoad());
        assertEquals(full.engineLoad(), summary.engineLoad());
        assertEquals(full.hardKvReserved(), summary.hardKvReserved());
        assertEquals(full.expectedKvReserved(), summary.expectedKvReserved());
        assertEquals(full.reserved().size(), summary.totalLoad());
    }

    // ==================== deficit helpers mirror the planner ====================

    @Test
    void deficits_mirrorEvictionPlannerSemantics() {
        updateStatus(null, 1_000);
        endpoint.reserve(1L, 800, 900, 30, 1_000);
        endpoint.reserve(2L, 100, 150, 40, 2_000);

        DecodeEndpointSummary summary = DecodeEndpointSummary.capture(endpoint, 2);
        DecodeEndpointSnapshot full = DecodeEndpointSnapshot.capture(endpoint, 2);

        // Slot: engineLoad 2 + 1 > limit 2 → deficit 1 (same as full-based math).
        assertEquals(Math.max(0, full.engineLoad() + 1 - full.concurrencyLimit()),
                summary.slotDeficit());
        // KV: available = 1000 - 900 hard reserved = 100 < incoming 500.
        assertEquals(400, summary.kvDeficit(500));
        assertEquals(0, summary.kvDeficit(50));

        PriorityRequestEnvelope envelope = new PriorityRequestEnvelope(
                9L, 70, 500, 100, System.currentTimeMillis(), 1_000,
                System.currentTimeMillis() + 1_000, 500, 600);
        assertTrue(EvictionPlanner.hasDeficit(envelope, summary));

        DecodeEndpointSummary unlimited = DecodeEndpointSummary.capture(endpoint, 0);
        assertEquals(0, unlimited.slotDeficit());
        PriorityRequestEnvelope tiny = new PriorityRequestEnvelope(
                9L, 70, 10, 100, System.currentTimeMillis(), 1_000,
                System.currentTimeMillis() + 1_000, 10, 110);
        assertFalse(EvictionPlanner.hasDeficit(tiny, unlimited));
    }

    // ==================== helpers ====================

    private void updateStatus(Map<String, TaskInfo> running, long availableKvCacheTokens) {
        status.getAvailableKvCacheTokens().set(availableKvCacheTokens);
        status.getTotalKvCacheTokens().set(1_000_000);
        WorkerStatusResponse response = new WorkerStatusResponse();
        response.setRunningTaskInfo(running);
        endpoint.onWorkerStatusUpdate(status, response);
    }
}
