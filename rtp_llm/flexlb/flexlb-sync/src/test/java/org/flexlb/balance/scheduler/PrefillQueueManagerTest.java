package org.flexlb.balance.scheduler;

import org.flexlb.balance.scheduler.priority.PrefillQueueSnapshot;
import org.flexlb.balance.scheduler.priority.QueuedRequestSnapshot;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.ScheduleBudget;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.PriorityBlockingQueue;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantLock;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;

/**
 * Phase 2 tests for {@link PrefillQueueManager} + {@link WorkerBatcher}:
 * Auto-TPM queue order (design doc 8.1), the 8.4 wait estimate,
 * legacy-order regression and the version-checked atomic victim
 * replace (17.2).
 *
 * <p>Uses the {@code fixed_window} algorithm so {@code computeSortKey}
 * (FIFO: enqueuedAtMs) needs no predictor and the batcher can be built
 * without a live {@code PrefillEndpoint}. The batcher is never started —
 * the queue is inspected/mutated directly through the manager facade.
 */
class PrefillQueueManagerTest {

    private FlexlbConfig config;

    @BeforeEach
    void setUp() {
        config = new FlexlbConfig();
        config.setFlexlbBatchAlgorithm("fixed_window");
        config.setAutoTpmEnabled(true);
        // The pre-18:41 estimate tests pin the pure jump/depth semantics
        // against the batching-window fallback; the cold-start floor has its
        // own dedicated tests below.
        config.setFlexlbDispatchIntervalColdFloorMs(0);
    }

    private WorkerBatcher newBatcher() {
        return new WorkerBatcher("test-worker", null, config,
                mock(BatchDecisionHandler.class), mock(BatchSchedulerReporter.class));
    }

    // ==================== 8.1 queue order ====================

    @Test
    void auto_tpm_order_is_priority_desc_then_arrival_fifo() {
        WorkerBatcher batcher = newBatcher();
        long now = System.currentTimeMillis();

        // Insertion order deliberately scrambled
        assertTrue(batcher.tryOffer(item(1, 50, now + 5_000, now, 128)));
        assertTrue(batcher.tryOffer(item(2, 70, now + 9_000, now + 100, 128)));
        assertTrue(batcher.tryOffer(item(3, 50, now + 1_000, now + 200, 128)));
        assertTrue(batcher.tryOffer(item(4, 50, now + 5_000, now - 100, 128)));

        PrefillQueueSnapshot snapshot = batcher.queueManager().snapshot();
        List<Long> order = snapshot.items().stream().map(QueuedRequestSnapshot::requestId).toList();

        // P70 first (priority desc); P50s strictly FIFO by arrival — item 3's
        // earlier deadline must NOT let it jump ahead of earlier arrivals
        assertEquals(List.of(2L, 4L, 1L, 3L), order);
        assertEquals(4, snapshot.items().size());
        assertEquals(config.getFlexlbBatchQueueMaxSize(), snapshot.queueCapacity());
        for (QueuedRequestSnapshot item : snapshot.items()) {
            assertEquals(QueuedRequestSnapshot.PREFILL_QUEUED, item.state());
        }
    }

    @Test
    void auto_tpm_order_breaks_arrival_ties_by_request_id() {
        WorkerBatcher batcher = newBatcher();
        long now = System.currentTimeMillis();

        // Same priority + same arrival: requestId asc decides (deadline is
        // no longer part of the ordering rule — PR-B removed it).
        assertTrue(batcher.tryOffer(item(1, 50, now + 9_000, now, 128)));
        assertTrue(batcher.tryOffer(item(2, 50, now + 1_000, now, 128)));
        assertTrue(batcher.tryOffer(item(4, 50, now + 9_000, now, 128)));
        assertTrue(batcher.tryOffer(item(3, 50, now + 9_000, now, 128)));

        List<Long> order = batcher.queueManager().snapshot().items().stream()
                .map(QueuedRequestSnapshot::requestId).toList();
        assertEquals(List.of(1L, 2L, 3L, 4L), order);
    }

    @Test
    void legacy_order_is_fifo_and_ignores_priority() {
        config.setAutoTpmEnabled(false);
        WorkerBatcher batcher = newBatcher();
        long now = System.currentTimeMillis();

        // High priority arrives last: legacy order must keep pure FIFO
        assertTrue(batcher.tryOffer(item(1, 30, now + 1_000, now, 128)));
        assertTrue(batcher.tryOffer(item(2, 50, now + 500, now + 100, 128)));
        assertTrue(batcher.tryOffer(item(3, 70, now + 100, now + 200, 128)));

        List<Long> order = batcher.queueManager().snapshot().items().stream()
                .map(QueuedRequestSnapshot::requestId).toList();
        assertEquals(List.of(1L, 2L, 3L), order);
    }

    // ==================== 8.4 wait estimate ====================

    @Test
    void estimate_wait_counts_only_items_ahead_and_is_monotonic_in_priority() {
        // Legacy jump-in semantics: disable the na130_4 depth term so the
        // estimate only reflects the items ordered ahead of the probe.
        config.setFlexlbQueueDepthPenaltyEnabled(false);
        config.setFlexlbBatchSizeMax(1);
        config.setFlexlbBatchFixedWaitMs(200);
        WorkerBatcher batcher = newBatcher();
        long now = System.currentTimeMillis();
        // Ancient arrivals zero out the head's remaining window for determinism
        assertTrue(batcher.tryOffer(item(1, 50, now, now - 100_000, 128)));
        assertTrue(batcher.tryOffer(item(2, 50, now, now - 100_000, 128)));

        PrefillQueueManager manager = batcher.queueManager();
        long waitP70 = manager.estimateWaitMs(70, now + 60_000, 999);
        long waitP50 = manager.estimateWaitMs(50, now + 60_000, 999);
        long waitP30 = manager.estimateWaitMs(30, now + 60_000, 999);

        // P70 jumps ahead of both P50 items: 0 cycles ahead
        assertEquals(0, waitP70);
        // P50/P30 wait behind both: 2 cycles x avgDispatchIntervalMs
        // (no dispatch observed yet -> fixed_window fallback = fixedWaitMs)
        assertEquals(400, waitP50);
        assertEquals(400, waitP30);
        assertTrue(waitP70 <= waitP50 && waitP50 <= waitP30);
    }

    // ==================== na130_4 depth penalty ====================

    @Test
    void depth_penalty_exposes_saturated_queue_to_priority_jumps() {
        // 16 low-priority items pin the queue. A P70 probe jumps ahead of all
        // of them (jumpWait = 0), but the queue needs 16/8 = 2 dispatch cycles
        // to drain, so with the depth penalty on (default) the estimate must
        // expose that drain horizon instead of a near-zero wait.
        config.setFlexlbBatchSizeMax(8);
        config.setFlexlbBatchFixedWaitMs(200);
        WorkerBatcher batcher = newBatcher();
        long now = System.currentTimeMillis();
        for (long requestId = 1; requestId <= 16; requestId++) {
            assertTrue(batcher.tryOffer(item(requestId, 30, now, now - 100_000, 128)));
        }

        PrefillQueueManager manager = batcher.queueManager();
        config.setFlexlbQueueDepthPenaltyEnabled(false);
        long legacyWait = manager.estimateWaitMs(70, now + 60_000, 999);
        config.setFlexlbQueueDepthPenaltyEnabled(true);
        long penalizedWait = manager.estimateWaitMs(70, now + 60_000, 999);

        // Legacy jump-only estimate is blind to the saturated queue (0 ahead)
        assertEquals(0, legacyWait);
        // Depth term: (16/8) cycles × 200ms fallback interval × factor 1.0
        assertEquals(400, penalizedWait);
        assertTrue(penalizedWait > legacyWait,
                "deep queue must look significantly slower than with the gate off");
    }

    @Test
    void depth_penalty_scales_with_factor() {
        config.setFlexlbBatchSizeMax(8);
        config.setFlexlbBatchFixedWaitMs(200);
        WorkerBatcher batcher = newBatcher();
        long now = System.currentTimeMillis();
        for (long requestId = 1; requestId <= 16; requestId++) {
            assertTrue(batcher.tryOffer(item(requestId, 30, now, now - 100_000, 128)));
        }

        PrefillQueueManager manager = batcher.queueManager();
        config.setFlexlbQueueDepthPenaltyFactor(2.5);
        // (16/8) cycles × 200ms × 2.5 = 1000
        assertEquals(1000, manager.estimateWaitMs(70, now + 60_000, 999));

        config.setFlexlbQueueDepthPenaltyFactor(0.5);
        // (16/8) cycles × 200ms × 0.5 = 200 — still >= jumpWait(0)
        assertEquals(200, manager.estimateWaitMs(70, now + 60_000, 999));
    }

    @Test
    void depth_penalty_returns_jump_wait_when_gate_off_or_queue_empty() {
        config.setFlexlbBatchSizeMax(1);
        config.setFlexlbBatchFixedWaitMs(200);
        WorkerBatcher batcher = newBatcher();
        long now = System.currentTimeMillis();
        assertTrue(batcher.tryOffer(item(1, 50, now, now - 100_000, 128)));
        assertTrue(batcher.tryOffer(item(2, 50, now, now - 100_000, 128)));

        PrefillQueueManager manager = batcher.queueManager();
        // Gate off: bit-for-bit legacy value (2 cycles × 200ms)
        config.setFlexlbQueueDepthPenaltyEnabled(false);
        assertEquals(400, manager.estimateWaitMs(50, now + 60_000, 999));
        // Gate on, jump wait dominates the depth term here (same 400)
        config.setFlexlbQueueDepthPenaltyEnabled(true);
        assertEquals(400, manager.estimateWaitMs(50, now + 60_000, 999));
        // Empty queue: depth term is 0, P70 probe keeps its jump-in zero wait
        WorkerBatcher emptyBatcher = newBatcher();
        assertEquals(0, emptyBatcher.queueManager().estimateWaitMs(70, now + 60_000, 999));
    }

    // ==================== 18:41 cold-start dispatch-interval floor ====================

    @Test
    void cold_start_estimate_is_floored_until_dispatch_samples_exist() {
        config.setFlexlbBatchSizeMax(8);
        config.setFlexlbBatchFixedWaitMs(120);
        config.setFlexlbDispatchIntervalColdFloorMs(500);
        WorkerBatcher batcher = newBatcher();
        long now = System.currentTimeMillis();
        for (long requestId = 1; requestId <= 16; requestId++) {
            assertTrue(batcher.tryOffer(item(requestId, 30, now, now - 100_000, 128)));
        }

        // Cold EMA right after a master restart: the 120ms batching-window
        // fallback would dilute the depth term to (16/8)×120=240; the floor
        // restores the counterweight to (16/8)×500=1000.
        assertEquals(1000, batcher.queueManager().estimateWaitMs(70, now + 60_000, 999),
                "a cold EMA must not dilute the depth term below the floor cadence");
    }

    @Test
    void converged_ema_is_used_verbatim_and_ignores_the_cold_floor() throws InterruptedException {
        config.setFlexlbBatchSizeMax(8);
        config.setFlexlbBatchFixedWaitMs(120);
        // Floor far above the injected EMA so the "not re-floored" assertion
        // tolerates CI scheduler jitter on the 200ms sleep.
        config.setFlexlbDispatchIntervalColdFloorMs(5000);
        WorkerBatcher batcher = newBatcher();
        long now = System.currentTimeMillis();

        // A batcher-context pair whose EMA is driven by real dispatch samples:
        // two dispatch(List.of(), null) calls ~200ms apart — the first seeds
        // lastDispatchAtMs, the second produces the first positive EMA
        // (~200ms, far below the 5000ms floor).
        PriorityBlockingQueue<BatchItem> queue =
                new PriorityBlockingQueue<>(11, WorkerBatcher.AUTO_TPM_QUEUE_ORDER);
        BatcherContext warmCtx = new BatcherContext("warm-worker", null, config,
                mock(BatchDecisionHandler.class), queue, new AtomicInteger(),
                new AtomicLong(), new ReentrantLock(), WorkerBatcher.AUTO_TPM_QUEUE_ORDER,
                mock(BatchSchedulerReporter.class));
        PrefillQueueManager warm = new PrefillQueueManager(batcher, warmCtx);
        for (long requestId = 1; requestId <= 8; requestId++) {
            queue.offer(item(requestId, 30, now, now - 100_000, 128));
        }
        warmCtx.dispatch(List.of(), null);
        Thread.sleep(200);
        warmCtx.dispatch(List.of(), null);

        // Core regression: a converged EMA (~200ms) must NOT be re-floored to
        // 5000 — the depth horizon (8/8)×EMA stays in the measured band.
        long warmWait = warm.estimateWaitMs(70, now + 60_000, 999);
        assertTrue(warmWait >= 200 && warmWait < 5000,
                "converged EMA must be used verbatim, got " + warmWait);

        // Control: the same queue/config on a cold context (no dispatch
        // samples) is floored to 5000 — proving the flag, not the queue,
        // drives the difference.
        PriorityBlockingQueue<BatchItem> coldQueue =
                new PriorityBlockingQueue<>(11, WorkerBatcher.AUTO_TPM_QUEUE_ORDER);
        BatcherContext coldCtx = new BatcherContext("cold-worker", null, config,
                mock(BatchDecisionHandler.class), coldQueue, new AtomicInteger(),
                new AtomicLong(), new ReentrantLock(), WorkerBatcher.AUTO_TPM_QUEUE_ORDER,
                mock(BatchSchedulerReporter.class));
        PrefillQueueManager cold = new PrefillQueueManager(batcher, coldCtx);
        for (long requestId = 1; requestId <= 8; requestId++) {
            coldQueue.offer(item(requestId, 30, now, now - 100_000, 128));
        }
        assertEquals(5000, cold.estimateWaitMs(70, now + 60_000, 999));
    }

    @Test
    void cold_floor_zero_disables_the_clamp() {
        config.setFlexlbBatchSizeMax(8);
        config.setFlexlbBatchFixedWaitMs(120);
        config.setFlexlbDispatchIntervalColdFloorMs(0);
        WorkerBatcher batcher = newBatcher();
        long now = System.currentTimeMillis();
        for (long requestId = 1; requestId <= 16; requestId++) {
            assertTrue(batcher.tryOffer(item(requestId, 30, now, now - 100_000, 128)));
        }

        // Bit-for-bit legacy behavior: (16/8) × 120ms fallback, no clamp.
        assertEquals(240, batcher.queueManager().estimateWaitMs(70, now + 60_000, 999));
    }

    // ==================== 17.2 atomic victim replace ====================

    @Test
    void replace_victims_is_version_checked_and_atomic() {
        config.setFlexlbBatchQueueMaxSize(2);
        WorkerBatcher batcher = newBatcher();
        long now = System.currentTimeMillis();
        assertTrue(batcher.tryOffer(item(1, 30, now + 1_000, now, 128)));
        assertTrue(batcher.tryOffer(item(2, 40, now + 1_000, now, 128)));
        long staleVersion = batcher.queueVersion() - 1;
        long freshVersion = batcher.queueVersion();

        // Stale version: nothing applied
        PrefillQueueManager.ReplaceOutcome stale = batcher.queueManager()
                .tryReplaceVictimsWithIncoming(List.of(1L), item(9, 70, now + 500, now, 128),
                        staleVersion);
        assertTrue(stale.isVersionMismatch());
        assertTrue(stale.removed().isEmpty());
        assertEquals(2, batcher.queueSize());

        // Fresh version: victim swapped for the incoming item atomically
        PrefillQueueManager.ReplaceOutcome ok = batcher.queueManager()
                .tryReplaceVictimsWithIncoming(List.of(1L), item(9, 70, now + 500, now, 128),
                        freshVersion);
        assertTrue(ok.isSuccess());
        assertEquals(1, ok.removed().size());
        assertEquals(1L, ok.removed().get(0).requestId());
        List<Long> order = batcher.queueManager().snapshot().items().stream()
                .map(QueuedRequestSnapshot::requestId).toList();
        assertEquals(List.of(9L, 2L), order);
    }

    @Test
    void replace_more_than_eight_victims_is_one_atomic_queue_transaction() {
        config.setFlexlbBatchQueueMaxSize(10);
        WorkerBatcher batcher = newBatcher();
        long now = System.currentTimeMillis();
        List<Long> victimIds = new ArrayList<>();
        for (long requestId = 1; requestId <= 10; requestId++) {
            assertTrue(batcher.tryOffer(
                    item(requestId, 30, now + 1_000, now + requestId, 128)));
            victimIds.add(requestId);
        }
        long version = batcher.queueVersion();

        PrefillQueueManager.ReplaceOutcome outcome = batcher.queueManager()
                .tryReplaceVictimsWithIncoming(
                        victimIds, item(100, 70, now + 500, now + 100, 128), version);

        assertTrue(outcome.isSuccess());
        assertEquals(victimIds, outcome.removed().stream()
                .map(BatchItem::requestId)
                .toList());
        assertEquals(List.of(100L), batcher.queueManager().snapshot().items().stream()
                .map(QueuedRequestSnapshot::requestId)
                .toList());
    }

    // ==================== helpers ====================

    private static BatchItem item(long requestId, int priority, long deadlineMs,
                                  long enqueuedAtMs, long seqLen) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(seqLen);
        request.setPriority(priority);
        BalanceContext ctx = new BalanceContext();
        ctx.setRequest(request);
        ctx.setBudget(ScheduleBudget.forDeadline(priority, enqueuedAtMs, deadlineMs));
        BatchItem item = new BatchItem(ctx, new CompletableFuture<>(), null,
                null, null, null, null, enqueuedAtMs);
        return item;
    }
}
