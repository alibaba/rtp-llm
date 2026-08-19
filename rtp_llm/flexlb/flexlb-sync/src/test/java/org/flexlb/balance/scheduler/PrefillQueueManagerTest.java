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

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;

/**
 * Phase 2 tests for {@link PrefillQueueManager} + {@link WorkerBatcher}:
 * Auto-TPM queue order (design doc 8.1), the measured queue-age wait
 * estimate, legacy-order regression and the version-checked atomic victim
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

    // ==================== measured queue-age estimate ====================

    @Test
    void empty_queue_estimates_zero_wait() {
        WorkerBatcher batcher = newBatcher();
        assertEquals(0, batcher.queueManager().estimateWaitMs(70, Long.MAX_VALUE, 999));
    }

    @Test
    void queue_age_grows_with_wall_clock_since_head_enqueue() throws InterruptedException {
        WorkerBatcher batcher = newBatcher();
        long now = System.currentTimeMillis();
        assertTrue(batcher.tryOffer(item(1, 50, now + 60_000, now, 128)));
        PrefillQueueManager manager = batcher.queueManager();

        long before = manager.estimateWaitMs(70, now + 60_000, 999);
        Thread.sleep(120);
        long after = manager.estimateWaitMs(70, now + 60_000, 999);

        assertTrue(before < after, "age must grow with the wall clock");
        assertTrue(after >= 100, "120ms sleep must show up in the measured age, got " + after);
    }

    @Test
    void queue_age_is_priority_blind() {
        WorkerBatcher batcher = newBatcher();
        long now = System.currentTimeMillis();
        // Ancient head for a large, deterministic age.
        assertTrue(batcher.tryOffer(item(1, 50, now + 60_000, now - 100_000, 128)));
        PrefillQueueManager manager = batcher.queueManager();

        long waitP70 = manager.estimateWaitMs(70, now + 60_000, 999);
        long waitP50 = manager.estimateWaitMs(50, now + 60_000, 998);
        long waitP30 = manager.estimateWaitMs(30, now + 60_000, 997);

        // Priority-blind by design: every probe sees the same measured head
        // age (a small band tolerates wall-clock drift between the probes).
        assertTrue(Math.abs(waitP70 - waitP30) <= 50,
                "probes of any priority must see the same head age: "
                        + waitP70 + "/" + waitP50 + "/" + waitP30);
        assertTrue(waitP30 >= 99_000, "head waited 100s, the age must reflect it: " + waitP30);
    }

    @Test
    void queue_age_depends_on_the_head_not_depth() {
        // Same head age, wildly different depths: the estimate is the same —
        // the head age is the congestion signal. Behavioral pin of the O(1)
        // design (no per-item iteration, deep and shallow queues cost alike).
        WorkerBatcher shallow = newBatcher();
        WorkerBatcher deep = newBatcher();
        long now = System.currentTimeMillis();
        assertTrue(shallow.tryOffer(item(1, 50, now + 60_000, now - 50_000, 128)));
        for (long requestId = 1; requestId <= 100; requestId++) {
            assertTrue(deep.tryOffer(item(requestId, 30, now + 60_000, now - 50_000, 128)));
        }

        long shallowAge = shallow.queueManager().estimateWaitMs(70, now + 60_000, 999);
        long deepAge = deep.queueManager().estimateWaitMs(70, now + 60_000, 999);
        assertTrue(Math.abs(shallowAge - deepAge) <= 50,
                "depth must not change the head-age estimate: " + shallowAge + " vs " + deepAge);
        assertTrue(deepAge >= 49_000, "head waited 50s, the age must reflect it: " + deepAge);
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
