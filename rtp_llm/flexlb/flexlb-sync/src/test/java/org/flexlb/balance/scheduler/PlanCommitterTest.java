package org.flexlb.balance.scheduler;

import org.flexlb.balance.autotpm.EvictionPlanner;
import org.flexlb.balance.autotpm.PlanCommitter;
import org.flexlb.balance.autotpm.PrefillEvictionPlan;
import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.Test;

import java.util.Comparator;
import java.util.List;
import java.util.concurrent.PriorityBlockingQueue;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;

/**
 * Tests {@link PlanCommitter} against a real {@link BatcherContext}.
 *
 * <p>Lives in the {@code scheduler} test package so it can construct a
 * {@link BatcherContext} (package-private constructor) the same way
 * {@link PriorityDeadlineBatcherAlgorithmTest} does.
 */
class PlanCommitterTest {

    private static final long DEADLINE_MULT = 10_000_000_000_000L;

    private final EvictionPlanner planner = new EvictionPlanner();
    private final PlanCommitter committer = new PlanCommitter();

    // ==================== CAS success ====================

    @Test
    void casSuccess_victimsRemovedAndIncomingOffered() {
        // P30 (victim, eligible) + P70 (survivor, same priority as incoming → not evictable)
        BatchItem victim = priorityItem(1L, 30, 10_000_000L);
        BatchItem survivor = priorityItem(2L, 70, 10_000_000L);
        victim.setSortKey(computeSortKeyRaw(30, 10_000_000L));
        survivor.setSortKey(computeSortKeyRaw(70, 10_000_000L));
        BatcherContext ctx = context(queueWith(victim, survivor));

        // Incoming P70 request
        BatchItem incoming = priorityItem(3L, 70, 9_000_000L);
        incoming.setSortKey(computeSortKeyRaw(70, 9_000_000L));

        // Plan against the current snapshot
        QueueSnapshot snapshot = ctx.snapshot();
        PrefillEvictionPlan plan = planner.plan(snapshot, incoming.priority(),
                incoming.seqLen(), 8);
        assertFalse(plan.isEmpty(), "P30 victim must be eligible for P70 incoming");
        assertEquals(List.of(1L), plan.victimRequestIds(),
                "only P30 is eligible; P70 survivor is protected by the hard rule");

        PlanCommitter.CommitResult result = committer.execute(plan, incoming, ctx);
        assertTrue(result.isSuccess(), "CAS commit must succeed");
        assertEquals(1, result.victims().size(), "exactly one victim removed");
        assertEquals(1L, result.victims().get(0).requestId());

        // Queue now contains survivor (P70) + incoming (P70), victim (P30) gone
        assertEquals(2, ctx.size(), "queue depth: survivor + incoming");
        QueueSnapshot after = ctx.snapshot();
        assertEquals(2, after.queueSize());
        List<Long> ids = after.items().stream()
                .map(QueueSnapshot.ItemSummary::requestId).sorted().toList();
        assertTrue(ids.contains(2L), "survivor still queued");
        assertTrue(ids.contains(3L), "incoming offered");
        assertFalse(ids.contains(1L), "victim removed");
    }

    @Test
    void victimsListReturnedForFutureCompletion() {
        BatchItem v1 = priorityItem(10L, 30, 10_000_000L);
        BatchItem v2 = priorityItem(11L, 40, 10_000_000L);
        v1.setSortKey(computeSortKeyRaw(30, 10_000_000L));
        v2.setSortKey(computeSortKeyRaw(40, 10_000_000L));
        BatcherContext ctx = context(queueWith(v1, v2));

        BatchItem incoming = priorityItem(20L, 70, 9_000_000L);
        incoming.setSortKey(computeSortKeyRaw(70, 9_000_000L));

        PrefillEvictionPlan plan = planner.plan(ctx.snapshot(), 70, 100, 8);
        PlanCommitter.CommitResult result = committer.execute(plan, incoming, ctx);
        assertTrue(result.isSuccess());
        assertEquals(2, result.victims().size());
        // Caller can now complete victims' futures with PRIORITY_PREEMPTED.
        // Removal order follows heap iteration, not plan order, so assert as a set.
        List<Long> victimIds = result.victims().stream()
                .map(BatchItem::requestId).sorted().toList();
        assertEquals(List.of(10L, 11L), victimIds);
    }

    // ==================== CAS version mismatch ====================

    @Test
    void versionMismatch_returnsFailureAndQueueUnchanged() {
        BatchItem victim = priorityItem(1L, 30, 10_000_000L);
        victim.setSortKey(computeSortKeyRaw(30, 10_000_000L));
        BatcherContext ctx = context(queueWith(victim));

        BatchItem incoming = priorityItem(3L, 70, 9_000_000L);
        incoming.setSortKey(computeSortKeyRaw(70, 9_000_000L));

        QueueSnapshot snapshot = ctx.snapshot();
        PrefillEvictionPlan plan = planner.plan(snapshot, 70, 100, 8);
        assertEquals(0L, plan.snapshotVersion(), "version starts at 0");

        // Mutate the queue after snapshot so the version no longer matches
        ctx.remove(victim);
        long newVersion = ctx.version();
        assertTrue(newVersion > plan.snapshotVersion(),
                "version must bump after intervening mutation");

        PlanCommitter.CommitResult result = committer.execute(plan, incoming, ctx);
        assertFalse(result.isSuccess());
        assertEquals("version_mismatch", result.failureReason());
        assertTrue(result.victims().isEmpty(),
                "no victims on version mismatch");
    }

    // ==================== Still-full after (no-op) eviction ====================

    @Test
    void stillFullAfterEviction_returnsVersionMismatch() {
        // Queue full (maxSize=1) with a P30 item. Build a plan that targets a
        // requestId NOT present in the queue, so removal is a no-op and the
        // queue stays full → tryRemoveAndOffer returns null.
        FlexlbConfig cfg = new FlexlbConfig();
        cfg.setFlexlbBatchQueueMaxSize(1);
        BatchItem present = priorityItem(1L, 30, 10_000_000L);
        present.setSortKey(computeSortKeyRaw(30, 10_000_000L));
        BatcherContext ctx = context(cfg, queueWith(present));

        BatchItem incoming = priorityItem(3L, 70, 9_000_000L);
        incoming.setSortKey(computeSortKeyRaw(70, 9_000_000L));

        // Manually construct a plan whose victim id is absent from the queue
        long version = ctx.version();
        PrefillEvictionPlan plan = new PrefillEvictionPlan(
                List.of(999L), // not in queue
                new org.flexlb.balance.autotpm.PlanCost(1L, 1, 0L, 1, 0.0, 999L),
                version);

        PlanCommitter.CommitResult result = committer.execute(plan, incoming, ctx);
        assertFalse(result.isSuccess(), "still-full after no-op eviction must fail");
        assertEquals("version_mismatch", result.failureReason());
        // Incoming was NOT offered
        assertEquals(1, ctx.size(), "queue unchanged: incoming not offered");
        QueueSnapshot after = ctx.snapshot();
        assertTrue(after.items().stream()
                .noneMatch(i -> i.requestId() == 3L));
    }

    // ==================== Empty plan ====================

    @Test
    void emptyPlan_noMutation() {
        BatchItem survivor = priorityItem(1L, 50, 10_000_000L);
        survivor.setSortKey(computeSortKeyRaw(50, 10_000_000L));
        BatcherContext ctx = context(queueWith(survivor));

        BatchItem incoming = priorityItem(2L, 50, 9_000_000L); // same priority → no victims
        incoming.setSortKey(computeSortKeyRaw(50, 9_000_000L));

        PrefillEvictionPlan plan = planner.plan(ctx.snapshot(), 50, 100, 8);
        assertTrue(plan.isEmpty());

        PlanCommitter.CommitResult result = committer.execute(plan, incoming, ctx);
        assertFalse(result.isSuccess());
        assertEquals("empty_plan", result.failureReason());
        assertEquals(1, ctx.size(), "queue untouched on empty plan");
        // Incoming was NOT offered by the committer (caller handles empty plan)
        QueueSnapshot after = ctx.snapshot();
        assertTrue(after.items().stream()
                .noneMatch(i -> i.requestId() == 2L), "incoming not offered");
    }

    // ==================== Helpers ====================

    private static long computeSortKeyRaw(int priority, long deadlineMs) {
        return (long) (70 - priority) * DEADLINE_MULT + deadlineMs;
    }

    private static BatchItem priorityItem(long requestId, int priority, long deadlineMs) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(100);
        BalanceContext ctx = new BalanceContext();
        ctx.setRequest(request);
        ctx.setPriority(priority);
        ctx.setDeadlineMs(deadlineMs);
        BatchItem item = new BatchItem(ctx, new java.util.concurrent.CompletableFuture<>(),
                null, null, null, null, null, System.currentTimeMillis());
        item.setPriority(priority);
        item.setDeadlineMs(deadlineMs);
        return item;
    }

    private static PriorityBlockingQueue<BatchItem> queueWith(BatchItem... items) {
        PriorityBlockingQueue<BatchItem> queue =
                new PriorityBlockingQueue<>(11, Comparator.comparingLong(BatchItem::sortKey));
        for (BatchItem item : items) {
            queue.add(item);
        }
        return queue;
    }

    private static BatcherContext context(PriorityBlockingQueue<BatchItem> queue) {
        return context(new FlexlbConfig(), queue);
    }

    private static BatcherContext context(FlexlbConfig cfg,
                                          PriorityBlockingQueue<BatchItem> queue) {
        return new BatcherContext("test", mock(PrefillEndpoint.class), cfg,
                mock(BatchDecisionHandler.class), queue,
                new AtomicInteger(queue.size()), mock(BatchSchedulerReporter.class));
    }
}
