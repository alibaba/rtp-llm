package org.flexlb.balance.scheduler;

import org.flexlb.balance.scheduler.priority.PrefillQueueSnapshot;
import org.flexlb.balance.scheduler.priority.QueuedRequestSnapshot;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.ScheduleBudget;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Random;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;

/**
 * task61 M2 (snapshot sort outside the queue lock) pins, two suites:
 *
 * <ol>
 *   <li><b>Sort equivalence</b> — the outside-lock snapshot produces exactly
 *       the full-sort order of {@code WorkerBatcher.AUTO_TPM_QUEUE_ORDER}
 *       (priority desc → enqueue-seq asc → requestId asc), including the
 *       same-priority/same-arrival requestId tie-break.</li>
 *   <li><b>Outside-lock concurrency</b> — under concurrent offers and
 *       removals every snapshot stays internally sorted and duplicate-free,
 *       and the "version unchanged ⇒ content unchanged" invariant holds
 *       across snapshot pairs even though the sort now runs outside the
 *       queue lock.</li>
 * </ol>
 */
class Task61SortLockOptimizationTest {

    // ==================== 1. sort equivalence ====================

    @Test
    void snapshotMatchesFullSortOrderIncludingTieBreak() {
        FlexlbConfig config = new FlexlbConfig();
        config.setFlexlbBatchAlgorithm("fixed_window");
        config.setAutoTpmEnabled(true);
        WorkerBatcher batcher = new WorkerBatcher("test-worker", null, config,
                mock(BatchDecisionHandler.class), mock(BatchSchedulerReporter.class));
        long now = System.currentTimeMillis();
        // Narrow priority/arrival ranges force ties so the requestId
        // tie-break is exercised: id 1 and 5 share (priority 70, arrival
        // now+2); id 2 and 3 share (priority 50, arrival now).
        assertTrue(batcher.tryOffer(item(3, 50, now + 5_000, now, 128)));
        assertTrue(batcher.tryOffer(item(1, 70, now + 9_000, now + 2, 256)));
        assertTrue(batcher.tryOffer(item(2, 50, now + 1_000, now, 128)));
        assertTrue(batcher.tryOffer(item(4, 30, now + 5_000, now - 100, 64)));
        assertTrue(batcher.tryOffer(item(5, 70, now + 9_000, now + 2, 128)));

        PrefillQueueSnapshot snapshot = batcher.queueManager().snapshot();
        assertEquals(List.of(1L, 5L, 2L, 3L, 4L), requestIds(snapshot.items()),
                "outside-lock sort must equal the AUTO_TPM full-sort order");
    }

    // ==================== 2. outside-lock concurrency (M2) ====================

    @Test
    void snapshotVersionInvariantHoldsUnderConcurrentMutation() throws Exception {
        FlexlbConfig config = new FlexlbConfig();
        config.setFlexlbBatchAlgorithm("fixed_window");
        config.setAutoTpmEnabled(true);
        config.setFlexlbBatchQueueMaxSize(10_000);
        WorkerBatcher batcher = new WorkerBatcher("test-worker", null, config,
                mock(BatchDecisionHandler.class), mock(BatchSchedulerReporter.class));
        PrefillQueueManager manager = batcher.queueManager();

        int writers = 3;
        int opsPerWriter = 2_000;
        AtomicBoolean writersDone = new AtomicBoolean(false);
        AtomicLong nextId = new AtomicLong(1);
        ExecutorService pool = Executors.newFixedThreadPool(writers + 2);
        try {
            List<Future<?>> futures = new java.util.ArrayList<>();
            for (int w = 0; w < writers; w++) {
                futures.add(pool.submit(() -> {
                    Random random = new Random(Thread.currentThread().threadId());
                    long now = System.currentTimeMillis();
                    for (int i = 0; i < opsPerWriter; i++) {
                        long id = nextId.getAndIncrement();
                        int priority = 30 + 10 * random.nextInt(5);
                        long arrival = now + random.nextInt(8);
                        batcher.tryOffer(item(id, priority, arrival + 30_000, arrival, 128));
                        if (random.nextInt(4) == 0) {
                            // Remove a random recently-offered request
                            manager.tryRemove(1 + random.nextLong(id), "task61-test");
                        }
                    }
                    return null;
                }));
            }
            // Readers: sorted+unique per snapshot; same version ⇒ same content
            for (int r = 0; r < 2; r++) {
                futures.add(pool.submit(() -> {
                    PrefillQueueSnapshot previous = null;
                    int sameVersionPairs = 0;
                    while (!writersDone.get() || sameVersionPairs == 0) {
                        PrefillQueueSnapshot snapshot = manager.snapshot();
                        assertSortedAndUnique(snapshot.items());
                        if (previous != null
                                && previous.queueVersion() == snapshot.queueVersion()) {
                            assertEquals(previous.items(), snapshot.items(),
                                    "version unchanged ⇒ content unchanged (M2 invariant)");
                            sameVersionPairs++;
                        }
                        previous = snapshot;
                    }
                    return null;
                }));
            }
            // Complete writers first, then let readers observe the quiescent
            // queue (guarantees at least one same-version pair).
            for (int i = 0; i < writers; i++) {
                futures.get(i).get();
            }
            writersDone.set(true);
            for (Future<?> future : futures) {
                future.get();
            }
        } finally {
            pool.shutdownNow();
        }
    }

    private static void assertSortedAndUnique(List<QueuedRequestSnapshot> items) {
        for (int i = 1; i < items.size(); i++) {
            QueuedRequestSnapshot a = items.get(i - 1);
            QueuedRequestSnapshot b = items.get(i);
            // The AUTO_TPM_QUEUE_ORDER rule: priority desc, then enqueue-seq
            // (arrival) asc, then requestId asc.
            int cmp = Integer.compare(b.priority(), a.priority());
            if (cmp == 0) {
                cmp = Long.compare(a.arrivalTimeMs(), b.arrivalTimeMs());
            }
            if (cmp == 0) {
                cmp = Long.compare(a.requestId(), b.requestId());
            }
            assertTrue(cmp < 0, "snapshot must be strictly sorted in queue order"
                    + " (got " + a.requestId() + " before " + b.requestId() + ")");
        }
    }

    // ==================== helpers ====================

    private static List<Long> requestIds(List<QueuedRequestSnapshot> items) {
        return items.stream().map(QueuedRequestSnapshot::requestId).toList();
    }

    private static BatchItem item(long requestId, int priority, long deadlineMs,
                                  long enqueuedAtMs, long seqLen) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(seqLen);
        request.setPriority(priority);
        BalanceContext ctx = new BalanceContext();
        ctx.setRequest(request);
        ctx.setBudget(ScheduleBudget.forDeadline(priority, enqueuedAtMs, deadlineMs));
        return new BatchItem(ctx, new CompletableFuture<>(), null,
                null, null, null, null, enqueuedAtMs);
    }
}
