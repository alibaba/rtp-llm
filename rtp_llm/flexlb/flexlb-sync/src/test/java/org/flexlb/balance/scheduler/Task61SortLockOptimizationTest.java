package org.flexlb.balance.scheduler;

import org.flexlb.balance.scheduler.priority.PrefillQueueSnapshot;
import org.flexlb.balance.scheduler.priority.QueuedRequestSnapshot;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.SchedulingMetadata;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.Test;

import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
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
 * Invariant pins for the outside-lock snapshot sort in
 * {@link PrefillQueueManager#snapshot()}, two suites (intent-ported from
 * auto-tpm v3 core 51af09456f onto the decision/delivery-split queue
 * architecture):
 *
 * <ol>
 *   <li><b>Sort equivalence</b> — the outside-lock snapshot produces
 *       exactly the full-sort order of
 *       {@link WorkerBatcher#PRIORITY_QUEUE_ORDER} (priority desc →
 *       enqueue-seq asc → requestId asc), including the same-priority
 *       construction-order tie-break: a scrambled arrival timestamp never
 *       reorders same-priority members.</li>
 *   <li><b>Outside-lock concurrency</b> — under concurrent offers and
 *       removals every snapshot stays priority-sorted and duplicate-free,
 *       and the "version unchanged ⇒ content unchanged" invariant holds
 *       across snapshot pairs even though the sort runs outside the queue
 *       lock.</li>
 * </ol>
 */
class Task61SortLockOptimizationTest {

    // ==================== 1. sort equivalence ====================

    @Test
    void snapshotMatchesFullSortOrderIncludingTieBreak() {
        FlexlbConfig config = new FlexlbConfig();
        SchedulingTestConfig.usePriorityQueue(config);
        WorkerBatcher batcher = newBatcher(config);
        long now = System.currentTimeMillis();
        // Constructed in id order 1..5. Same-priority pairs (1,5) and (2,3)
        // deliberately get arrival timestamps OPPOSITE to construction
        // order: enqueue-seq (assigned at construction) is the same-priority
        // rule, so the arrival timestamp must never reorder them.
        assertTrue(batcher.tryOffer(item(config, 1, 70, now + 5_000, now + 4, 256)));
        assertTrue(batcher.tryOffer(item(config, 2, 50, now + 5_000, now + 9, 128)));
        assertTrue(batcher.tryOffer(item(config, 3, 50, now + 5_000, now + 1, 128)));
        assertTrue(batcher.tryOffer(item(config, 4, 30, now + 5_000, now - 100, 64)));
        assertTrue(batcher.tryOffer(item(config, 5, 70, now + 5_000, now, 128)));

        PrefillQueueSnapshot snapshot = batcher.queueManager().snapshot();
        assertEquals(List.of(1L, 5L, 2L, 3L, 4L), requestIds(snapshot.items()),
                "outside-lock sort must equal the PRIORITY_QUEUE_ORDER full-sort order");
    }

    // ==================== 2. outside-lock concurrency ====================

    @Test
    void snapshotVersionInvariantHoldsUnderConcurrentMutation() throws Exception {
        FlexlbConfig config = new FlexlbConfig();
        SchedulingTestConfig.usePriorityQueue(config);
        SchedulingTestConfig.useQueueCapacity(config)
                .setMaxWaitingRequestsPerPrefillWorker(10_000);
        WorkerBatcher batcher = newBatcher(config);
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
                        batcher.tryOffer(item(config, id, priority,
                                arrival + 30_000, arrival, 128));
                        if (random.nextInt(4) == 0) {
                            // Remove a random recently-offered request
                            manager.tryRemove(1 + random.nextLong(id), "task61-test");
                        }
                    }
                    return null;
                }));
            }
            // Readers: priority-sorted + unique per snapshot; same version
            // ⇒ same content — the atomicity pin for the lock-held
            // copy+version capture with the lock-free sort.
            for (int r = 0; r < 2; r++) {
                futures.add(pool.submit(() -> {
                    PrefillQueueSnapshot previous = null;
                    int sameVersionPairs = 0;
                    while (!writersDone.get() || sameVersionPairs == 0) {
                        PrefillQueueSnapshot snapshot = manager.snapshot();
                        assertPrioritySortedAndUnique(snapshot.items());
                        if (previous != null
                                && previous.queueVersion() == snapshot.queueVersion()) {
                            assertEquals(previous.items(), snapshot.items(),
                                    "version unchanged ⇒ content unchanged "
                                            + "(outside-lock sort invariant)");
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

    /**
     * The snapshot items are not observable by enqueue-seq, so the
     * concurrent suite pins the priority-descending skeleton plus
     * duplicate-freedom; the deterministic tie-break suite above pins the
     * full order.
     */
    private static void assertPrioritySortedAndUnique(List<QueuedRequestSnapshot> items) {
        Set<Long> seen = new HashSet<>();
        for (QueuedRequestSnapshot item : items) {
            assertTrue(seen.add(item.requestId()),
                    "snapshot must be duplicate-free (requestId="
                            + item.requestId() + " twice)");
        }
        for (int i = 1; i < items.size(); i++) {
            assertTrue(items.get(i - 1).priority() >= items.get(i).priority(),
                    "snapshot must be priority-descending (got "
                            + items.get(i - 1).priority() + " before "
                            + items.get(i).priority() + ")");
        }
    }

    // ==================== helpers ====================

    private static WorkerBatcher newBatcher(FlexlbConfig config) {
        return new WorkerBatcher("test-worker", null, config,
                mock(DecisionGroupHandler.class),
                TestCapacityAdmission.alwaysAvailable(),
                mock(BatchSchedulerReporter.class));
    }

    private static List<Long> requestIds(List<QueuedRequestSnapshot> items) {
        return items.stream().map(QueuedRequestSnapshot::requestId).toList();
    }

    private static BatchItem item(FlexlbConfig config, long requestId, int priority,
                                  long expiresAtMs, long enqueuedAtMs, long seqLen) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(seqLen);
        request.setPriority(priority);
        BalanceContext ctx = new BalanceContext();
        ctx.setRequest(request);
        ctx.setConfig(config);
        ctx.setSchedulingMetadata(SchedulingMetadata.explicit(priority, expiresAtMs));
        return new BatchItem(ctx, new CompletableFuture<>(), null,
                null, null, null, null, enqueuedAtMs);
    }
}
