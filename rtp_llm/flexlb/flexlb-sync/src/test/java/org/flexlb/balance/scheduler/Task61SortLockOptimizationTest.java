package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.scheduler.priority.PrefillQueueSnapshot;
import org.flexlb.balance.scheduler.priority.QueuedRequestSnapshot;
import org.flexlb.balance.strategy.PrefillTimePredictor;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.ScheduleBudget;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.flexlb.util.PriorityOrdering;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.PriorityBlockingQueue;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.locks.ReentrantLock;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * task61 sort/lock optimization pins, three suites:
 *
 * <ol>
 *   <li><b>Sort equivalence</b> — {@code sortedHeadItems(k)} ≡ full-sort
 *       prefix (randomized + leader-pinned tie-break scenario: same priority,
 *       same arrival timestamp, different requestId), {@code peek()} ≡ sorted
 *       head, and the L2 {@code estimateWaitMs} count against a full-sort
 *       reference.</li>
 *   <li><b>Outside-lock concurrency (M2)</b> — under concurrent offers and
 *       removals every snapshot stays internally sorted and duplicate-free,
 *       and the "version unchanged ⇒ content unchanged" invariant holds
 *       across snapshot pairs even though the sort now runs outside the
 *       queue lock.</li>
 *   <li><b>Switch two-state</b> — M1/M2 flags on vs off produce identical
 *       dispatch batches and identical snapshots; the M1 top-k path is gated
 *       off on the legacy (non-Auto-TPM) order, which is not total.</li>
 * </ol>
 */
class Task61SortLockOptimizationTest {

    // ==================== 1. sort equivalence ====================

    @Test
    void sortedHeadItemsMatchesFullSortPrefixOnRandomQueues() {
        Random random = new Random(61);
        for (int round = 0; round < 200; round++) {
            int n = random.nextInt(41);
            BatcherContext ctx = autoTpmContext();
            long base = 1_000_000L;
            for (long id = 1; id <= n; id++) {
                // Narrow ranges force plenty of priority and arrival ties so
                // the requestId tie-break is exercised constantly.
                int priority = 30 + 10 * random.nextInt(5);
                long arrival = base + random.nextInt(4);
                offer(ctx, item(id, priority, arrival + 30_000, arrival, 128));
            }
            List<BatchItem> full = ctx.sortedItems();
            for (int k : new int[]{0, 1, 2, n / 2, n, n + 5}) {
                List<BatchItem> head = ctx.sortedHeadItems(k);
                int expected = Math.max(0, Math.min(k, n));
                assertEquals(expected, head.size());
                for (int i = 0; i < head.size(); i++) {
                    assertSame(full.get(i), head.get(i),
                            "top-k element " + i + " must be the identical full-sort prefix element");
                }
            }
        }
    }

    /**
     * Leader-pinned tie-break scenario: same priority + same arrival
     * timestamp, different requestId. The requestId tie-break is exactly what
     * makes {@code AUTO_TPM_QUEUE_ORDER} a total order — the premise behind
     * both peek≡sorted-head (L2) and top-k≡full-sort-prefix (M1). If someone
     * changes the comparator and breaks totality, this fails loudly.
     */
    @Test
    void tieBreakSamePrioritySameArrivalIsResolvedByRequestIdEverywhere() {
        BatcherContext ctx = autoTpmContext();
        long arrival = 1_000_000L;
        for (long id : new long[]{5, 1, 9, 3, 7, 2}) {
            offer(ctx, item(id, 50, arrival + 30_000, arrival, 128));
        }

        List<BatchItem> full = ctx.sortedItems();
        assertEquals(List.of(1L, 2L, 3L, 5L, 7L, 9L), requestIds(full),
                "full sort must break the total tie by requestId asc");

        // peek ≡ sorted head (the L2 estimateWaitMs premise)
        assertSame(full.get(0), ctx.peek());
        // top-k ≡ full-sort prefix under ties (the M1 premise)
        assertEquals(List.of(1L, 2L, 3L), requestIds(ctx.sortedHeadItems(3)));
        assertSame(full.get(0), ctx.sortedHeadItems(1).get(0));
    }

    @Test
    void estimateWaitMsMatchesFullSortReferenceCount() {
        FlexlbConfig config = new FlexlbConfig();
        config.setFlexlbBatchAlgorithm("fixed_window");
        config.setAutoTpmEnabled(true);
        config.setFlexlbBatchSizeMax(2);
        config.setFlexlbBatchFixedWaitMs(200);
        WorkerBatcher batcher = new WorkerBatcher("test-worker", null, config,
                mock(BatchDecisionHandler.class), mock(BatchSchedulerReporter.class));
        long now = System.currentTimeMillis();
        // Ancient arrivals zero out the head's remaining window for determinism
        assertTrue(batcher.tryOffer(item(1, 70, now, now - 100_000, 128)));
        assertTrue(batcher.tryOffer(item(2, 50, now, now - 100_000, 128)));
        assertTrue(batcher.tryOffer(item(3, 50, now, now - 99_000, 128)));
        assertTrue(batcher.tryOffer(item(4, 30, now, now - 100_000, 128)));

        PrefillQueueManager manager = batcher.queueManager();
        // Reference itemsAhead from the full-sort snapshot with the shared
        // ordering rule — what the pre-task61 implementation counted.
        for (int probePriority : new int[]{80, 60, 50, 30, 20}) {
            long probeId = 999;
            int itemsAhead = 0;
            for (QueuedRequestSnapshot queued : manager.snapshot().items()) {
                int cmp = PriorityOrdering.compareStrict(
                        queued.priority(), queued.arrivalTimeMs(), probePriority, now);
                if (cmp < 0 || (cmp == 0 && queued.requestId() < probeId)) {
                    itemsAhead++;
                }
            }
            long expected = (long) (itemsAhead / 2) * 200;
            assertEquals(expected, manager.estimateWaitMs(probePriority, now + 60_000, probeId),
                    "estimate for probe priority " + probePriority
                            + " must equal the full-sort reference count");
        }
    }

    // ==================== 2. outside-lock concurrency (M2) ====================

    @Test
    void snapshotVersionInvariantHoldsUnderConcurrentMutation() throws Exception {
        FlexlbConfig config = new FlexlbConfig();
        config.setFlexlbBatchAlgorithm("fixed_window");
        config.setAutoTpmEnabled(true);
        config.setFlexlbBatchQueueMaxSize(10_000);
        assertTrue(config.isFlexlbSnapshotSortOutsideLockEnabled(), "M2 default must be on");
        WorkerBatcher batcher = new WorkerBatcher("test-worker", null, config,
                mock(BatchDecisionHandler.class), mock(BatchSchedulerReporter.class));
        PrefillQueueManager manager = batcher.queueManager();

        int writers = 3;
        int opsPerWriter = 2_000;
        AtomicBoolean writersDone = new AtomicBoolean(false);
        AtomicLong nextId = new AtomicLong(1);
        ExecutorService pool = Executors.newFixedThreadPool(writers + 2);
        try {
            List<Future<?>> futures = new ArrayList<>();
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
            int cmp = PriorityOrdering.compareStrict(
                    a.priority(), a.arrivalTimeMs(), b.priority(), b.arrivalTimeMs());
            if (cmp == 0) {
                cmp = Long.compare(a.requestId(), b.requestId());
            }
            assertTrue(cmp < 0, "snapshot must be strictly sorted in queue order"
                    + " (got " + a.requestId() + " before " + b.requestId() + ")");
        }
    }

    // ==================== 3. switch two-state ====================

    @Test
    void snapshotIsIdenticalWithSortOutsideLockOnAndOff() {
        FlexlbConfig config = new FlexlbConfig();
        config.setFlexlbBatchAlgorithm("fixed_window");
        config.setAutoTpmEnabled(true);
        WorkerBatcher batcher = new WorkerBatcher("test-worker", null, config,
                mock(BatchDecisionHandler.class), mock(BatchSchedulerReporter.class));
        long now = System.currentTimeMillis();
        assertTrue(batcher.tryOffer(item(3, 50, now + 5_000, now, 128)));
        assertTrue(batcher.tryOffer(item(1, 70, now + 9_000, now + 2, 256)));
        assertTrue(batcher.tryOffer(item(2, 50, now + 1_000, now, 128)));
        assertTrue(batcher.tryOffer(item(4, 30, now + 5_000, now - 100, 64)));

        config.setFlexlbSnapshotSortOutsideLockEnabled(true);
        PrefillQueueSnapshot outside = batcher.queueManager().snapshot();
        config.setFlexlbSnapshotSortOutsideLockEnabled(false);
        PrefillQueueSnapshot inside = batcher.queueManager().snapshot();

        assertEquals(inside, outside, "M2 on/off must produce bit-identical snapshots");
    }

    @Test
    void sloBudgetDispatchesIdenticalBatchWithTopKOnAndOff() throws Exception {
        List<Long> dispatchedWithTopK = dispatchWithFlushTopK(true);
        List<Long> dispatchedWithFullSort = dispatchWithFlushTopK(false);
        assertEquals(dispatchedWithFullSort, dispatchedWithTopK,
                "M1 on/off must dispatch the identical batch");
        assertEquals(List.of(1L, 2L, 3L, 5L, 7L, 9L), dispatchedWithTopK,
                "batch must follow queue order incl. the requestId tie-break");
    }

    private static List<Long> dispatchWithFlushTopK(boolean topKEnabled) throws Exception {
        FlexlbConfig config = new FlexlbConfig();
        config.setFlexlbBatchAlgorithm("slo_budget");
        config.setAutoTpmEnabled(true);
        config.setFlexlbFlushTopKSortEnabled(topKEnabled);
        BatchDecisionHandler handler = mock(BatchDecisionHandler.class);
        BatcherContext ctx = autoTpmContext(config, handler);
        long now = System.currentTimeMillis();
        // Same priority + same arrival: only the requestId tie-break orders
        // them, pinning the total-order premise on the dispatch path too.
        for (long id : new long[]{5, 1, 9, 3, 7, 2}) {
            BatchItem item = item(id, 50, now + 30_000, now - 100, 128);
            item.setSortKey(now - 100); // budget < 0 → deadline_guard dispatch
            offer(ctx, item);
        }

        new SloBudgetBatcherAlgorithm().processQueue(ctx);

        @SuppressWarnings("unchecked")
        ArgumentCaptor<List<BatchItem>> batch = ArgumentCaptor.forClass(List.class);
        verify(handler).onBatchReady(batch.capture(), org.mockito.ArgumentMatchers.any());
        return requestIds(batch.getValue());
    }

    /**
     * Gate pin: the legacy queue order ({@code sortKey} only) is not a total
     * order, so even with the M1 flag on the candidate pick must fall back to
     * the full sort when Auto-TPM is off.
     */
    @Test
    void pickCandidatesUsesFullSortOnLegacyOrderEvenWithFlagOn() throws Exception {
        FlexlbConfig config = new FlexlbConfig();
        config.setFlexlbBatchAlgorithm("slo_budget");
        config.setAutoTpmEnabled(false);
        config.setFlexlbFlushTopKSortEnabled(true);
        PriorityBlockingQueue<BatchItem> queue =
                new PriorityBlockingQueue<>(11, WorkerBatcher.LEGACY_QUEUE_ORDER);
        BatcherContext ctx = new BatcherContext("test", endpoint(), config,
                mock(BatchDecisionHandler.class), queue,
                new AtomicInteger(0), new AtomicLong(), new ReentrantLock(),
                WorkerBatcher.LEGACY_QUEUE_ORDER, mock(BatchSchedulerReporter.class));
        long now = System.currentTimeMillis();
        for (long id = 1; id <= 6; id++) {
            BatchItem item = item(id, 0, now + 30_000, now, 128);
            item.setSortKey(now); // deliberate sortKey ties: legacy order is not total
            offer(ctx, item);
        }

        Method pickCandidates = SloBudgetBatcherAlgorithm.class
                .getDeclaredMethod("pickCandidates", BatcherContext.class, int.class);
        pickCandidates.setAccessible(true);
        @SuppressWarnings("unchecked")
        List<BatchItem> legacy = (List<BatchItem>) pickCandidates.invoke(null, ctx, 2);
        assertEquals(6, legacy.size(), "legacy path must keep the full sorted list");

        // Auto-TPM path with the flag on is bounded by k = maxScan + 1
        config.setAutoTpmEnabled(true);
        BatcherContext autoTpmCtx = autoTpmContext(config, mock(BatchDecisionHandler.class));
        for (long id = 1; id <= 6; id++) {
            offer(autoTpmCtx, item(id, 50, now + 30_000, now, 128));
        }
        @SuppressWarnings("unchecked")
        List<BatchItem> topK = (List<BatchItem>) pickCandidates.invoke(null, autoTpmCtx, 2);
        assertEquals(3, topK.size(), "auto-tpm path must select only maxScan+1 items");
        assertEquals(List.of(1L, 2L, 3L), requestIds(topK));
    }

    // ==================== helpers ====================

    private static BatcherContext autoTpmContext() {
        FlexlbConfig config = new FlexlbConfig();
        config.setFlexlbBatchAlgorithm("slo_budget");
        config.setAutoTpmEnabled(true);
        return autoTpmContext(config, mock(BatchDecisionHandler.class));
    }

    private static BatcherContext autoTpmContext(FlexlbConfig config,
                                                 BatchDecisionHandler handler) {
        PriorityBlockingQueue<BatchItem> queue =
                new PriorityBlockingQueue<>(11, WorkerBatcher.AUTO_TPM_QUEUE_ORDER);
        return new BatcherContext("test", endpoint(), config, handler, queue,
                new AtomicInteger(0), new AtomicLong(), new ReentrantLock(),
                WorkerBatcher.AUTO_TPM_QUEUE_ORDER, mock(BatchSchedulerReporter.class));
    }

    private static PrefillEndpoint endpoint() {
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        PrefillTimePredictor predictor = mock(PrefillTimePredictor.class);
        when(endpoint.getPredictor()).thenReturn(predictor);
        when(endpoint.getInflightBatchCount()).thenReturn(0);
        when(predictor.estimateMs(anyLong(), anyLong())).thenReturn(0L);
        when(predictor.predictBatchMs(anyList())).thenReturn(0.0);
        when(predictor.predictBatchMsUncached(anyList())).thenReturn(0.0);
        return endpoint;
    }

    /** Direct queue insertion for BatcherContext-level tests (no batcher). */
    private static void offer(BatcherContext ctx, BatchItem item) {
        // restorePendingDispatch is not applicable; add through the raw queue
        // the context was constructed around, mirroring WorkerBatcher.offer.
        try {
            java.lang.reflect.Field field = BatcherContext.class.getDeclaredField("queue");
            field.setAccessible(true);
            @SuppressWarnings("unchecked")
            PriorityBlockingQueue<BatchItem> queue =
                    (PriorityBlockingQueue<BatchItem>) field.get(ctx);
            queue.add(item);
            java.lang.reflect.Field depth = BatcherContext.class.getDeclaredField("queueDepth");
            depth.setAccessible(true);
            ((AtomicInteger) depth.get(ctx)).incrementAndGet();
        } catch (ReflectiveOperationException e) {
            throw new IllegalStateException(e);
        }
    }

    private static List<Long> requestIds(List<BatchItem> items) {
        return items.stream().map(BatchItem::requestId).toList();
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
