package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.strategy.PrefillTimePredictor;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import java.util.List;
import java.util.Map;
import java.util.concurrent.CancellationException;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.mockito.Mockito.withSettings;

class WorkerBatcherOwnershipTest {

    @Test
    void removingQueuedHandleIsStableAndReleasesCapacity() {
        RecordingHandler handler = new RecordingHandler();
        WorkerBatcher batcher = batcher(1, 60_000L, handler);
        try {
            WorkerBatcher.QueueHandle first = batcher.offer(item(1L));

            assertNotNull(first);
            assertEquals(1, batcher.queueSize());
            assertEquals(WorkerBatcher.RemoveResult.REMOVED,
                    batcher.remove(first));
            assertEquals(WorkerBatcher.RemoveResult.REMOVED,
                    batcher.remove(first));
            assertEquals(0, batcher.queueSize());

            assertNotNull(batcher.offer(item(2L)),
                    "removed owner must release bounded queue capacity");
            assertEquals(1, batcher.queueSize());
        } finally {
            batcher.shutdown();
        }
    }

    @Test
    @Timeout(value = 5, unit = TimeUnit.SECONDS)
    void claimedHandleCannotBeRemovedByLateCancellation() throws Exception {
        RecordingHandler handler = new RecordingHandler();
        WorkerBatcher batcher = batcher(1, 0L, handler);
        WorkerBatcher.QueueHandle handle = batcher.offer(item(3L));

        batcher.start();
        try {
            assertTrue(handler.claimed.await(3, TimeUnit.SECONDS));
            assertEquals(WorkerBatcher.RemoveResult.CLAIMED,
                    batcher.remove(handle));
            assertEquals(WorkerBatcher.RemoveResult.CLAIMED,
                    batcher.remove(handle));
            assertEquals(1, handler.readyCount.get());
            assertEquals(0, batcher.queueSize());
        } finally {
            batcher.shutdown();
        }
    }

    @Test
    @Timeout(value = 5, unit = TimeUnit.SECONDS)
    void externalRemovalClearsSloParkTrace() throws Exception {
        RecordingHandler handler = new RecordingHandler();
        WorkerBatcher batcher = sloBatcher(handler);
        batcher.offer(item(4L));
        batcher.start();
        try {
            awaitParkTraceCount(batcher, 1);

            BatcherContext context = batcherContext(batcher);
            BatchItem queued = context.peek();
            WorkerBatcher.QueueHandle handle = handleFor(context, queued);
            assertEquals(WorkerBatcher.RemoveResult.REMOVED, batcher.remove(handle));

            awaitParkTraceCount(batcher, 0);
        } finally {
            batcher.shutdown();
        }
    }

    @Test
    @Timeout(value = 5, unit = TimeUnit.SECONDS)
    void externalRemovalBeforeParkDoesNotRecreateSloTrace() throws Exception {
        CountDownLatch predictionStarted = new CountDownLatch(1);
        CountDownLatch releasePrediction = new CountDownLatch(1);
        CountDownLatch batchPredictionCompleted = new CountDownLatch(1);
        PrefillTimePredictor predictor = mock(PrefillTimePredictor.class);
        when(predictor.estimateMs(anyLong(), anyLong())).thenReturn(0L);
        when(predictor.predictBatchMsUncached(anyList())).thenAnswer(ignored -> {
            predictionStarted.countDown();
            assertTrue(releasePrediction.await(3, TimeUnit.SECONDS));
            return 0.0;
        });
        when(predictor.predictBatchMs(anyList())).thenAnswer(ignored -> {
            batchPredictionCompleted.countDown();
            return 0.0;
        });

        WorkerBatcher batcher = sloBatcher(new RecordingHandler(), predictor);
        WorkerBatcher.QueueHandle handle = batcher.offer(item(5L));
        batcher.start();
        try {
            assertTrue(predictionStarted.await(3, TimeUnit.SECONDS));
            assertEquals(WorkerBatcher.RemoveResult.REMOVED, batcher.remove(handle));
            releasePrediction.countDown();
            assertTrue(batchPredictionCompleted.await(3, TimeUnit.SECONDS));
            awaitWorkerWaiting(batcher);

            assertEquals(0, parkTraceCount(batcher));
        } finally {
            releasePrediction.countDown();
            batcher.shutdown();
        }
    }

    private static WorkerBatcher batcher(int queueSize,
                                         long waitMs,
                                         BatchDecisionHandler handler) {
        FlexlbConfig config = new FlexlbConfig();
        config.setFlexlbBatchAlgorithm("fixed_window");
        config.setFlexlbBatchSizeMax(1);
        config.setFlexlbBatchQueueMaxSize(queueSize);
        config.setFlexlbBatchFixedWaitMs(waitMs);
        config.setFlexlbBatchMaxCapacity(Integer.MAX_VALUE);
        return new WorkerBatcher(
                "ownership-test",
                mock(PrefillEndpoint.class, withSettings().stubOnly()),
                config,
                handler,
                mock(BatchSchedulerReporter.class, withSettings().stubOnly()));
    }

    private static WorkerBatcher sloBatcher(BatchDecisionHandler handler) {
        PrefillTimePredictor predictor = mock(PrefillTimePredictor.class);
        when(predictor.estimateMs(anyLong(), anyLong())).thenReturn(0L);
        when(predictor.predictBatchMsUncached(anyList())).thenReturn(0.0);
        when(predictor.predictBatchMs(anyList())).thenReturn(0.0);
        return sloBatcher(handler, predictor);
    }

    private static WorkerBatcher sloBatcher(BatchDecisionHandler handler,
                                             PrefillTimePredictor predictor) {
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        when(endpoint.getPredictor()).thenReturn(predictor);

        return new WorkerBatcher(
                "slo-ownership-test",
                endpoint,
                sloConfig(),
                handler,
                mock(BatchSchedulerReporter.class, withSettings().stubOnly()));
    }

    private static FlexlbConfig sloConfig() {
        FlexlbConfig config = new FlexlbConfig();
        config.setFlexlbBatchAlgorithm("slo_budget");
        config.setFlexlbBatchSizeMax(32);
        config.setFlexlbBatchQueueMaxSize(32);
        config.setFlexlbBatchWindowMs(10);
        config.setCostSloMs(60_000L);
        config.setFlexlbBatchMaxCapacity(Integer.MAX_VALUE);
        return config;
    }

    private static void awaitWorkerWaiting(WorkerBatcher batcher) throws Exception {
        java.lang.reflect.Field field = WorkerBatcher.class.getDeclaredField("workerThread");
        field.setAccessible(true);
        Thread worker = (Thread) field.get(batcher);
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(3);
        while (System.nanoTime() < deadline) {
            if (worker.getState() == Thread.State.WAITING) {
                return;
            }
            TimeUnit.MILLISECONDS.sleep(5);
        }
        assertEquals(Thread.State.WAITING, worker.getState());
    }

    private static void awaitParkTraceCount(WorkerBatcher batcher, int expected) throws Exception {
        long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(3);
        while (System.nanoTime() < deadline) {
            if (parkTraceCount(batcher) == expected) {
                return;
            }
            TimeUnit.MILLISECONDS.sleep(5);
        }
        assertEquals(expected, parkTraceCount(batcher));
    }

    private static int parkTraceCount(WorkerBatcher batcher) throws Exception {
        java.lang.reflect.Field algorithmField = WorkerBatcher.class.getDeclaredField("algorithm");
        algorithmField.setAccessible(true);
        Object algorithm = algorithmField.get(batcher);
        java.lang.reflect.Field tracesField = SloBudgetBatcherAlgorithm.class
                .getDeclaredField("lastParkByRequest");
        tracesField.setAccessible(true);
        return ((Map<?, ?>) tracesField.get(algorithm)).size();
    }

    private static BatcherContext batcherContext(WorkerBatcher batcher) throws Exception {
        java.lang.reflect.Field field = WorkerBatcher.class.getDeclaredField("ctx");
        field.setAccessible(true);
        return (BatcherContext) field.get(batcher);
    }

    @SuppressWarnings("unchecked")
    private static WorkerBatcher.QueueHandle handleFor(BatcherContext context,
                                                        BatchItem item) throws Exception {
        java.lang.reflect.Field field = BatcherContext.class.getDeclaredField("handles");
        field.setAccessible(true);
        return ((Map<BatchItem, WorkerBatcher.QueueHandle>) field.get(context)).get(item);
    }

    private static BatchItem item(long requestId) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(1L);
        BalanceContext context = new BalanceContext();
        context.setRequest(request);
        return new BatchItem(
                context,
                null,
                null,
                0,
                null,
                System.currentTimeMillis());
    }

    private static final class RecordingHandler implements BatchDecisionHandler {
        private final CountDownLatch claimed = new CountDownLatch(1);
        private final AtomicInteger readyCount = new AtomicInteger();

        @Override
        public void onExpired(BatchItem head) {
            throw new AssertionError("unexpected expiry");
        }

        @Override
        public void onBatchReady(List<BatchItem> items, DispatchMeta meta) {
            readyCount.incrementAndGet();
            claimed.countDown();
        }

        @Override
        public void onOfferFailure(BatchItem item, Throwable error) {
            if (error instanceof CancellationException) {
                return;
            }
            throw new AssertionError("unexpected queue rejection", error);
        }
    }
}
