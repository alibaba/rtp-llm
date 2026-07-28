package org.flexlb.balance.scheduler;

import org.flexlb.balance.endpoint.PrefillEndpoint;
import org.flexlb.balance.strategy.PrefillTimePredictor;
import org.flexlb.config.FlexlbConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.service.monitor.BatchSchedulerReporter;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.anyList;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

@Tag("performance-regression")
class WorkerBatcherPerformanceTest {

    @Test
    void concurrentOfferAndExactRemoveMeetsThroughputFloor() throws Exception {
        int threads = 8;
        int operationsPerThread = 20_000;
        long minimumQps = Long.getLong("flexlb.perf.min-batcher-qps", 100_000L);
        FlexlbConfig config = new FlexlbConfig();
        config.setFlexlbBatchAlgorithm("fixed_window");
        config.setFlexlbBatchQueueMaxSize(0);
        WorkerBatcher batcher = new WorkerBatcher("performance", mock(PrefillEndpoint.class),
                config, mock(BatchDecisionHandler.class), mock(BatchSchedulerReporter.class));
        ExecutorService executor = Executors.newFixedThreadPool(threads);
        CountDownLatch start = new CountDownLatch(1);
        List<Callable<Integer>> tasks = new ArrayList<>();
        for (int thread = 0; thread < threads; thread++) {
            BatchItem item = item(thread + 1L);
            tasks.add(() -> {
                start.await();
                int removed = 0;
                for (int operation = 0; operation < operationsPerThread; operation++) {
                    WorkerBatcher.QueueHandle handle = batcher.offer(item);
                    if (batcher.remove(handle) == WorkerBatcher.RemoveResult.REMOVED) {
                        removed++;
                    }
                }
                return removed;
            });
        }

        try {
            List<Future<Integer>> results = new ArrayList<>();
            for (Callable<Integer> task : tasks) {
                results.add(executor.submit(task));
            }
            long startedAt = System.nanoTime();
            start.countDown();
            int completed = 0;
            for (Future<Integer> result : results) {
                completed += result.get(30, TimeUnit.SECONDS);
            }
            double qps = completed * 1_000_000_000.0 / (System.nanoTime() - startedAt);

            System.out.printf("WorkerBatcher offer/remove performance: threads=%d operations=%d qps=%.1f%n",
                    threads, completed, qps);
            assertEquals(threads * operationsPerThread, completed);
            assertEquals(0, batcher.queueSize());
            assertTrue(qps >= minimumQps,
                    () -> String.format("batcher throughput %.1f QPS is below regression floor %d QPS",
                            qps, minimumQps));
        } finally {
            executor.shutdownNow();
            assertTrue(executor.awaitTermination(3, TimeUnit.SECONDS));
            batcher.shutdown();
        }
    }

    @Test
    void concurrentOfferAndRemoveWithActiveDeepQueueMeetsThroughputFloor() throws Exception {
        int backlog = 1_024;
        int threads = 4;
        int operationsPerThread = 5_000;
        long minimumQps = Long.getLong(
                "flexlb.perf.min-active-batcher-qps", 35_000L);
        FlexlbConfig config = new FlexlbConfig();
        config.setFlexlbBatchAlgorithm("fixed_window");
        config.setFlexlbBatchQueueMaxSize(0);
        config.setFlexlbBatchSizeMax(Integer.MAX_VALUE);
        config.setFlexlbBatchFixedWaitMs(Long.MAX_VALUE);
        config.setFlexlbBatchPredictThresholdMs(1L);
        PrefillEndpoint endpoint = mock(PrefillEndpoint.class);
        PrefillTimePredictor predictor = mock(PrefillTimePredictor.class);
        CountDownLatch queueScanned = new CountDownLatch(1);
        when(endpoint.getPredictor()).thenReturn(predictor);
        when(predictor.predictBatchMs(anyList())).thenAnswer(ignored -> {
            queueScanned.countDown();
            return 0.0;
        });
        WorkerBatcher batcher = new WorkerBatcher("active-performance", endpoint,
                config, mock(BatchDecisionHandler.class), mock(BatchSchedulerReporter.class));
        for (int i = 0; i < backlog; i++) {
            batcher.offer(item(10_000L + i));
        }
        batcher.start();
        assertTrue(queueScanned.await(3, TimeUnit.SECONDS));

        ExecutorService executor = Executors.newFixedThreadPool(threads);
        CountDownLatch start = new CountDownLatch(1);
        List<Callable<Integer>> tasks = new ArrayList<>();
        for (int thread = 0; thread < threads; thread++) {
            BatchItem item = item(20_000L + thread);
            tasks.add(() -> {
                start.await();
                int removed = 0;
                for (int operation = 0; operation < operationsPerThread; operation++) {
                    WorkerBatcher.QueueHandle handle = batcher.offer(item);
                    if (batcher.remove(handle) == WorkerBatcher.RemoveResult.REMOVED) {
                        removed++;
                    }
                }
                return removed;
            });
        }

        try {
            List<Future<Integer>> results = new ArrayList<>();
            for (Callable<Integer> task : tasks) {
                results.add(executor.submit(task));
            }
            long startedAt = System.nanoTime();
            start.countDown();
            int completed = 0;
            for (Future<Integer> result : results) {
                completed += result.get(30, TimeUnit.SECONDS);
            }
            double qps = completed * 1_000_000_000.0
                    / (System.nanoTime() - startedAt);

            System.out.printf("WorkerBatcher active-queue performance: backlog=%d "
                            + "threads=%d operations=%d qps=%.1f%n",
                    backlog, threads, completed, qps);
            assertEquals(threads * operationsPerThread, completed);
            assertEquals(backlog, batcher.queueSize());
            assertTrue(qps >= minimumQps,
                    () -> String.format("active-queue throughput %.1f QPS is below "
                                    + "regression floor %d QPS", qps, minimumQps));
        } finally {
            executor.shutdownNow();
            assertTrue(executor.awaitTermination(3, TimeUnit.SECONDS));
            batcher.shutdown();
        }
    }

    private static BatchItem item(long requestId) {
        Request request = new Request();
        request.setRequestId(requestId);
        request.setSeqLen(128);
        BalanceContext context = new BalanceContext();
        context.setRequest(request);
        return new BatchItem(context, null, null, 0, null, 0);
    }
}
