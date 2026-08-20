package org.flexlb.balance.strategy;

import org.flexlb.balance.scheduler.BatchItem;
import org.flexlb.balance.scheduler.SchedulingTestConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.DebugInfo;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class LearningPredictorTest {

    @Test
    @DisplayName("default model produces a non-negative estimate")
    void defaultModelEstimateMs() {
        LearningPredictor p = new LearningPredictor();
        assertTrue(p.estimateMs(1000, 200) >= 0);
    }

    @Test
    @DisplayName("estimateMs with zero tokens")
    void estimateMsZeroTokens() {
        LearningPredictor p = new LearningPredictor();
        assertTrue(p.estimateMs(0, 0) >= 0);
    }

    @Test
    @DisplayName("estimateMs bounds hitTokens to totalTokens")
    void estimateMsHitTokensBounded() {
        LearningPredictor p = new LearningPredictor();
        assertEquals(p.estimateMs(100, 100), p.estimateMs(100, 500));
    }

    @Test
    @DisplayName("predictBatchMs aggregates correctly")
    void predictBatchMsAggregation() {
        LearningPredictor p = new LearningPredictor();
        BatchItem item1 = batchItem(500, 200);
        BatchItem item2 = batchItem(300, 100);
        assertTrue(p.predictBatchMs(List.of(item1, item2)) >= 0);
    }

    @Test
    @DisplayName("predictBatchMs empty list returns 0")
    void predictBatchMsEmpty() {
        LearningPredictor p = new LearningPredictor();
        assertEquals(0, p.predictBatchMs(List.of()));
    }

    @Test
    @DisplayName("learn accepts completed batches")
    void learnAcceptsCompletedBatches() {
        LearningPredictor p = new LearningPredictor();
        List<BatchItem> batchItems = List.of(
                batchItem(500, 200),
                batchItem(300, 100),
                batchItem(1000, 500));
        PrefillBatchFeatures features = PrefillBatchFeatures.from(batchItems);
        for (int i = 0; i < 4; i++) {
            assertDoesNotThrow(() -> p.learn(features, 300, 400));
        }
        assertEquals(1L, p.generation(),
                "one generation must be published with one weight update");
    }

    @Test
    @DisplayName("generation changes only when new weights are published")
    void generationTracksPublishedLearningUpdates() {
        LearningPredictor p = new LearningPredictor();
        PrefillBatchFeatures features = PrefillBatchFeatures.from(
                List.of(batchItem(500, 200)));

        assertEquals(0L, p.generation());
        for (int i = 0; i < 3; i++) {
            p.learn(features, 300, 400);
            assertEquals(0L, p.generation());
        }
        p.learn(features, 300, 400);
        assertEquals(1L, p.generation());
    }

    @Test
    @DisplayName("concurrent completion learning serializes Adam state")
    void concurrentLearnSerializesOptimizerState() throws Exception {
        LearningPredictor p = new LearningPredictor();
        PrefillBatchFeatures features = PrefillBatchFeatures.from(List.of(
                batchItem(500, 200), batchItem(1000, 500)));
        int sampleCount = 64;
        ExecutorService executor = Executors.newFixedThreadPool(8);
        CountDownLatch start = new CountDownLatch(1);
        List<Future<?>> futures = new ArrayList<>();
        try {
            for (int i = 0; i < sampleCount; i++) {
                futures.add(executor.submit(() -> {
                    start.await();
                    p.learn(features, 300, 400);
                    return null;
                }));
            }
            start.countDown();
            for (Future<?> future : futures) {
                future.get(5, TimeUnit.SECONDS);
            }
        } finally {
            executor.shutdownNow();
        }

        Field step = LearningPredictor.class.getDeclaredField("t");
        step.setAccessible(true);
        assertEquals(1 + sampleCount / 4, step.getLong(p),
                "every four samples must produce exactly one serialized Adam update");
        assertEquals(sampleCount / 4, p.generation(),
                "generation must match the number of published weight updates");
    }

    private static BatchItem batchItem(long seqLen, long hitCacheLen) {
        Request request = new Request();
        request.setRequestId(1L);
        request.setSeqLen(seqLen);

        BalanceContext ctx = new BalanceContext();
        ctx.setConfig(SchedulingTestConfig.batchConfig());
        ctx.setRequest(request);

        ServerStatus prefill = new ServerStatus();
        DebugInfo debugInfo = new DebugInfo();
        debugInfo.setHitCacheLen(hitCacheLen);
        prefill.setDebugInfo(debugInfo);

        return new BatchItem(ctx, null, null, prefill, null, null, null, System.currentTimeMillis());
    }
}
