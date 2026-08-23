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
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
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
    @DisplayName("generation and weights advance in one immutable model snapshot")
    void generationAndWeightsAdvanceInOneSnapshot() throws Exception {
        LearningPredictor p = new LearningPredictor();
        PrefillBatchFeatures features = PrefillBatchFeatures.from(
                List.of(batchItem(500, 200)));
        AtomicReference<Object> modelRef = modelReference(p);

        Object before = modelRef.get();
        SnapshotState beforeState = snapshotState(before);
        for (int i = 0; i < 4; i++) {
            p.learn(features, 300, 400);
        }
        Object after = modelRef.get();
        SnapshotState afterState = snapshotState(after);

        assertNotSame(before, after,
                "an update must publish a new immutable snapshot");
        assertEquals(beforeState.generation() + 1L, afterState.generation());
        assertEquals(afterState.generation(), p.generation());
        assertFalse(Arrays.equals(beforeState.weights(), afterState.weights()),
                "the published generation must carry its updated weights");
        assertArrayEquals(beforeState.weights(), snapshotState(before).weights(),
                "publishing the next model must not mutate the old snapshot");
    }

    @Test
    @DisplayName("concurrent readers never observe two weight sets for one generation")
    void concurrentReadersObserveAtomicModelSnapshots() throws Exception {
        LearningPredictor p = new LearningPredictor();
        PrefillBatchFeatures features = PrefillBatchFeatures.from(List.of(
                batchItem(500, 200), batchItem(1000, 500)));
        AtomicReference<Object> modelRef = modelReference(p);
        int updateCount = 128;
        int readerCount = 4;
        ExecutorService executor = Executors.newFixedThreadPool(readerCount + 1);
        CountDownLatch start = new CountDownLatch(1);
        AtomicBoolean writerDone = new AtomicBoolean();
        Map<Long, double[]> weightsByGeneration = new ConcurrentHashMap<>();
        List<Future<?>> futures = new ArrayList<>();
        try {
            weightsByGeneration.put(0L, snapshotState(modelRef.get()).weights());
            futures.add(executor.submit(() -> {
                start.await();
                try {
                    for (int update = 0; update < updateCount; update++) {
                        for (int sample = 0; sample < 4; sample++) {
                            p.learn(features, 300, 400 + update);
                        }
                        Thread.yield();
                    }
                } finally {
                    writerDone.set(true);
                }
                return null;
            }));
            for (int reader = 0; reader < readerCount; reader++) {
                futures.add(executor.submit(() -> {
                    start.await();
                    long lastGeneration = -1L;
                    do {
                        SnapshotState state = snapshotState(modelRef.get());
                        assertTrue(state.generation() >= lastGeneration,
                                "a reader must not observe model generations moving backwards");
                        lastGeneration = state.generation();
                        recordSnapshot(weightsByGeneration, state);
                    } while (!writerDone.get());
                    recordSnapshot(weightsByGeneration,
                            snapshotState(modelRef.get()));
                    return null;
                }));
            }
            start.countDown();
            for (Future<?> future : futures) {
                future.get(10, TimeUnit.SECONDS);
            }
        } finally {
            executor.shutdownNow();
        }

        SnapshotState finalState = snapshotState(modelRef.get());
        assertEquals(updateCount, finalState.generation());
        assertEquals(finalState.generation(), p.generation());
        assertArrayEquals(finalState.weights(),
                weightsByGeneration.get(finalState.generation()));
        assertTrue(weightsByGeneration.size() >= 2,
                "the stress run must observe both the initial and learned models");
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

    @SuppressWarnings("unchecked")
    private static AtomicReference<Object> modelReference(LearningPredictor predictor)
            throws ReflectiveOperationException {
        Field modelRef = LearningPredictor.class.getDeclaredField("modelRef");
        modelRef.setAccessible(true);
        return (AtomicReference<Object>) modelRef.get(predictor);
    }

    private static SnapshotState snapshotState(Object snapshot)
            throws ReflectiveOperationException {
        Field generation = snapshot.getClass().getDeclaredField("generation");
        generation.setAccessible(true);
        Field weights = snapshot.getClass().getDeclaredField("weights");
        weights.setAccessible(true);
        return new SnapshotState(generation.getLong(snapshot),
                ((double[]) weights.get(snapshot)).clone());
    }

    private static void recordSnapshot(Map<Long, double[]> weightsByGeneration,
                                       SnapshotState state) {
        weightsByGeneration.compute(state.generation(), (generation, existing) -> {
            if (existing != null) {
                assertArrayEquals(existing, state.weights(),
                        "one generation must identify exactly one immutable weight set");
                return existing;
            }
            return state.weights();
        });
    }

    private record SnapshotState(long generation, double[] weights) {
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
