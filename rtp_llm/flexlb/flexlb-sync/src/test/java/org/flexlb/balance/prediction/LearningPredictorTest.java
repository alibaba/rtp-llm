package org.flexlb.balance.prediction;

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

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

class LearningPredictorTest {

    @Test
    @DisplayName("default model produces a non-negative estimate")
    void defaultModelEstimateMs() {
        LearningPredictor p = new LearningPredictor();
        assertTrue(p.evaluator().estimateMs(1000, 200) >= 0);
    }

    @Test
    @DisplayName("estimateMs with zero tokens")
    void estimateMsZeroTokens() {
        LearningPredictor p = new LearningPredictor();
        assertTrue(p.evaluator().estimateMs(0, 0) >= 0);
    }

    @Test
    @DisplayName("estimateMs bounds hitTokens to totalTokens")
    void estimateMsHitTokensBounded() {
        LearningPredictor p = new LearningPredictor();
        assertEquals(p.evaluator().estimateMs(100, 100), p.evaluator().estimateMs(100, 500));
    }

    @Test
    @DisplayName("predictBatchMs aggregates correctly")
    void predictBatchMsAggregation() {
        LearningPredictor p = new LearningPredictor();
        PrefillBatchFeatures features = batchFeatures(
                item(500, 200), item(300, 100));
        assertTrue(p.evaluator().predictBatchMs(features) >= 0);
    }

    @Test
    @DisplayName("predictBatchMs empty features return 0")
    void predictBatchMsEmpty() {
        LearningPredictor p = new LearningPredictor();
        assertEquals(0, p.evaluator().predictBatchMs(batchFeatures()));
    }

    @Test
    @DisplayName("learn accepts completed batches")
    void learnAcceptsCompletedBatches() {
        LearningPredictor p = new LearningPredictor();
        PrefillBatchFeatures features = batchFeatures(
                item(500, 200), item(300, 100), item(1000, 500));
        PrefillTimePredictor.Evaluator initial = p.evaluator();
        for (int i = 0; i < 4; i++) {
            assertDoesNotThrow(() -> p.learn(features, 300, 400));
        }
        assertNotSame(initial, p.evaluator(),
                "four samples must publish one replacement evaluator");
    }

    @Test
    @DisplayName("evaluator changes only when new weights are published")
    void evaluatorTracksPublishedLearningUpdates() {
        LearningPredictor p = new LearningPredictor();
        PrefillBatchFeatures features = batchFeatures(item(500, 200));
        PrefillTimePredictor.Evaluator initial = p.evaluator();

        for (int i = 0; i < 3; i++) {
            p.learn(features, 300, 400);
            assertSame(initial, p.evaluator());
        }
        p.learn(features, 300, 400);
        assertNotSame(initial, p.evaluator());
    }

    @Test
    @DisplayName("weights advance in one immutable evaluator snapshot")
    void weightsAdvanceInOneSnapshot() throws Exception {
        LearningPredictor p = new LearningPredictor();
        PrefillBatchFeatures features = batchFeatures(item(500, 200));

        PrefillTimePredictor.Evaluator before = p.evaluator();
        double[] beforeWeights = snapshotWeights(before);
        for (int i = 0; i < 4; i++) {
            p.learn(features, 300, 400);
        }
        PrefillTimePredictor.Evaluator after = p.evaluator();
        double[] afterWeights = snapshotWeights(after);

        assertNotSame(before, after,
                "an update must publish a new immutable snapshot");
        assertFalse(Arrays.equals(beforeWeights, afterWeights),
                "the replacement evaluator must carry updated weights");
        assertArrayEquals(beforeWeights, snapshotWeights(before),
                "publishing the next model must not mutate the old snapshot");
    }

    @Test
    @DisplayName("concurrent readers observe immutable evaluator snapshots")
    void concurrentReadersObserveAtomicModelSnapshots() throws Exception {
        LearningPredictor p = new LearningPredictor();
        PrefillBatchFeatures features = batchFeatures(
                item(500, 200), item(1000, 500));
        int updateCount = 128;
        int readerCount = 4;
        ExecutorService executor = Executors.newFixedThreadPool(readerCount + 1);
        CountDownLatch start = new CountDownLatch(1);
        AtomicBoolean writerDone = new AtomicBoolean();
        Map<PrefillTimePredictor.Evaluator, double[]> weightsByEvaluator =
                new ConcurrentHashMap<>();
        List<PrefillTimePredictor.Evaluator> published = new ArrayList<>();
        List<Future<?>> futures = new ArrayList<>();
        try {
            PrefillTimePredictor.Evaluator initial = p.evaluator();
            weightsByEvaluator.put(initial, snapshotWeights(initial));
            published.add(initial);
            futures.add(executor.submit(() -> {
                start.await();
                try {
                    for (int update = 0; update < updateCount; update++) {
                        for (int sample = 0; sample < 4; sample++) {
                            p.learn(features, 300, 400 + update);
                        }
                        published.add(p.evaluator());
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
                    do {
                        PrefillTimePredictor.Evaluator evaluator = p.evaluator();
                        recordSnapshot(weightsByEvaluator, evaluator);
                    } while (!writerDone.get());
                    recordSnapshot(weightsByEvaluator, p.evaluator());
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

        PrefillTimePredictor.Evaluator finalEvaluator = p.evaluator();
        assertEquals(updateCount + 1L, published.stream().distinct().count(),
                "every completed learning batch must publish a fresh evaluator");
        assertArrayEquals(snapshotWeights(finalEvaluator),
                weightsByEvaluator.get(finalEvaluator));
        assertTrue(weightsByEvaluator.size() >= 2,
                "the stress run must observe both the initial and learned models");
    }

    @Test
    @DisplayName("concurrent completion learning serializes Adam state")
    void concurrentLearnSerializesOptimizerState() throws Exception {
        LearningPredictor p = new LearningPredictor();
        PrefillBatchFeatures features = batchFeatures(
                item(500, 200), item(1000, 500));
        PrefillTimePredictor.Evaluator initial = p.evaluator();
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
        assertNotSame(initial, p.evaluator(),
                "serialized updates must publish a replacement evaluator");
    }

    private static double[] snapshotWeights(
            PrefillTimePredictor.Evaluator evaluator)
            throws ReflectiveOperationException {
        Field weights = evaluator.getClass().getDeclaredField("weights");
        weights.setAccessible(true);
        return ((double[]) weights.get(evaluator)).clone();
    }

    private static void recordSnapshot(
            Map<PrefillTimePredictor.Evaluator, double[]> weightsByEvaluator,
            PrefillTimePredictor.Evaluator evaluator)
            throws ReflectiveOperationException {
        double[] observed = snapshotWeights(evaluator);
        weightsByEvaluator.compute(evaluator, (ignored, existing) -> {
            if (existing != null) {
                assertArrayEquals(existing, observed,
                        "one evaluator must identify exactly one immutable weight set");
                return existing;
            }
            return observed;
        });
    }

    private static PrefillBatchFeatures batchFeatures(
            PrefillBatchFeatures.Item... items) {
        return new PrefillBatchFeatures(List.of(items));
    }

    private static PrefillBatchFeatures.Item item(long seqLen, long hitCache) {
        return new PrefillBatchFeatures.Item(seqLen, hitCache);
    }
}
