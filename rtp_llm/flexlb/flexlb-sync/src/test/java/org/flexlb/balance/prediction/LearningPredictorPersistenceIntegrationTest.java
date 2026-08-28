package org.flexlb.balance.prediction;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.io.IOException;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Collaboration tests for {@link LearningPredictor} wired to a real
 * {@link LearningPredictorPersistence}: the learn path persisting state,
 * restart restore, cold-start refit semantics and regression protection for
 * the persistence-less historical behavior.
 */
class LearningPredictorPersistenceIntegrationTest {

    @Test
    @DisplayName("learn 跨过 saveInterval 后模型快照与历史完整落盘")
    void learnAcrossSaveIntervalPersistsModelSnapshot(@TempDir Path tempDir) throws Exception {
        Path stateFile = tempDir.resolve("state.json");
        LearningPredictor predictor = new LearningPredictor(
                new LearningPredictorPersistence(stateFile, 100, 8), 0);
        for (int i = 0; i < 8; i++) {
            predictor.learn(features(1000L + i * 100L, 300L), 100L, 400L + i);
        }
        assertEquals(2L, predictor.generation(),
                "eight samples must publish two batched weight updates");
        assertTrue(Files.exists(stateFile),
                "crossing saveInterval must write the state file");

        SnapshotState live = new SnapshotState(
                predictor.generation(), predictor.weightsSnapshot());
        LearningPredictorPersistence.LoadedState loaded =
                new LearningPredictorPersistence(stateFile, 100, 8).load();
        assertArrayEquals(live.weights(), loaded.weights(),
                "the persisted weights must match the live model snapshot");
        assertEquals(live.generation(), loaded.generation(),
                "the persisted generation must match the live model snapshot");
        assertEquals(8, loaded.history().size(),
                "every learned sample must enter the rolling history");
    }

    @Test
    @DisplayName("saveInterval 与批大小不对齐时仍持久化当时的模型")
    void saveIntervalNotAlignedWithBatchStillPersistsCurrentModel(@TempDir Path tempDir)
            throws Exception {
        Path stateFile = tempDir.resolve("state.json");
        LearningPredictor predictor = new LearningPredictor(
                new LearningPredictorPersistence(stateFile, 100, 5), 0);
        for (int i = 0; i < 3; i++) {
            assertEquals(PrefillTimePredictor.LearningResult.MODEL_UNCHANGED,
                    predictor.learn(features(1000L, 300L), 100L, 400L),
                    "the first three samples must not fill one learning batch");
        }
        assertEquals(PrefillTimePredictor.LearningResult.MODEL_UPDATED,
                predictor.learn(features(1000L, 300L), 100L, 400L),
                "the fourth sample fills the batch and publishes new weights");
        assertEquals(PrefillTimePredictor.LearningResult.MODEL_UNCHANGED,
                predictor.learn(features(1000L, 300L), 100L, 400L),
                "the fifth sample starts a new batch but crosses saveInterval");
        assertTrue(Files.exists(stateFile),
                "the throttled save must not wait for a full learning batch");

        LearningPredictorPersistence.LoadedState loaded =
                new LearningPredictorPersistence(stateFile, 100, 5).load();
        assertEquals(1L, loaded.generation(),
                "only the fourth sample published weights, so the saved generation is one");
        assertEquals(5, loaded.history().size(),
                "all five samples must already be persisted");
        assertArrayEquals(weightsOf(predictor), loaded.weights(),
                "the persisted model must be the current live weights");
    }

    @Test
    @DisplayName("重启后恢复 weights、generation 继续递增且历史承接新样本")
    void restartRestoresWeightsAndContinuesGenerationAndHistory(@TempDir Path tempDir)
            throws Exception {
        Path stateFile = tempDir.resolve("state.json");
        LearningPredictor first = new LearningPredictor(
                new LearningPredictorPersistence(stateFile, 100, 8), 0);
        for (int i = 0; i < 8; i++) {
            first.learn(features(1000L + i * 100L, 300L), 100L, 400L + i);
        }
        double[] savedWeights = weightsOf(first);
        assertEquals(2L, first.generation(),
                "the first life must publish two batched updates");

        LearningPredictor restarted = new LearningPredictor(
                new LearningPredictorPersistence(stateFile, 100, 8), 0);
        assertEquals(2L, restarted.generation(),
                "a restart must resume the saved generation instead of resetting it");
        assertArrayEquals(savedWeights, weightsOf(restarted),
                "a restart must restore the saved weights");

        for (int i = 0; i < 8; i++) {
            restarted.learn(features(5000L + i * 100L, 300L), 100L, 600L + i);
        }
        assertEquals(4L, restarted.generation(),
                "learning after a restart must continue from the restored generation");
        LearningPredictorPersistence.LoadedState reloaded =
                new LearningPredictorPersistence(stateFile, 100, 8).load();
        assertEquals(4L, reloaded.generation(),
                "the second save must persist the advanced generation");
        assertEquals(16, reloaded.history().size(),
                "the reloaded history must keep old samples and absorb new ones");
    }

    @Test
    @DisplayName("参数损坏但历史保留时启动触发冷启动 refit")
    void corruptedParametersTriggerColdStartRefit(@TempDir Path tempDir) throws Exception {
        Path stateFile = prepareCorruptedStateWithHistory(tempDir, 8);

        LearningPredictor predictor = new LearningPredictor(
                new LearningPredictorPersistence(stateFile, 100, 1000), 3);
        assertFalse(Arrays.equals(initialWeights(), weightsOf(predictor)),
                "the cold-start refit must move the weights off their initial values");
        assertEquals(6L, predictor.generation(),
                "eight samples over three epochs must publish 2*3 batched updates");
        assertEquals(7L, tOf(predictor),
                "the refit must drive exactly six Adam steps starting from t=1");
    }

    @Test
    @DisplayName("refitEpochs 为 0 或负值时安全跳过 refit")
    void nonPositiveRefitEpochsSkipsRefit(@TempDir Path tempDir) throws Exception {
        Path stateFile = prepareCorruptedStateWithHistory(tempDir, 8);

        LearningPredictor zeroEpochs = new LearningPredictor(
                new LearningPredictorPersistence(stateFile, 100, 1000), 0);
        assertArrayEquals(initialWeights(), weightsOf(zeroEpochs),
                "refitEpochs=0 must keep the built-in initial weights");
        assertEquals(0L, zeroEpochs.generation(),
                "refitEpochs=0 must publish no weight updates");

        LearningPredictor negativeEpochs = new LearningPredictor(
                new LearningPredictorPersistence(stateFile, 100, 1000), -7);
        assertArrayEquals(initialWeights(), weightsOf(negativeEpochs),
                "negative refitEpochs must be treated as 'no refit'");
        assertEquals(0L, negativeEpochs.generation(),
                "negative refitEpochs must publish no weight updates");
    }

    @Test
    @DisplayName("refit 按时间序重放历史且与在线学习路径逐位等价")
    void refitReplaysHistoryInTimeOrderLikeOnlineLearning(@TempDir Path tempDir)
            throws Exception {
        List<LearningPredictorPersistence.LearningSample> samples = new ArrayList<>();
        for (int i = 0; i < 8; i++) {
            samples.add(new LearningPredictorPersistence.LearningSample(
                    features(1000L + i * 100L, 300L), 400L + i * 25L));
        }
        LearningPredictor online = new LearningPredictor();
        for (LearningPredictorPersistence.LearningSample sample : samples) {
            online.learn(sample.features(), 100L, sample.actualMs());
        }
        assertEquals(2L, online.generation(),
                "the online path must publish two batched updates");

        Path stateFile = prepareCorruptedStateWithHistory(tempDir, samples);
        LearningPredictor refitted = new LearningPredictor(
                new LearningPredictorPersistence(stateFile, 100, 1000), 1);
        assertArrayEquals(weightsOf(online), weightsOf(refitted),
                "one refit epoch must reproduce the online learning trajectory bit-exactly");
        assertEquals(online.generation(), refitted.generation(),
                "refit and online learning must publish the same generation");
    }

    @Test
    @DisplayName("冷启动 refit 从保留历史中学到已知规律（MAE 显著低于初始权重）")
    void refitLearnsFromRetainedHistory(@TempDir Path tempDir) throws Exception {
        double[] teacherWeights = initialWeights().clone();
        teacherWeights[0] += 15.0;
        teacherWeights[2] += 2.5;
        teacherWeights[3] *= 1.1;
        LearningPredictor teacher = new LearningPredictor();
        installWeights(teacher, 0L, teacherWeights);

        int sampleCount = 500;
        Path stateFile = tempDir.resolve("state.json");
        LearningPredictorPersistence writer =
                new LearningPredictorPersistence(stateFile, 2000, 1000000);
        long[] actuals = new long[sampleCount];
        for (int i = 0; i < sampleCount; i++) {
            long seq = 1000L + i * 77L;
            long hit = seq / 3;
            actuals[i] = teacher.estimateMs(seq, hit);
            writer.recordSample(features(seq, hit), actuals[i]);
        }
        writer.save(initialWeights(), 0L);
        tamperParamCount(stateFile);

        LearningPredictor student = new LearningPredictor(
                new LearningPredictorPersistence(stateFile, 2000, 1000000), 40);
        double refitMae = maeOverSamples(student, sampleCount, actuals);
        double initialMae = maeOverSamples(new LearningPredictor(), sampleCount, actuals);
        assertTrue(refitMae < initialMae,
                "the refit must fit the retained history better than the initial weights"
                        + " (refit MAE=" + refitMae + ", initial MAE=" + initialMae + ")");
    }

    @Test
    @DisplayName("恢复后的在线学习延续权重、Adam 从 t=1 重启且不劣化到初始水平")
    void restoredModelContinuesOnlineLearningAboveColdStart(@TempDir Path tempDir)
            throws Exception {
        double[] teacherWeights = initialWeights().clone();
        teacherWeights[0] += 15.0;
        teacherWeights[3] *= 1.1;
        LearningPredictor teacher = new LearningPredictor();
        installWeights(teacher, 0L, teacherWeights);

        Path stateFile = tempDir.resolve("state.json");
        LearningPredictor first = new LearningPredictor(
                new LearningPredictorPersistence(stateFile, 2000, 8), 0);
        long[] actuals = new long[800];
        for (int i = 0; i < 800; i++) {
            long seq = 1000L + i * 77L;
            long hit = seq / 3;
            actuals[i] = teacher.estimateMs(seq, hit);
            first.learn(features(seq, hit), 100L, actuals[i]);
        }
        assertTrue(maeOverSamples(first, 800, actuals)
                        < maeOverSamples(new LearningPredictor(), 800, actuals),
                "precondition: the first life must actually learn from the teacher samples");
        double[] learnedWeights = weightsOf(first);

        LearningPredictor restarted = new LearningPredictor(
                new LearningPredictorPersistence(stateFile, 2000, 8), 0);
        assertArrayEquals(learnedWeights, weightsOf(restarted),
                "the restart must resume from the learned weights");
        assertEquals(200L, restarted.generation(),
                "the restart must resume from the learned generation");
        assertEquals(201L, tOf(restarted),
                "the Adam step counter must resume one past the restored generation");

        long[] freshActuals = new long[8];
        for (int i = 0; i < 8; i++) {
            long seq = 30000L + i * 911L;
            long hit = seq / 3;
            freshActuals[i] = teacher.estimateMs(seq, hit);
            restarted.learn(features(seq, hit), 100L, freshActuals[i]);
            first.learn(features(seq, hit), 100L, freshActuals[i]);
        }
        assertFalse(Arrays.equals(learnedWeights, weightsOf(restarted)),
                "online learning must keep advancing the restored weights");
        double restartedMae = maeOverSamples(restarted, 8, freshActuals, 30000L, 911L);
        double continuousMae = maeOverSamples(first, 8, freshActuals, 30000L, 911L);
        double initialMae = maeOverSamples(new LearningPredictor(), 8, freshActuals,
                30000L, 911L);
        assertTrue(restartedMae < initialMae,
                "the restored model must not degrade to the initial level on fresh samples"
                        + " (restored MAE=" + restartedMae + ", initial MAE=" + initialMae
                        + ")");
        assertTrue(continuousMae < initialMae,
                "the continuous learner must also stay below the initial level"
                        + " (continuous MAE=" + continuousMae + ", initial MAE="
                        + initialMae + ")");
        assertEquals(202L, restarted.generation(),
                "800+8 samples must publish 202 batched updates in total");
    }

    @Test
    @DisplayName("persistence=null 时与无参构造行为完全一致（回归保护）")
    void nullPersistenceKeepsHistoricalBehavior() throws Exception {
        LearningPredictor legacy = new LearningPredictor();
        LearningPredictor withNull = new LearningPredictor(null, 100);
        assertArrayEquals(weightsOf(legacy), weightsOf(withNull),
                "null persistence must reproduce the no-arg constructor weights");
        assertEquals(legacy.generation(), withNull.generation(),
                "null persistence must reproduce the no-arg generation");

        PrefillBatchFeatures sample = features(1000L, 300L);
        for (int i = 0; i < 4; i++) {
            withNull.learn(sample, 100L, 400L);
            legacy.learn(sample, 100L, 400L);
        }
        assertEquals(1L, withNull.generation(),
                "four samples must publish one update with null persistence");
        assertArrayEquals(weightsOf(legacy), weightsOf(withNull),
                "learning with null persistence must match the historical path exactly");
        assertEquals(legacy.estimateMs(2000L, 500L), withNull.estimateMs(2000L, 500L),
                "predictions must stay identical without persistence");
        assertDoesNotThrow(() -> new LearningPredictor(null, -1),
                "null persistence with odd refitEpochs must still construct cleanly");
    }

    @Test
    @DisplayName("flushState 固化未跨节流窗口的模型与样本且幂等")
    void flushStatePersistsCurrentModelAndIsIdempotent(@TempDir Path tempDir)
            throws Exception {
        Path stateFile = tempDir.resolve("state.json");
        LearningPredictor predictor = new LearningPredictor(
                new LearningPredictorPersistence(stateFile, 100, 8), 0);
        for (int i = 0; i < 4; i++) {
            predictor.learn(features(1000L + i * 100L, 300L), 100L, 400L + i);
        }
        assertEquals(1L, predictor.generation(),
                "four samples must publish one batched update before the flush");
        assertFalse(Files.exists(stateFile),
                "the throttled save must not have fired below saveInterval");

        predictor.flushState();

        LearningPredictorPersistence.LoadedState loaded =
                new LearningPredictorPersistence(stateFile, 100, 8).load();
        assertArrayEquals(weightsOf(predictor), loaded.weights(),
                "the flushed weights must match the live model snapshot");
        assertEquals(1L, loaded.generation(),
                "the flushed generation must match the live model snapshot");
        assertEquals(4, loaded.history().size(),
                "the flush must persist trailing samples that never crossed saveInterval");

        predictor.flushState();
        LearningPredictorPersistence.LoadedState flushedAgain =
                new LearningPredictorPersistence(stateFile, 100, 8).load();
        assertArrayEquals(loaded.weights(), flushedAgain.weights(),
                "a repeated flush must be idempotent");
        assertEquals(loaded.generation(), flushedAgain.generation(),
                "a repeated flush must not advance the generation");

        assertDoesNotThrow(() -> new LearningPredictor(null, 0).flushState(),
                "a persistence-less predictor must treat flushState as a no-op");
    }

    @Test
    @DisplayName("refit 结果未跨节流窗口时由 flushState 固化")
    void flushStatePersistsRefitIncrement(@TempDir Path tempDir) throws Exception {
        Path stateFile = prepareCorruptedStateWithHistory(tempDir, 8);
        LearningPredictor predictor = new LearningPredictor(
                new LearningPredictorPersistence(stateFile, 100, 1000), 3);
        assertEquals(6L, predictor.generation(),
                "the constructor refit must publish six batched updates");
        double[] refitWeights = weightsOf(predictor);
        assertFalse(Arrays.equals(initialWeights(), refitWeights),
                "the refit must have moved the weights off their initial values");

        predictor.flushState();

        LearningPredictorPersistence.LoadedState loaded =
                new LearningPredictorPersistence(stateFile, 100, 1000).load();
        assertArrayEquals(refitWeights, loaded.weights(),
                "the flush must fix up the refit result on disk");
        assertEquals(6L, loaded.generation(),
                "the flushed generation must carry the refit advances");
        assertEquals(8, loaded.history().size(),
                "the flush must keep the retained history intact");
    }

    @Test
    @DisplayName("refit 发散到非有限权重时回退到 refit 前快照并重置优化器")
    void divergedRefitRollsBackToPreRefitWeights() throws Exception {
        LearningPredictor predictor = new LearningPredictor();
        // w5 * (reuse*compute) overflows to infinity, so the very first replay
        // step poisons the whole weight vector with NaN/Inf.
        double[] evilWeights = initialWeights().clone();
        evilWeights[5] = 1e300;
        installWeights(predictor, 0L, evilWeights);

        List<LearningPredictorPersistence.LearningSample> history = List.of(
                new LearningPredictorPersistence.LearningSample(
                        features(1_000_000_000_000L, 500_000_000_000L), 400L));
        invokeRefit(predictor, history, 2);

        assertArrayEquals(evilWeights, weightsOf(predictor),
                "a diverging refit must roll back to the pre-refit weights");
        assertEquals(0L, predictor.generation(),
                "the rollback must restore the pre-refit generation");
        assertEquals(1L, tOf(predictor),
                "the rollback must reset the Adam optimizer to a pristine state");
    }

    @Test
    @DisplayName("allFinite 识别有限、NaN 与 Inf 权重")
    void allFiniteDetectsNonFiniteWeights() {
        assertTrue(LearningPredictor.allFinite(new double[] {1.0, -2.5, 0.0, 1e300}),
                "finite weights, however large, must pass");
        assertTrue(LearningPredictor.allFinite(new double[] {}),
                "an empty weight vector is vacuously finite");
        assertFalse(LearningPredictor.allFinite(new double[] {1.0, Double.NaN}),
                "a NaN weight must fail the finite check");
        assertFalse(LearningPredictor.allFinite(
                        new double[] {Double.POSITIVE_INFINITY, 1.0}),
                "a +Inf weight must fail the finite check");
        assertFalse(LearningPredictor.allFinite(
                        new double[] {1.0, Double.NEGATIVE_INFINITY}),
                "a -Inf weight must fail the finite check");
    }

    private static void invokeRefit(LearningPredictor predictor,
            List<LearningPredictorPersistence.LearningSample> history, int epochs)
            throws Exception {
        Method refit = LearningPredictor.class.getDeclaredMethod("refit", List.class, int.class);
        refit.setAccessible(true);
        refit.invoke(predictor, history, epochs);
    }

    private static Path prepareCorruptedStateWithHistory(@TempDir Path tempDir, int sampleCount)
            throws Exception {
        List<LearningPredictorPersistence.LearningSample> samples = new ArrayList<>();
        for (int i = 0; i < sampleCount; i++) {
            samples.add(new LearningPredictorPersistence.LearningSample(
                    features(1000L + i * 100L, 300L), 400L + i * 25L));
        }
        return prepareCorruptedStateWithHistory(tempDir, samples);
    }

    private static Path prepareCorruptedStateWithHistory(@TempDir Path tempDir,
            List<LearningPredictorPersistence.LearningSample> samples) throws Exception {
        Path stateFile = tempDir.resolve("state.json");
        LearningPredictorPersistence writer =
                new LearningPredictorPersistence(stateFile, 100, 1000000);
        for (LearningPredictorPersistence.LearningSample sample : samples) {
            writer.recordSample(sample.features(), sample.actualMs());
        }
        writer.save(initialWeights(), 5L);
        tamperParamCount(stateFile);
        return stateFile;
    }

    private static void tamperParamCount(Path stateFile) throws IOException {
        String json = Files.readString(stateFile);
        assertTrue(json.contains("\"paramCount\":9"),
                "the saved state must carry the nine-parameter count");
        Files.writeString(stateFile, json.replace("\"paramCount\":9", "\"paramCount\":8"));
    }

    private static double maeOverSamples(LearningPredictor predictor, int sampleCount,
            long[] actuals) {
        return maeOverSamples(predictor, sampleCount, actuals, 1000L, 77L);
    }

    private static double maeOverSamples(LearningPredictor predictor, int sampleCount,
            long[] actuals, long seqBase, long seqStep) {
        double sum = 0.0;
        for (int i = 0; i < sampleCount; i++) {
            long seq = seqBase + i * seqStep;
            long hit = seq / 3;
            sum += Math.abs(predictor.estimateMs(seq, hit) - actuals[i]);
        }
        return sum / sampleCount;
    }

    private static PrefillBatchFeatures features(long seqLen, long hitCache) {
        return new PrefillBatchFeatures(
                List.of(new PrefillBatchFeatures.Item(seqLen, hitCache)));
    }

    @SuppressWarnings("unchecked")
    private static AtomicReference<Object> modelReference(LearningPredictor predictor)
            throws ReflectiveOperationException {
        Field modelRef = LearningPredictor.class.getDeclaredField("modelRef");
        modelRef.setAccessible(true);
        return (AtomicReference<Object>) modelRef.get(predictor);
    }

    private static double[] weightsOf(LearningPredictor predictor) {
        return predictor.weightsSnapshot();
    }

    private static long tOf(LearningPredictor predictor) throws ReflectiveOperationException {
        Field step = LearningPredictor.class.getDeclaredField("t");
        step.setAccessible(true);
        return step.getLong(predictor);
    }

    private static double[] initialWeights() {
        return new LearningPredictor().weightsSnapshot();
    }

    /**
     * Replaces the published model weights through reflection and rewinds the
     * Adam step counter to match the installed generation (teacher models).
     */
    private static void installWeights(LearningPredictor predictor, long generation,
            double[] weights) throws ReflectiveOperationException {
        AtomicReference<Object> modelRef = modelReference(predictor);
        Class<?> modelType = modelRef.get().getClass();
        Constructor<?> constructor = modelType.getDeclaredConstructor(double[].class);
        constructor.setAccessible(true);
        modelRef.set(constructor.newInstance(weights.clone()));
        Field step = LearningPredictor.class.getDeclaredField("t");
        step.setAccessible(true);
        step.setLong(predictor, generation + 1L);
    }

    private record SnapshotState(long generation, double[] weights) {
    }
}
