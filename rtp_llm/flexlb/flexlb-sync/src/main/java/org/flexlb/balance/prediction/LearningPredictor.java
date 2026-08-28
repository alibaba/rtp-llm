package org.flexlb.balance.prediction;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.Collectors;

/**
 * Prefill-time predictor with linear regression and online Adam-optimizer learning.
 *
 * <p>
 * Formula: {@code y = w0*1 + w1*batchSize + w2*sum(reuse) + w3*sum(compute)
 * + w4*sum(compute^2) + w5*sum(reuse*compute)}
 * where {@code reuse = hitCache / 1024}, {@code compute = (seqLen - hitCache) / 1024}.
 *
 * <p>
 * The immutable model evaluator is atomically published. The
 * {@link #learn(PrefillBatchFeatures, long, long)}
 * callback uses an Adam optimizer to perform online gradient descent on
 * completed batches.
 *
 * <p>
 * An optional {@link LearningPredictorPersistence} adds cross-restart state:
 * the constructor restores saved weights or refits retained history on a cold
 * start, and every learned sample feeds the rolling history and its throttled
 * state-file saves. {@code null} persistence keeps the historical
 * in-memory-only behavior.
 */
public class LearningPredictor implements PrefillTimePredictor {
    private record BatchUpdateItem(PrefillBatchFeatures features, long actualMs) {
    }

    /**
     * Atomically published prediction model. The weights array is owned by the
     * evaluator and is never mutated after it is constructed.
     */
    private static final class ModelEvaluator implements Evaluator {
        private final double[] weights;

        private ModelEvaluator(double[] weights) {
            this.weights = weights.clone();
        }

        @Override
        public long estimateMs(long totalTokens, long hitTokens) {
            long seq = Math.max(0L, totalTokens);
            long hit = Math.max(0L, Math.min(hitTokens, seq));
            double thisReuse = hit / 1024.0;
            double thisCompute = (seq - hit) / 1024.0;
            double[] inputs = new double[weights.length];
            inputs[0] = 1.0;
            inputs[1] = 1.0;
            inputs[2] = thisReuse;
            inputs[3] = thisCompute;
            inputs[4] = thisCompute * thisCompute;
            inputs[5] = thisReuse * thisCompute;
            double linear = calcLinear(inputs, weights);
            double[] values = new double[5];
            calcNonLinear(weights, linear, values);
            return (long) values[0];
        }

        @Override
        public double predictBatchMs(PrefillBatchFeatures features) {
            if (logger.isDebugEnabled()) {
                logger.debug("learn predictor predictBatchMs: {}, items count: {}",
                        formulaStringParam(weights), features.batchSize());
            }
            if (features.items().isEmpty()) {
                return 0;
            }
            double[] inputs = collectInput(features);
            double linear = calcLinear(inputs, weights);
            double[] values = new double[5];
            calcNonLinear(weights, linear, values);
            return values[0];
        }

        private double[] weightsCopy() {
            return weights.clone();
        }

        private int parameterCount() {
            return weights.length;
        }
    }

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");

    private static final int LINEAR_PARAM_COUNT = 6;
    private static final double COFF1 = 0.005;
    private static final double COFF2 = 0.02;
    private static final double COFF3 = 320;

    private final AtomicReference<ModelEvaluator> modelRef;
    private final double[] adamMoment1;
    private final double[] adamMoment2;
    private final double beta1 = 0.9;
    private final double beta2 = 0.95;
    private final double epsilon = 1e-20;
    private final double alpha = 0.022;
    private long t = 1;
    private final int batchSize = 4;
    private final List<BatchUpdateItem> itemBatch;
    /** Optional persistence; null keeps the historical in-memory-only behavior. */
    private final LearningPredictorPersistence persistence;

    public LearningPredictor() {
        this(null, 0);
    }

    /**
     * @param persistence optional state persistence; null disables persistence
     * @param refitEpochs  epochs replayed over retained history when the saved
     *                     parameters are unusable (cold start)
     */
    public LearningPredictor(LearningPredictorPersistence persistence, int refitEpochs) {
        ModelEvaluator initialModel = new ModelEvaluator(
                new double[] { -4.40538432604287, 10.522208701202377, 1.5043093890711503,
                               21.40103419118763, 0.11145680735428248, 0.08305932028650383,
                               1.451617309598213, 1.0268830123611967, -4.405384326042869 });
        this.modelRef = new AtomicReference<>(initialModel);
        this.adamMoment1 = new double[initialModel.parameterCount()];
        this.adamMoment2 = new double[initialModel.parameterCount()];
        this.itemBatch = new ArrayList<>();
        this.persistence = persistence;
        if (persistence != null) {
            restoreFromPersistence(refitEpochs);
        }
        logger.debug(
                "learn predictor created, t: {}, total param {}, init param: {}, "
                        + "beta1: {}, beta2: {}, alpha: {}, batchSize: {}",
                this.t, initialModel.parameterCount(),
                formulaStringParam(this.modelRef.get().weightsCopy()),
                this.beta1, this.beta2, this.alpha, this.batchSize);
    }

    /** Restore saved weights, or rebuild them by refitting retained history. */
    private void restoreFromPersistence(int refitEpochs) {
        LearningPredictorPersistence.LoadedState loaded = this.persistence.load();
        if (loaded.weights() != null
                && loaded.weights().length == this.modelRef.get().weightsCopy().length) {
            this.modelRef.set(new ModelEvaluator(loaded.weights()));
            // The persisted generation counts completed weight publications; the
            // Adam step counter starts one past it, matching the online path
            // where each publication bumps both together.
            this.t = loaded.generation() + 1L;
            logger.debug("learn predictor state restored: generation: {}, history: {}",
                    loaded.generation(), loaded.history().size());
            return;
        }
        if (!loaded.history().isEmpty()) {
            refit(loaded.history(), refitEpochs);
        }
    }

    /**
     * Cold-start refit: replay the retained history in chronological order,
     * applying the same batched Adam updates as online learning. The Adam
     * optimizer state starts fresh; only the weights are rebuilt. A refit
     * that ends on non-finite weights is rolled back to the pre-refit
     * evaluator (the built-in initial weights in production) with a clean
     * optimizer state, so a diverging replay can never poison predictions.
     */
    private void refit(List<LearningPredictorPersistence.LearningSample> history, int epochs) {
        if (epochs <= 0) {
            return;
        }
        ModelEvaluator preRefit = this.modelRef.get();
        List<BatchUpdateItem> items = new ArrayList<>(history.size());
        for (LearningPredictorPersistence.LearningSample sample : history) {
            items.add(new BatchUpdateItem(sample.features(), sample.actualMs()));
        }
        for (int epoch = 0; epoch < epochs; epoch++) {
            for (int start = 0; start < items.size(); start += this.batchSize) {
                int end = Math.min(start + this.batchSize, items.size());
                updateWeights(items.subList(start, end));
            }
        }
        if (!allFinite(this.modelRef.get().weightsCopy())) {
            rollbackRefit(preRefit, items.size(), epochs);
            return;
        }
        logger.info("learn predictor cold-start refit completed: samples: {}, epochs: {}, generation: {}",
                items.size(), epochs, this.t - 1L);
    }

    /** Restore the pre-refit evaluator and a pristine optimizer state. */
    private void rollbackRefit(ModelEvaluator preRefit, int samples, int epochs) {
        this.modelRef.set(preRefit);
        Arrays.fill(this.adamMoment1, 0.0);
        Arrays.fill(this.adamMoment2, 0.0);
        this.t = 1;
        this.itemBatch.clear();
        logger.error("learn predictor cold-start refit diverged to non-finite weights, "
                + "reverting to the initial weights: samples: {}, epochs: {}", samples, epochs);
    }

    /** True when every weight is finite; guards refit divergence rollbacks. */
    static boolean allFinite(double[] weights) {
        for (double weight : weights) {
            if (!Double.isFinite(weight)) {
                return false;
            }
        }
        return true;
    }

    @Override
    public Evaluator evaluator() {
        return modelRef.get();
    }

    /**
     * Defensive copy of the currently published model parameters.
     *
     * <p>The nine weights map one-to-one onto the navi_sched non-linear
     * prefill-model parameters and feed the NAVI_BATCH PGD optimizer as its
     * per-node latency parameter row. The array is a snapshot copy; callers may
     * mutate it freely without affecting the atomically published model.
     *
     * @return a fresh {@code double[]} holding the model weights, or
     *         {@code null} when the published parameters are not finite
     */
    public double[] weightsSnapshot() {
        double[] weights = this.modelRef.get().weightsCopy();
        return allFinite(weights) ? weights : null;
    }

    /**
     * Generation of the currently published weights: the number of completed
     * batched Adam publications since construction (or since the last
     * restore). Mirrors the persisted generation counter via {@code t - 1}.
     */
    public synchronized long generation() {
        return this.t - 1L;
    }

    /** Single-request estimate delegating to the currently published evaluator. */
    public long estimateMs(long totalTokens, long hitTokens) {
        return this.modelRef.get().estimateMs(totalTokens, hitTokens);
    }

    private static double calcLinear(double[] inputs, double[] weights) {
        double sum = 0.0;
        for (int i = 0; i < inputs.length; i++) {
            sum += inputs[i] * weights[i];
        }
        return sum / COFF3;
    }

    private static void calcNonLinear(
            double[] weights, double linearOutput, double[] output) {
        // param6 / coff1 + param7 / coff2 * ((linear + 1 + p8) + Sqrt((linear + 1 + p8)^2 + 4))
        double p6 = weights[LINEAR_PARAM_COUNT] / COFF1;
        double p7 = weights[LINEAR_PARAM_COUNT + 1] / COFF2;
        double p8 = weights[LINEAR_PARAM_COUNT + 2] + 1.0;
        double linearAddP8 = linearOutput + p8;
        double sqrt_value = Math.sqrt(linearAddP8 * linearAddP8 + 4.0);
        double non_linear_value = linearAddP8 + sqrt_value;
        double predict = p6 + p7 * non_linear_value;
        double grad = p7 * (1.0 + linearAddP8 / sqrt_value);
        double p6_grad = 1.0 / COFF1;
        double p7_grad = non_linear_value / COFF2;
        double p8_grad = grad;
        output[0] = predict;
        output[1] = grad;
        output[2] = p6_grad;
        output[3] = p7_grad;
        output[4] = p8_grad;
    }

    private static double[] collectInput(PrefillBatchFeatures features) {
        double reuse = 0.0;
        double compute = 0.0;
        double compute_square = 0.0;
        double reuse_mul_compute = 0.0;
        for (PrefillBatchFeatures.Item item : features.items()) {
            long seq = Math.max(0L, item.seqLen());
            long hit = Math.max(0L, Math.min(item.hitCache(), seq));
            double thisReuse = hit / 1024.0;
            double thisCompute = (seq - hit) / 1024.0;
            reuse += thisReuse;
            compute += thisCompute;
            compute_square += thisCompute * thisCompute;
            reuse_mul_compute += thisReuse * thisCompute;
        }
        double[] inputs = new double[LINEAR_PARAM_COUNT];
        inputs[0] = 1.0;
        inputs[1] = (double) features.batchSize();
        inputs[2] = reuse;
        inputs[3] = compute;
        inputs[4] = compute_square;
        inputs[5] = reuse_mul_compute;
        return inputs;
    }

    @Override
    public synchronized LearningResult learn(
            PrefillBatchFeatures features, long predictedMs, long actualMs) {
        boolean saveDue = this.persistence != null
                && this.persistence.recordSample(features, actualMs);
        this.itemBatch.add(new BatchUpdateItem(features, actualMs));
        if (this.itemBatch.size() < this.batchSize) {
            persistIfDue(saveDue);
            return LearningResult.MODEL_UNCHANGED;
        }
        updateWeights(this.itemBatch);
        this.itemBatch.clear();
        if (logger.isDebugEnabled()) {
            logger.debug("t: {}, learn predictor param: {}",
                    this.t, formulaStringParam(this.modelRef.get().weightsCopy()));
        }
        persistIfDue(saveDue);
        return LearningResult.MODEL_UPDATED;
    }

    private void persistIfDue(boolean saveDue) {
        if (!saveDue) {
            return;
        }
        this.persistence.save(this.modelRef.get().weightsCopy(), this.t - 1L);
    }

    /**
     * Final state flush at endpoint retirement: persists the current model
     * weights together with the rolling history, fixing up a completed
     * cold-start refit and any trailing samples that never crossed the
     * throttled save interval. Idempotent and failure-tolerant; a
     * persistence-less predictor is a no-op.
     */
    public synchronized void flushState() {
        if (this.persistence == null) {
            return;
        }
        this.persistence.save(this.modelRef.get().weightsCopy(), this.t - 1L);
    }

    /**
     * One batched Adam update over the given samples, publishing the new
     * immutable evaluator atomically.
     */
    private void updateWeights(List<BatchUpdateItem> items) {
        ModelEvaluator oldModel = this.modelRef.get();
        double[] oldWeights = oldModel.weightsCopy();
        double[] gradient = new double[oldWeights.length];
        for (BatchUpdateItem batchItem : items) {
            double[] thisGradient = new double[oldWeights.length];
            double[] inputs = collectInput(batchItem.features());
            double linear = calcLinear(inputs, oldWeights);
            double[] nonLinearOutput = new double[5];
            calcNonLinear(oldWeights, linear, nonLinearOutput);
            double predict = nonLinearOutput[0];
            double nonLinearGrad = nonLinearOutput[1];
            double nonLinearP6Grad = nonLinearOutput[2];
            double nonLinearP7Grad = nonLinearOutput[3];
            double nonLinearP8Grad = nonLinearOutput[4];
            thisGradient[LINEAR_PARAM_COUNT] = nonLinearP6Grad;
            thisGradient[LINEAR_PARAM_COUNT + 1] = nonLinearP7Grad;
            thisGradient[LINEAR_PARAM_COUNT + 2] = nonLinearP8Grad;
            double linearGrad = nonLinearGrad / COFF3;
            for (int i = 0; i < inputs.length; i++) {
                thisGradient[i] = linearGrad * inputs[i];
            }
            double diff = predict - batchItem.actualMs();
            for (int i = 0; i < oldWeights.length; i++) {
                gradient[i] += diff * thisGradient[i];
            }
        }
        int samples = items.size();
        for (int i = 0; i < oldWeights.length; i++) {
            gradient[i] = gradient[i] / samples;
        }
        for (int i = 0; i < oldWeights.length; i++) {
            this.adamMoment1[i] = this.adamMoment1[i] * this.beta1 + (1 - this.beta1) * gradient[i];
            this.adamMoment2[i] = this.adamMoment2[i] * this.beta2 + (1 - this.beta2) * gradient[i] * gradient[i];
        }
        double[] newWeights = oldWeights.clone();
        for (int i = 0; i < newWeights.length; i++) {
            newWeights[i] -= this.alpha * Math.sqrt(1.0 - Math.pow(this.beta2, this.t))
                    / (1.0 - Math.pow(this.beta1, this.t))
                    * this.adamMoment1[i] / (Math.sqrt(this.adamMoment2[i] + this.epsilon));
        }

        this.modelRef.set(new ModelEvaluator(newWeights));
        this.t = this.t + 1;
    }

    private static String formulaStringParam(double[] weights) {
        return Arrays.stream(weights)
                .mapToObj(String::valueOf)
                .collect(Collectors.joining(", "));
    }

}
