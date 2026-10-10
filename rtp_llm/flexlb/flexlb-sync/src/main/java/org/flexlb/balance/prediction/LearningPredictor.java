package org.flexlb.balance.prediction;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.Collectors;

import static com.google.common.base.Preconditions.checkArgument;

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
 */
public class LearningPredictor implements PrefillTimePredictor {

    private static final double TOKENS_PER_FEATURE_UNIT = 1_024.0;
    private record BatchUpdateItem(PrefillBatchFeatures features, long actualMs) {
    }

    /**
     * Atomically published prediction model. The weights array is owned by the
     * snapshot and is never mutated after the snapshot is constructed.
     */
    private static final class ModelEvaluator implements Evaluator {
        private final double[] weights;

        private ModelEvaluator(double[] weights) {
            this.weights = weights.clone();
        }

        @Override
        public long estimateMs(long totalTokens, long hitTokens) {
            long seq = Math.max(0L, totalTokens);
            long hit = Math.clamp(hitTokens, 0L, seq);
            double[] inputs = new double[LINEAR_PARAM_COUNT];
            appendInput(inputs, seq, hit);
            return (long) predict(inputs);
        }

        @Override
        public BatchPrediction newBatchPrediction() {
            double[] inputs = new double[LINEAR_PARAM_COUNT];
            return (seqLen, hitCache) -> {
                checkArgument(seqLen >= 0 && hitCache >= 0 && hitCache <= seqLen, "Invalid request token counts");
                appendInput(inputs, seqLen, hitCache);
                return predict(inputs);
            };
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
            return predict(collectInput(features));
        }

        private double predict(double[] inputs) {
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

    public LearningPredictor() {
        ModelEvaluator initialModel = new ModelEvaluator(
                new double[] { -4.40538432604287, 10.522208701202377, 1.5043093890711503,
                               21.40103419118763, 0.11145680735428248, 0.08305932028650383,
                               1.451617309598213, 1.0268830123611967, -4.405384326042869 });
        this.modelRef = new AtomicReference<>(initialModel);
        this.adamMoment1 = new double[initialModel.parameterCount()];
        this.adamMoment2 = new double[initialModel.parameterCount()];
        this.itemBatch = new ArrayList<>();
        logger.debug(
                "learn predictor created, t: {}, total param {}, init param: {}, "
                        + "beta1: {}, beta2: {}, alpha: {}, batchSize: {}",
                this.t, initialModel.parameterCount(),
                formulaStringParam(initialModel.weightsCopy()),
                this.beta1, this.beta2, this.alpha, this.batchSize);
    }

    @Override
    public Evaluator evaluator() {
        return modelRef.get();
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
        double[] inputs = new double[LINEAR_PARAM_COUNT];
        inputs[0] = 1.0;
        for (PrefillBatchFeatures.Item item : features.items()) {
            appendInput(inputs, item.seqLen(), item.hitCache());
        }
        return inputs;
    }

    /** Append in request order so every prefix retains the full evaluation's rounding. */
    private static void appendInput(double[] inputs, long seqLen, long hitCache) {
        double reuse = hitCache / TOKENS_PER_FEATURE_UNIT;
        double compute = (seqLen - hitCache) / TOKENS_PER_FEATURE_UNIT;
        inputs[0] = 1.0;
        inputs[1]++;
        inputs[2] += reuse;
        inputs[3] += compute;
        inputs[4] += compute * compute;
        inputs[5] += reuse * compute;
    }

    @Override
    public synchronized LearningResult learn(
            PrefillBatchFeatures features, long predictedMs, long actualMs) {
        this.itemBatch.add(new BatchUpdateItem(features, actualMs));
        if (this.itemBatch.size() < this.batchSize) {
            return LearningResult.MODEL_UNCHANGED;
        }
        ModelEvaluator oldModel = this.modelRef.get();
        double[] oldWeights = oldModel.weightsCopy();
        double[] gradient = new double[oldWeights.length];
        for (BatchUpdateItem batchItem : this.itemBatch) {
            double[] inputs = collectInput(batchItem.features());
            double linear = calcLinear(inputs, oldWeights);
            double[] nonLinearOutput = new double[5];
            calcNonLinear(oldWeights, linear, nonLinearOutput);
            double diff = nonLinearOutput[0] - batchItem.actualMs();
            double linearGrad = nonLinearOutput[1] / COFF3;
            for (int i = 0; i < inputs.length; i++) {
                gradient[i] += diff * (linearGrad * inputs[i]);
            }
            for (int i = LINEAR_PARAM_COUNT; i < oldWeights.length; i++) {
                gradient[i] += diff * nonLinearOutput[i - LINEAR_PARAM_COUNT + 2];
            }
        }
        for (int i = 0; i < oldWeights.length; i++) {
            gradient[i] = gradient[i] / this.batchSize;
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
        this.itemBatch.clear();
        if (logger.isDebugEnabled()) {
            logger.debug("t: {}, learn predictor param: {}", this.t, formulaStringParam(newWeights));
        }
        return LearningResult.MODEL_UPDATED;
    }

    private static String formulaStringParam(double[] weights) {
        return Arrays.stream(weights)
                .mapToObj(String::valueOf)
                .collect(Collectors.joining(", "));
    }

}
