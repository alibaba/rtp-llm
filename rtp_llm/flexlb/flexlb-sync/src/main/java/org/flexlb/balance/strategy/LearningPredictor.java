package org.flexlb.balance.strategy;

import lombok.Data;
import lombok.NoArgsConstructor;
import org.flexlb.balance.scheduler.BatchItem;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;
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
 * The six weights ({@code w0}–{@code w5}) are stored in an {@link AtomicReference}
 * and updated at runtime via {@link #setParameter(String, double)}.
 * The {@link #learn(List, long, long)} callback uses an Adam optimizer
 * to perform online gradient descent on completed batches.
 */
public class LearningPredictor implements PrefillTimePredictor {
    @Data
    @NoArgsConstructor
    private class BatchUpdateItem {
        private List<BatchItem> item;
        private long predictedMs;
        private long actualMs;
    };

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");

    private final AtomicReference<double[]> weightsRef;
    private final int linear_param_count;
    private final int total_param_count;
    private final double[] adamMoment1;
    private final double[] adamMoment2;
    private final double beta1 = 0.9;
    private final double beta2 = 0.95;
    private final double epsilon = 1e-20;
    private final double stable_alpha2 = 1;
    private double alpha = 0.022;
    private double coff1 = 0.005;
    private double coff2 = 0.02;
    private double coff3 = 320;
    private long t = 1;
    private int batchSize = 4;
    private List<BatchUpdateItem> itemBatch;

    public LearningPredictor() {
        this.weightsRef = new AtomicReference<>(new double[] { -4.40538432604287, 10.522208701202377, 1.5043093890711503,
                                                               21.40103419118763, 0.11145680735428248, 0.08305932028650383,
                                                               1.451617309598213, 1.0268830123611967, -4.405384326042869});
        this.linear_param_count = 6;
        this.total_param_count = this.weightsRef.get().length;
        this.adamMoment1 = new double[this.total_param_count];
        this.adamMoment2 = new double[this.total_param_count];
        this.itemBatch = new ArrayList<>();
        logger.warn(
                "learn predictor created, t: {}, total param {}, init param: {}, beta1: {}, beta2: {}, alpha: {}, batchSize: {}",
                this.t, this.total_param_count, formulaStringParam(this.weightsRef.get()), this.beta1, this.beta2, this.alpha, this.batchSize);
    }

    @Override
    public long estimateMs(long totalTokens, long hitTokens) {
        long seq = Math.max(0L, totalTokens);
        long hit = Math.max(0L, Math.min(hitTokens, seq));
        double thisReuse = hit / 1024.0;
        double thisCompute = (seq - hit) / 1024.0;
        double[] inputs = new double[this.total_param_count];
        inputs[0] = 1.0;
        inputs[1] = 1.0;
        inputs[2] = thisReuse;
        inputs[3] = thisCompute;
        inputs[4] = thisCompute * thisCompute;
        inputs[5] = thisReuse * thisCompute;
        double[] weights = this.weightsRef.get();
        double linear = calcLinear(inputs, weights);
        double[] values = new double[5];
        calcNonLinear(weights, linear, values);
        return (long) values[0];
    }

    @Override
    public double predictBatchMs(List<BatchItem> items) {
        logger.info("t: {}, learn predictor predictBatchMs: {}, items count: {}",
                this.t, formulaStringParam(this.weightsRef.get()), items.size());
        if (items.isEmpty()) {
            return 0;
        }
        double[] inputs = this.collectInput(items);
        double[] weights = this.weightsRef.get();
        double linear = calcLinear(inputs, weights);
        double[] values = new double[5];
        calcNonLinear(weights, linear, values);
        return values[0];
    }

    @Override
    public double predictBatchMsUncached(List<BatchItem> items) {
        // LearningPredictor has no cache — delegate directly.
        return predictBatchMs(items);
    }

    @Override
    public double predictBatchMs(List<BatchItem> existingItems, long newSeqLen, long newCacheHit) {
        if (existingItems.isEmpty()) {
            return estimateMs(newSeqLen, newCacheHit);
        }
        double[] inputs = this.collectInputWithExtra(existingItems, newSeqLen, newCacheHit);
        double[] weights = this.weightsRef.get();
        double linear = calcLinear(inputs, weights);
        double[] values = new double[5];
        calcNonLinear(weights, linear, values);
        return values[0];
    }

    private double calcLinear(double[] inputs, double[] weights) {
        double sum = 0.0;
        for (int i = 0; i < inputs.length; i++) {
            sum += inputs[i] * weights[i];
        }
        return sum / this.coff3;
    }

    public void calcNonLinear(double[] weights, double linearOutput, double[] output) {
        calcNonLinearFast(weights, linearOutput, output);
    }

    public void calcNonLinearFast(double[] weights, double linearOutput, double[] output) {
        // param6 / coff1 + param7 / coff2 * ((linear + 1 + p8) + Sqrt((linear + 1 + p8)^2 + 4))
        double p6 = weights[this.linear_param_count] / this.coff1;
        double p7 = weights[this.linear_param_count + 1] / this.coff2;
        double p8 = weights[this.linear_param_count + 2] + 1.0;
        double linearAddP8 = linearOutput + p8;
        double sqrt_value = Math.sqrt(linearAddP8 * linearAddP8 + 4.0);
        double non_linear_value = linearAddP8 + sqrt_value;
        double predict = p6 + p7 * non_linear_value;
        double grad = p7 * (1.0 + linearAddP8 / sqrt_value);
        double p6_grad = 1.0 / this.coff1;
        double p7_grad = non_linear_value / this.coff2;
        double p8_grad = grad;
        output[0] = predict;
        output[1] = grad;
        output[2] = p6_grad;
        output[3] = p7_grad;
        output[4] = p8_grad;
    }

    public void calcNonLinearExp(double[] weights, double linearOutput, double[] output) {
        // param6 / coff1 + param7 / coff2 * Log(1 + (1 + param8) * Exp(linear))
        double p6 = weights[this.linear_param_count] / this.coff1;
        double p7 = weights[this.linear_param_count + 1] / this.coff2;
        double p8 = weights[this.linear_param_count + 2] + 1;
        double exp_value = Math.exp(linearOutput);
        double non_linear_value = Math.log(1.0 + p8 * exp_value);
        double predict = p6 + p7 * non_linear_value;
        double grad = p7 * p8 / (p8 + Math.exp(-linearOutput));
        double p6_grad = 1.0 / this.coff1;
        double p7_grad = non_linear_value / this.coff2;
        double p8_grad = p7 * exp_value / (1.0 + p8 * exp_value);
        output[0] = predict;
        output[1] = grad;
        output[2] = p6_grad;
        output[3] = p7_grad;
        output[4] = p8_grad;
    }

    private double[] collectInput(List<BatchItem> items) {
        double reuse = 0.0;
        double compute = 0.0;
        double compute_square = 0.0;
        double reuse_mul_compute = 0.0;
        for (BatchItem item : items) {
            long seq = Math.max(0L, item.seqLen());
            long hit = Math.max(0L, Math.min(item.hitCache(), seq));
            double thisReuse = hit / 1024.0;
            double thisCompute = (seq - hit) / 1024.0;
            reuse += thisReuse;
            compute += thisCompute;
            compute_square += thisCompute * thisCompute;
            reuse_mul_compute += thisReuse * thisCompute;
        }
        double[] inputs = new double[this.linear_param_count];
        inputs[0] = 1.0;
        inputs[1] = (double) items.size();
        inputs[2] = reuse;
        inputs[3] = compute;
        inputs[4] = compute_square;
        inputs[5] = reuse_mul_compute;
        return inputs;
    }

    /**
     * Like {@link #collectInput(List)} but appends a virtual item for a new
     * request that hasn't been enqueued yet. The virtual item's {@code seqLen}
     * and {@code hitCache} participate in all aggregations, and
     * {@code batchSize = items.size() + 1}.
     */
    private double[] collectInputWithExtra(List<BatchItem> items, long extraSeqLen, long extraCacheHit) {
        double reuse = 0.0;
        double compute = 0.0;
        double compute_square = 0.0;
        double reuse_mul_compute = 0.0;
        for (BatchItem item : items) {
            long seq = Math.max(0L, item.seqLen());
            long hit = Math.max(0L, Math.min(item.hitCache(), seq));
            double thisReuse = hit / 1024.0;
            double thisCompute = (seq - hit) / 1024.0;
            reuse += thisReuse;
            compute += thisCompute;
            compute_square += thisCompute * thisCompute;
            reuse_mul_compute += thisReuse * thisCompute;
        }
        // Virtual item for the new request
        long extraSeq = Math.max(0L, extraSeqLen);
        long extraHit = Math.max(0L, Math.min(extraCacheHit, extraSeq));
        double extraReuse = extraHit / 1024.0;
        double extraCompute = (extraSeq - extraHit) / 1024.0;
        reuse += extraReuse;
        compute += extraCompute;
        compute_square += extraCompute * extraCompute;
        reuse_mul_compute += extraReuse * extraCompute;

        double[] inputs = new double[this.linear_param_count];
        inputs[0] = 1.0;
        inputs[1] = (double) (items.size() + 1);
        inputs[2] = reuse;
        inputs[3] = compute;
        inputs[4] = compute_square;
        inputs[5] = reuse_mul_compute;
        return inputs;
    }

    @Override
    public void learn(List<BatchItem> items, long predictedMs, long actualMs) {
        // logger.debug("learn sample: batchSize={} predictedMs={} actualMs={}", items
        // != null ? items.size() : 0, predictedMs, actualMs);
        BatchUpdateItem item = new BatchUpdateItem();
        item.setItem(items);
        item.setPredictedMs(predictedMs);
        item.setActualMs(actualMs);
        this.itemBatch.add(item);
        if (this.itemBatch.size() < this.batchSize) {
            return;
        }
        this.weightsRef.updateAndGet(oldWeights -> {
            double[] gradient = new double[this.total_param_count];
            for (BatchUpdateItem batchItem : this.itemBatch) {
                double[] thisGradient = new double[this.total_param_count];
                double[] inputs = this.collectInput(batchItem.getItem());
                double linear = calcLinear(inputs, oldWeights);
                double[] nonLinearOutput = new double[5];
                calcNonLinear(oldWeights, linear, nonLinearOutput);
                double predict = nonLinearOutput[0];
                double nonLinearGrad = nonLinearOutput[1];
                double nonLinearP6Grad = nonLinearOutput[2];
                double nonLinearP7Grad = nonLinearOutput[3];
                double nonLinearP8Grad = nonLinearOutput[4];
                thisGradient[this.linear_param_count] = nonLinearP6Grad;
                thisGradient[this.linear_param_count + 1] = nonLinearP7Grad;
                thisGradient[this.linear_param_count + 2] = nonLinearP8Grad;
                double linearGrad = nonLinearGrad / this.coff3;
                for (int i = 0; i < inputs.length; i++) {
                    thisGradient[i] = linearGrad * inputs[i];
                }
                // check grad
                boolean checkGrad = false;
                if (checkGrad) {
                    for (int j = 0; j < oldWeights.length; j++) {
                        double[] test_weights = oldWeights.clone();
                        double test_delta = test_weights[j] != 0 ? Math.max(-10.0, Math.min(10.0, test_weights[j] * 0.0001))
                            : 1e-6;
                        test_weights[j] = test_weights[j] + test_delta;
                        double testLinear = calcLinear(inputs, test_weights);
                        double[] testNonLinearOutput = new double[5];
                        calcNonLinear(test_weights, testLinear, testNonLinearOutput);
                        double evaluateDiff = testNonLinearOutput[0] - predict;
                        double gradDiff = test_delta * thisGradient[j];
                        if (Math.abs(gradDiff - evaluateDiff) > Math.abs(evaluateDiff * 0.01)) {
                            System.out.println("grad check failed for param: " + j
                                               + ", gradDiff: " + gradDiff + " evaluateDiff: " + evaluateDiff
                                               + "grads: " + formulaWeights(thisGradient));
                        }
                    }
                }
                double diff = predict - batchItem.getActualMs();
                for (int i = 0; i < oldWeights.length; i++) {
                    gradient[i] += diff * thisGradient[i];
                }
                // System.out.println("this gradient: " + formulaStringParam(thisGradient));
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

            // System.out.println("t: " + this.t);
            // System.out.println("gradient: " + formulaStringParam(gradient));
            // System.out.println("old: " + formulaStringParam(oldWeights));
            // System.out.println("new: " + formulaStringParam(newWeights));
            // System.out.println("moment1: " + formulaStringParam(this.adamMoment1));
            // System.out.println("moment2: " + formulaStringParam(this.adamMoment2));
            // System.out.println("");
            return newWeights;
        });
        this.t = this.t + 1;
        this.itemBatch.clear();
        logger.info("t: {}, learn predictor param: {}", this.t, formulaStringParam(this.weightsRef.get()));
    }

    // ---- parameter management ----

    public double[] getParams() {
        return weightsRef.get();
    }

    public double getParameter(String name) {
        int idx = weightIndex(name);
        return weightsRef.get()[idx];
    }

    public void setParameter(String name, double value) {
        int idx = weightIndex(name);
        weightsRef.updateAndGet(old -> {
            double[] updated = old.clone();
            updated[idx] = value;
            return updated;
        });
    }

    public void setCoff(double coff1, double coff2, double coff3, double alpha) {
        this.coff1 = coff1;
        this.coff2 = coff2;
        this.coff3 = coff3;
        this.alpha = alpha;
    }

    public Set<String> parameterNames() {
        return Set.of("w0", "w1", "w2", "w3", "w4", "w5");
    }

    public Map<String, Double> getParameters() {
        double[] weights = weightsRef.get();
        Map<String, Double> result = new LinkedHashMap<>();
        for (int i = 0; i < weights.length; i++) {
            result.put("w" + i, weights[i]);
        }
        return result;
    }

    private int weightIndex(String name) {
        for (int i = 0; i < this.total_param_count; i++) {
            if (("w" + i).equals(name)) {
                return i;
            }
        }
        throw new IllegalArgumentException("Unknown parameter: " + name);
    }

    public boolean hasParameters() {
        return true;
    }

    private String formulaStringParam(double[] weights) {
        String result = Arrays.stream(weights)
                .mapToObj(String::valueOf)
                .collect(Collectors.joining(", "));
        return result;
    }

    private String formulaWeights(double[] weights) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < weights.length; i++) {
            if (i > 0) sb.append(", ");
            sb.append("w").append(i).append("=").append(weights[i]);
        }
        return sb.toString();
    }

    public String formulaString() {
        return formulaWeights(this.weightsRef.get());
    }
}
