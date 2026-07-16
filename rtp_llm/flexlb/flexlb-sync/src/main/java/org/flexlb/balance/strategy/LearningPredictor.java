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
    private final double alpha = 0.005;
    private double coff1 = 0.55;
    private double coff2 = 1.2;
    private double coff3 = 1700.0;
    private long t = 1;
    private int batchSize = 4;
    private List<BatchUpdateItem> itemBatch;

    public LearningPredictor() {
        this.weightsRef = new AtomicReference<>(new double[] { 0, 0, 0, 0, 0, 0, 0, 0 });
        this.linear_param_count = 6;
        this.total_param_count = this.weightsRef.get().length;
        this.adamMoment1 = new double[this.total_param_count];
        this.adamMoment2 = new double[this.total_param_count];
        this.itemBatch = new ArrayList<>();
        logger.warn("learn predictor created, t: {}, init param: {}, beta1: {}, beta2: {}, alpha: {}, batchSize: {}",
                this.t, formulaStringParam(this.weightsRef.get()), this.beta1, this.beta2, this.alpha, this.batchSize);
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
        double linearExp = calcLinearExp(inputs, weights);
        return (long) calcOutput(weights, linearExp);
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
        double linearExp = calcLinearExp(inputs, weights);
        return calcOutput(weights, linearExp);
    }

    private double calcLinearExp(double[] inputs, double[] weights) {
        double sum = 0.0;
        for (int i = 0; i < inputs.length; i++) {
            sum += inputs[i] * weights[i];
        }
        return Math.exp(sum / this.coff3);
    }

    private double calcOutput(double[] weights, double linearExp) {
        return weights[this.linear_param_count] / this.coff1 +
                weights[this.linear_param_count + 1] / this.coff2 * linearExp;
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
        double[] inputs = new double[this.total_param_count];
        inputs[0] = 1.0;
        inputs[1] = (double) items.size();
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
                double[] inputs = this.collectInput(batchItem.getItem());
                double linearExp = calcLinearExp(inputs, oldWeights);
                double predict = calcOutput(oldWeights, linearExp);
                double diff = predict - batchItem.getActualMs();
                gradient[this.linear_param_count] += diff / this.coff1;
                gradient[this.linear_param_count + 1] += diff / this.coff2 * linearExp;
                double linearGrad = diff * oldWeights[this.linear_param_count + 1] / this.coff2 * linearExp
                        / this.coff3;
                gradient[0] = linearGrad;
                for (int i = 1; i < oldWeights.length; i++) {
                    gradient[i] += linearGrad * inputs[i];
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
            /*
             * System.out.println("t: " + this.t);
             * System.out.println("gradient: " + formulaStringParam(gradient));
             * System.out.println("old: " + formulaStringParam(oldWeights));
             * System.out.println("new: " + formulaStringParam(newWeights));
             * System.out.println("moment1: " + formulaStringParam(this.adamMoment1));
             * System.out.println("moment2: " + formulaStringParam(this.adamMoment2));
             */
            return newWeights;
        });
        this.t = this.t + 1;
        this.itemBatch.clear();
        logger.info("t: {}, learn predictor param: {}", this.t, formulaStringParam(this.weightsRef.get()));
    }

    // ---- parameter management ----

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

    public void setCoff(double coff1, double coff2, double coff3) {
        this.coff1 = coff1;
        this.coff2 = coff2;
        this.coff3 = coff3;
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

    public String formulaString() {
        double[] weights = this.weightsRef.get();
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < weights.length; i++) {
            if (i > 0) sb.append(", ");
            sb.append("w").append(i).append("=").append(weights[i]);
        }
        return sb.toString();
    }
}
