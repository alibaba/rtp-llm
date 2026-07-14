package org.flexlb.balance.strategy;

import lombok.Data;
import lombok.NoArgsConstructor;
import org.flexlb.balance.scheduler.BatchItem;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Arrays;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.Collectors;

/**
 * Prefill-time predictor with hardcoded linear regression and online learning.
 *
 * <p>
 * Formula: {@code y = w0 + w1 * sum(computeTokens) + w2 * sum(hitCacheTokens)}
 * where {@code computeTokens = inputTokens - hitCacheTokens} (floor at 0).
 *
 *
 *
 *
 *
 *
 *
 *
 *
 * <p>
 * The three weights ({@code w0}, {@code w1}, {@code w2}) are volatile doubles
 * that can be updated at runtime via {@link #setParameter(String, double)}.
 * The
 * {@link #learn(List, long, long)} callback is a stub — the actual online
 * learning algorithm will be implemented in a future iteration.
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
    private final int param_count;
    private final double[] adamMoment1;
    private final double[] adamMoment2;
    private final double beta1 = 0.9;
    private final double beta2 = 0.95;
    private final double epsilon = 1e-20;
    private final double alpha = 0.005;
    private long t = 1;
    private int batchSize = 4;
    private List<BatchUpdateItem> itemBatch;

    public LearningPredictor() {
        this.weightsRef = new AtomicReference<>(new double[] { 300, -5,
                0, 5, 0, 0 });
        this.param_count = this.weightsRef.get().length;
        this.adamMoment1 = new double[this.param_count];
        this.adamMoment2 = new double[this.param_count];
        this.itemBatch = new ArrayList<>();
        logger.warn("learn predictor created, t: {}, init param: {}, beta1: {}, beta2: {}, alpha: {}, batchSize: {}",
                this.t, formulaStringParam(this.weightsRef.get()), this.beta1, this.beta2, this.alpha, this.batchSize);
    }

    @Override
    public long estimateMs(long totalTokens, long hitTokens) {
        return 0;
    }

    @Override
    public long predictBatchMs(List<BatchItem> items) {
        logger.info("t: {}, learn predictor predictBatchMs: {}, items count: {}",
                this.t, formulaStringParam(this.weightsRef.get()), items.size());
        if (items.isEmpty()) {
            return 0;
        }
        double[] inputs = this.collectInput(items);
        double[] weights = this.weightsRef.get();
        return (long) calcOutput(inputs, weights);
    }

    private double calcOutput(double[] inputs, double[] weights) {
        double sum = 0.0;
        for (int i = 0; i < weights.length; i++) {
            sum += inputs[i] * weights[i];
        }
        return sum;
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
        double[] inputs = new double[this.param_count];
        inputs[0] = 1.0;
        inputs[1] = (double) items.size();
        inputs[2] = reuse;
        inputs[3] = compute;
        inputs[4] = compute_square;
        inputs[5] = reuse_mul_compute;
        return inputs;
    }

    @Override
    public synchronized void learn(List<BatchItem> items, long predictedMs, long actualMs) {
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
            double[] newWeights = oldWeights.clone();
            double[] gradient = new double[this.param_count];
            for (BatchUpdateItem batchItem : this.itemBatch) {
                double[] inputs = this.collectInput(batchItem.getItem());
                double predict = calcOutput(inputs, newWeights);
                double diff = predict - batchItem.getActualMs();
                for (int i = 0; i < newWeights.length; i++) {
                    gradient[i] += diff * inputs[i];
                }
            }
            for (int i = 0; i < newWeights.length; i++) {
                gradient[i] = gradient[i] / this.batchSize;
            }
            for (int i = 0; i < newWeights.length; i++) {
                this.adamMoment1[i] = this.adamMoment1[i] * this.beta1 + (1 - this.beta1) * gradient[i];
                this.adamMoment2[i] = this.adamMoment2[i] * this.beta2 + (1 - this.beta2) * gradient[i] * gradient[i];
            }
            for (int i = 0; i < newWeights.length; i++) {
                newWeights[i] -= this.alpha * Math.sqrt(1.0 - Math.pow(this.beta2, this.t))
                        / (1.0 - Math.pow(this.beta1, this.t))
                        * this.adamMoment1[i] / (Math.sqrt(this.adamMoment2[i] + this.epsilon));
            }
            // System.out.println("gradient: " + formulaStringParam(gradient));
            // System.out.println("old: " + formulaStringParam(oldWeights));
            // System.out.println("new: " + formulaStringParam(newWeights));
            // System.out.println("moment1: " + formulaStringParam(this.adamMoment1));
            // System.out.println("moment2: " + formulaStringParam(this.adamMoment2));
            return newWeights;
        });
        this.t = this.t + 1;
        this.itemBatch.clear();
        if (this.t % 10 == 0) {
            logger.info("t: {}, learn predictor param: {}", this.t, formulaStringParam(this.weightsRef.get()));
        }
    }

    // ---- parameter management ----

    public double getParameter(String name) {
        return 0;
    }

    public void setParameter(String name, double value) {
    }

    public Set<String> parameterNames() {
        return Set.of("w0", "w1", "w2");
    }

    public Map<String, Double> getParameters() {
        return Map.of("w0", 0.0, "w1", 0.0, "w2", 0.0);
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
        return formulaStringParam(weights);
    }
}
