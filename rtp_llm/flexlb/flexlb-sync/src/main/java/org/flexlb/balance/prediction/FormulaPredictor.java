package org.flexlb.balance.prediction;

import org.flexlb.balance.prediction.PrefillTimePredictor.PrefillBatchFeatures;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.List;

import static com.google.common.base.Preconditions.checkArgument;

/** Immutable formula evaluator shared by single-request and batch predictions.
 * Endpoints with equal configured expressions share a model identity; learning only records samples. */
public class FormulaPredictor implements PrefillTimePredictor, PrefillTimePredictor.Evaluator {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");
    private final String formulaIdentity;
    private final PrefillTimeFormula formula;

    public FormulaPredictor(String formulaString) {
        this.formulaIdentity = java.util.Objects.requireNonNull(formulaString, "formulaString");
        this.formula = PrefillTimeFormula.parse(formulaString);
        logger.trace("formula predictor created");
    }

    @Override
    public Evaluator evaluator() {
        return this;
    }

    @Override
    public Object snapshotIdentity() {
        return formulaIdentity;
    }

    @Override
    public long estimateMs(long totalTokens, long hitTokens) {
        PrefillTimeVariableBindings.BindingContext vars =
                PrefillTimeVariableBindings.singleRequestVariables(totalTokens, hitTokens);
        return formula.evaluate(vars.topLevelVars, vars.itemVars);
    }

    @Override
    public double predictBatchMs(PrefillBatchFeatures features) {
        if (features.items().isEmpty()) {
            return 0.0;
        }
        double[] vars =
                PrefillTimeVariableBindings.batchVariables(
                        features, formula.requiresBatchStatistics());
        return formula.evaluateBatch(vars, features.items());
    }

    @Override
    public BatchPrediction newBatchPrediction() {
        ArithmeticFormula.Aggregation aggregation = formula.newAggregation();
        if (aggregation == null) return PrefillTimePredictor.Evaluator.super.newBatchPrediction();
        var bindings = new PrefillTimeVariableBindings.AppendBindings();
        return (seqLen, hitCache) -> {
            bindings.append(seqLen, hitCache);
            aggregation.append(bindings.item);
            return aggregation.evaluate(bindings.batch);
        };
    }

    @Override
    public LearningResult learn(PrefillBatchFeatures features, long predictedMs, long actualMs) {
        logger.debug(
                "learn sample: batchSize={} predictedMs={} actualMs={}",
                features != null ? features.batchSize() : 0,
                predictedMs,
                actualMs);
        return LearningResult.MODEL_UNCHANGED;
    }

    /** Thread-local scalar bindings; immutable items supply aggregate bindings directly. */
    static final class PrefillTimeVariableBindings {

        private static final ThreadLocal<BindingContext> BINDING_CTX =
                ThreadLocal.withInitial(BindingContext::new);

        private PrefillTimeVariableBindings() {}

        static BindingContext singleRequestVariables(long totalTokens, long hitCacheTokens) {
            BindingContext ctx = BINDING_CTX.get();
            ctx.reset();
            totalTokens = Math.max(0L, totalTokens);
            hitCacheTokens = Math.clamp(hitCacheTokens, 0L, totalTokens);
            fillRequestVars(ctx.topLevelVars, totalTokens, hitCacheTokens);
            ctx.topLevelVars[PrefillTimeFormula.IDX_BATCH_SIZE] = 1.0;
            long inputTokens = (long) ctx.topLevelVars[PrefillTimeFormula.IDX_INPUT_TOKENS];
            long boundedHitCacheTokens =
                    (long) ctx.topLevelVars[PrefillTimeFormula.IDX_HIT_CACHE_TOKENS];
            fillBatchVars(
                    ctx.topLevelVars,
                    inputTokens,
                    boundedHitCacheTokens,
                    inputTokens,
                    inputTokens - boundedHitCacheTokens);

            return ctx;
        }

        static double[] batchVariables(PrefillBatchFeatures features, boolean requiresStatistics) {
            BindingContext ctx = BINDING_CTX.get();
            ctx.reset();
            ctx.topLevelVars[PrefillTimeFormula.IDX_BATCH_SIZE] = features.batchSize();
            if (requiresStatistics) {
                long totalInput = 0L, totalHit = 0L, maxInput = 0L, maxCompute = 0L;
                for (PrefillBatchFeatures.Item item : features.items()) {
                    // Preserve the scalar binding's long/double conversion at token boundaries.
                    long input = (long) (double) item.seqLen();
                    long hit = (long) (double) item.hitCache();
                    totalInput += input;
                    totalHit += hit;
                    maxInput = Math.max(maxInput, input);
                    maxCompute = Math.max(maxCompute, input - hit);
                }
                fillBatchVars(ctx.topLevelVars, totalInput, totalHit, maxInput, maxCompute);
            }
            return ctx.topLevelVars;
        }

        static final class AppendBindings {
            final double[] batch = new double[PrefillTimeFormula.VAR_COUNT];
            final double[] item = new double[PrefillTimeFormula.VAR_COUNT];
            private long totalInput, totalHit, maxInput, maxCompute;
            private int count;

            void append(long seqLen, long hitCache) {
                checkArgument(
                        seqLen >= 0 && hitCache >= 0 && hitCache <= seqLen,
                        "Invalid request token counts");
                fillRequestVars(item, seqLen, hitCache);
                long input = (long) item[PrefillTimeFormula.IDX_INPUT_TOKENS];
                long hit = (long) item[PrefillTimeFormula.IDX_HIT_CACHE_TOKENS];
                totalInput += input;
                totalHit += hit;
                maxInput = Math.max(maxInput, input);
                maxCompute = Math.max(maxCompute, input - hit);
                batch[PrefillTimeFormula.IDX_BATCH_SIZE] = ++count;
                fillBatchVars(batch, totalInput, totalHit, maxInput, maxCompute);
            }
        }

        private static void fillBatchVars(
                double[] vars,
                long totalInputTokens,
                long totalHitCacheTokens,
                long maxInputTokens,
                long maxComputeTokens) {
            vars[PrefillTimeFormula.IDX_TOTAL_INPUT_TOKENS] = totalInputTokens;
            vars[PrefillTimeFormula.IDX_TOTAL_HIT_CACHE_TOKENS] = totalHitCacheTokens;
            vars[PrefillTimeFormula.IDX_TOTAL_COMPUTE_TOKENS] =
                    totalInputTokens - totalHitCacheTokens;
            vars[PrefillTimeFormula.IDX_MAX_INPUT_TOKENS] = maxInputTokens;
            vars[PrefillTimeFormula.IDX_MAX_COMPUTE_TOKENS] = maxComputeTokens;
        }

        /**
         * Fill the four per-request variables from normalized or validated token counts.
         * Other slots in pooled item arrays stay zero; formulas only read these bindings.
         */
        private static void fillRequestVars(double[] vars, long totalTokens, long hitCacheTokens) {
            vars[PrefillTimeFormula.IDX_INPUT_TOKENS] = totalTokens;
            vars[PrefillTimeFormula.IDX_HIT_CACHE_TOKENS] = hitCacheTokens;
            vars[PrefillTimeFormula.IDX_COMPUTE_TOKENS] = totalTokens - hitCacheTokens;
            vars[PrefillTimeFormula.IDX_HAS_HIT_CACHE] = hitCacheTokens > 0 ? 1.0 : 0.0;
        }

        static final class BindingContext {
            final double[] topLevelVars = new double[PrefillTimeFormula.VAR_COUNT];
            final List<double[]> itemVars = List.of(topLevelVars);

            void reset() {
                Arrays.fill(topLevelVars, 0.0);
            }
        }
    }
}
