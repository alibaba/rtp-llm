package org.flexlb.balance.prediction;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Evaluates one immutable parsed Prefill formula for single requests or batches, including per-item
 * sums and batch-wide features. Shared expression identity permits prediction reuse; learning
 * records eligible completion samples without replacing this evaluator.
 */
public class FormulaPredictor implements PrefillTimePredictor, PrefillTimePredictor.Evaluator {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");
    private final String formulaIdentity;
    private final PrefillTimeFormula formula;

    /**
     * Create a predictor with the given formula string.
     *
     * @param formulaString the cost formula expression
     */
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
        PrefillTimeVariableBindings.EvaluationVariables vars =
                PrefillTimeVariableBindings.singleRequestVariables(totalTokens, hitTokens);
        return formula.evaluate(vars.topLevelVars(), vars.itemVars());
    }

    @Override
    public double predictBatchMs(PrefillBatchFeatures features) {
        if (features.items().isEmpty()) {
            return 0.0;
        }
        PrefillTimeVariableBindings.EvaluationVariables vars =
                PrefillTimeVariableBindings.batchVariables(features);
        return formula.evaluateAsDouble(vars.topLevelVars(), vars.itemVars());
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
}
