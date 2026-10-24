package org.flexlb.balance.prediction;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** Immutable formula evaluator shared by single-request and batch predictions.
 * Endpoints with equal configured expressions share a model identity; learning only records samples. */
public class FormulaPredictor
        implements PrefillTimePredictor, PrefillTimePredictor.Evaluator {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");
    private final String formulaIdentity;
    private final PrefillTimeFormula formula;

    public FormulaPredictor(String formulaString) {
        this.formulaIdentity = java.util.Objects.requireNonNull(
                formulaString, "formulaString");
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
                PrefillTimeVariableBindings.singleRequestVariables(
                        totalTokens, hitTokens);
        return formula.evaluate(vars.topLevelVars, vars.itemVars);
    }

    @Override
    public double predictBatchMs(PrefillBatchFeatures features) {
        if (features.items().isEmpty()) {
            return 0.0;
        }
        double[] vars = PrefillTimeVariableBindings.batchVariables(
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
    public LearningResult learn(
            PrefillBatchFeatures features, long predictedMs, long actualMs) {
        logger.debug("learn sample: batchSize={} predictedMs={} actualMs={}",
                features != null ? features.batchSize() : 0, predictedMs, actualMs);
        return LearningResult.MODEL_UNCHANGED;
    }
}
