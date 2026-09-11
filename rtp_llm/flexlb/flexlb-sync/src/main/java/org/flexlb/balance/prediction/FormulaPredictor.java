package org.flexlb.balance.prediction;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Prefill-time predictor driven by a user-configurable formula.
 *
 * <p>Two evaluation modes on its immutable evaluator share the same formula string:
 * <ul>
 *   <li>{@link PrefillTimePredictor.Evaluator#estimateMs(long, long)} — single request:
 *       fills per-request variables and sets {@code batchSize=1}</li>
 *   <li>{@link PrefillTimePredictor.Evaluator#predictBatchMs(PrefillBatchFeatures)}
 *       — batch: aggregates token statistics,
 *       exposes explicit {@code total*}/{@code max*} variables and evaluates
 *       {@code sum(expr)} over the batch items when per-request distribution is needed</li>
 * </ul>
 *
 * <p>Each predictor owns one parsed immutable formula. Endpoints built from
 * the same immutable configuration share the expression object as their model
 * identity, allowing a routing invocation to reuse an equal prediction without
 * a process-wide formula cache.
 *
 * <p>{@link #learn(PrefillBatchFeatures, long, long)} observes each eligible
 * batch completion. This immutable implementation records the sample without
 * replacing its evaluator.
 */
public class FormulaPredictor
        implements PrefillTimePredictor, PrefillTimePredictor.Evaluator {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");
    private final String formulaIdentity;
    private final PrefillTimeFormula formula;

    /**
     * Create a predictor with the given formula string.
     *
     * @param formulaString the cost formula expression
     */
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
        PrefillTimeVariableBindings.EvaluationVariables vars =
                PrefillTimeVariableBindings.singleRequestVariables(
                        totalTokens, hitTokens);
        return formula.evaluate(vars.topLevelVars(), vars.itemVars());
    }

    @Override
    public double predictBatchMs(PrefillBatchFeatures features) {
        if (features.items().isEmpty()) {
            return 0.0;
        }
        PrefillTimeVariableBindings.EvaluationVariables vars =
                PrefillTimeVariableBindings.batchVariables(features);
        return (double) formula.evaluate(
                vars.topLevelVars(), vars.itemVars());
    }

    @Override
    public LearningResult learn(
            PrefillBatchFeatures features, long predictedMs, long actualMs) {
        logger.debug("learn sample: batchSize={} predictedMs={} actualMs={}",
                features != null ? features.batchSize() : 0, predictedMs, actualMs);
        return LearningResult.MODEL_UNCHANGED;
    }
}
