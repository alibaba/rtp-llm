package org.flexlb.balance.strategy;

import org.flexlb.balance.scheduler.BatchItem;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

/**
 * Prefill-time predictor driven by a user-configurable formula.
 *
 * <p>Two evaluation modes share the same formula string:
 * <ul>
 *   <li>{@link #estimateMs(long, long)} — single request:
 *       fills per-request variables and sets {@code batchSize=1}</li>
 *   <li>{@link #predictBatchMs(List)} — batch: aggregates token statistics,
 *       exposes explicit {@code total*}/{@code max*} variables and evaluates
 *       {@code sum(expr)} over the batch items when per-request distribution is needed</li>
 * </ul>
 *
 * <p>Construction is cheap — the formula is parsed once and the AST is shared
 * across all evaluations.
 *
 * <p>{@link #learn(PrefillBatchFeatures, long, long)} observes each eligible
 * batch completion. This immutable implementation records the sample without
 * changing the model generation.
 */
public class FormulaPredictor implements PrefillTimePredictor {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");

    private final PrefillTimeFormula formula;
    private final String sourceFormula;

    /**
     * Create a predictor with the given formula string.
     *
     * @param formulaString the cost formula expression
     */
    public FormulaPredictor(String formulaString) {
        this.sourceFormula = formulaString;
        this.formula = PrefillTimeFormula.parse(formulaString);
        logger.debug("formula predictor created");
    }

    @Override
    public long estimateMs(long totalTokens, long hitTokens) {
        PrefillTimeVariableBindings.EvaluationVariables vars =
                PrefillTimeVariableBindings.singleRequestVariables(totalTokens, hitTokens);
        return formula.evaluate(vars.topLevelVars(), vars.itemVars());
    }

    @Override
    public double predictBatchMs(List<BatchItem> items) {
        if (items.isEmpty()) {
            return 0.0;
        }
        PrefillTimeVariableBindings.EvaluationVariables vars =
                PrefillTimeVariableBindings.batchVariables(items);
        return (double) formula.evaluate(vars.topLevelVars(), vars.itemVars());
    }

    @Override
    public LearningResult learn(
            PrefillBatchFeatures features, long predictedMs, long actualMs) {
        logger.debug("learn sample: batchSize={} predictedMs={} actualMs={}",
                features != null ? features.batchSize() : 0, predictedMs, actualMs);
        return LearningResult.MODEL_UNCHANGED;
    }

    String immutableFormulaKey() {
        return sourceFormula;
    }
}
