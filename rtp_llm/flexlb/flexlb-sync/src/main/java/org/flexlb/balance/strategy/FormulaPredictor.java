package org.flexlb.balance.strategy;

import org.flexlb.balance.scheduler.BatchItem;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Prefill-time predictor driven by a user-configurable formula.
 *
 * <p>Two evaluation modes share the same formula string:
 * <ul>
 *   <li>{@link #estimateMs(long, long)} — single request:
 *       fills per-request variables and sets {@code batchSize=1}</li>
 *   <li>{@link #predictBatchMs(List)} — batch: aggregates token statistics,
 *       then evaluates {@code sum(expr)} over the batch items</li>
 * </ul>
 *
 * <p>Construction is cheap — the formula is parsed once and the AST is shared
 * across all evaluations.
 *
 * <p>An optional {@link #learn(List, long, long)} callback is invoked on each batch
 * completion to feed back the actual-vs-predicted timing. The current
 * implementation is a stub — the learning logic is to be added.
 */
public class FormulaPredictor implements PrefillTimePredictor {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");

    private final PrefillTimeFormula formula;

    /**
     * Create a predictor with the given formula string.
     *
     * @param formulaString the cost formula expression
     */
    public FormulaPredictor(String formulaString) {
        this.formula = PrefillTimeFormula.parse(formulaString);
    }

    @Override
    public long estimateMs(long totalTokens, long hitTokens) {
        PrefillTimeVariableBindings.EvaluationVariables vars =
                PrefillTimeVariableBindings.singleRequestVariables(totalTokens, hitTokens);
        return formula.evaluate(vars.topLevelVars(), vars.itemVars());
    }

    @Override
    public long predictBatchMs(List<BatchItem> items) {
        if (items.isEmpty()) {
            return 0;
        }
        PrefillTimeVariableBindings.EvaluationVariables vars =
                PrefillTimeVariableBindings.batchVariables(items);
        return formula.evaluate(vars.topLevelVars(), vars.itemVars());
    }

    @Override
    public void learn(List<BatchItem> items, long predictedMs, long actualMs) {
        logger.debug("learn sample: batchSize={} predictedMs={} actualMs={}",
                items != null ? items.size() : 0, predictedMs, actualMs);
    }

    // ---- parameter management (delegates to formula) ----

    public double getParameter(String name) {
        return formula.getParameter(name);
    }

    public void setParameter(String name, double value) {
        formula.setParameter(name, value);
    }

    public Set<String> parameterNames() {
        return formula.parameterNames();
    }

    public Map<String, Double> getParameters() {
        return formula.getParameters();
    }

    public boolean hasParameters() {
        return formula.hasParameters();
    }

    /** The parsed formula, for inspection. */
    public String formulaString() {
        return formula.toString();
    }
}
