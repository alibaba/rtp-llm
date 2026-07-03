package org.flexlb.balance.strategy;

import org.flexlb.balance.scheduler.BatchItem;

import java.util.List;

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
 */
public class PrefillTimePredictor {

    private final PrefillTimeFormula formula;

    public PrefillTimePredictor(String formulaString) {
        this.formula = PrefillTimeFormula.parse(formulaString);
    }

    /**
     * Estimate prefill time for a single request from raw token counts.
     *
     * @param totalTokens input length
     * @param hitTokens   cache-hit token count (0 ≤ hitTokens ≤ totalTokens)
     */
    public long estimateMs(long totalTokens, long hitTokens) {
        PrefillTimeVariableBindings.EvaluationVariables vars =
                PrefillTimeVariableBindings.singleRequestVariables(totalTokens, hitTokens);
        return formula.evaluate(vars.topLevelVars(), vars.itemVars());
    }

    /**
     * Estimate prefill time for a batch of requests.
     *
     * @param items batch items (may be empty)
     * @return predicted time in milliseconds, or 0 for an empty batch
     */
    public long predictBatchMs(List<BatchItem> items) {
        if (items.isEmpty()) {
            return 0;
        }
        PrefillTimeVariableBindings.EvaluationVariables vars =
                PrefillTimeVariableBindings.batchVariables(items);
        return formula.evaluate(vars.topLevelVars(), vars.itemVars());
    }

    /** The parsed formula, for inspection. */
    public String formulaString() {
        return formula.toString();
    }
}
