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
public class PrefillTimePredictor {

    private static final Logger logger = LoggerFactory.getLogger("syncLogger");

    private final PrefillTimeFormula formula;

    /**
     * Create a predictor with the given formula string.
     *
     * @param formulaString the cost formula expression
     */
    public PrefillTimePredictor(String formulaString) {
        this.formula = PrefillTimeFormula.parse(formulaString);
    }

    /**
     * Estimate prefill time for a single request from raw token counts.
     *
     * @param totalTokens input length
     * @param hitTokens   cache-hit token count (0 ≤ hitTokens ≤ totalTokens)
     * @return predicted time in milliseconds
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
     * @return predicted time in milliseconds (0 for an empty batch)
     */
    public long predictBatchMs(List<BatchItem> items) {
        if (items.isEmpty()) {
            return 0;
        }
        PrefillTimeVariableBindings.EvaluationVariables vars =
                PrefillTimeVariableBindings.batchVariables(items);
        return formula.evaluate(vars.topLevelVars(), vars.itemVars());
    }

    /**
     * Learn from a completed batch's actual execution time.
     * The batch items, predicted time, and actual time are provided for future
     * learning algorithm implementation.
     *
     * @param items       the batch requests (contains seqLen, hitCache, etc.)
     * @param predictedMs the formula-predicted execution time for the batch
     * @param actualMs    the engine-reported actual execution time
     */
    public void learn(List<BatchItem> items, long predictedMs, long actualMs) {
        logger.debug("learn sample: batchSize={} predictedMs={} actualMs={}",
                items != null ? items.size() : 0, predictedMs, actualMs);
    }

    /** The parsed formula, for inspection. */
    public String formulaString() {
        return formula.toString();
    }
}
