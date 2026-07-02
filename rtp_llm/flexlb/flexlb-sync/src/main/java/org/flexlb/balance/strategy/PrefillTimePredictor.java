package org.flexlb.balance.strategy;

import org.flexlb.balance.scheduler.BatchItem;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Prefill-time predictor driven by a user-configurable formula.
 *
 * <p>Two evaluation modes share the same formula string:
 * <ul>
 *   <li>{@link #estimateMs(long, long)} — single request:
 *       fills {@code c, p} as well as {@code sum_c, sum_c2, sum_cp, sum_p, n=1}</li>
 *   <li>{@link #predictBatchMs(List)} — batch: aggregates token statistics,
 *       then fills {@code sum_c, sum_c2, sum_cp, sum_p, n}</li>
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
        long c = Math.max(0, totalTokens - hitTokens);
        long p = hitTokens;
        Map<String, Double> vars = new HashMap<>();
        vars.put("c",      (double) c);
        vars.put("p",      (double) p);
        vars.put("sum_c",  (double) c);
        vars.put("sum_c2", (double) (c * c));
        vars.put("sum_cp", (double) (c * p));
        vars.put("sum_p",  (double) p);
        vars.put("n",      1.0);
        return formula.evaluate(vars);
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
        int n = items.size();
        long sumC = 0;
        long sumC2 = 0;
        long sumCp = 0;
        long sumP = 0;

        for (BatchItem item : items) {
            long c = item.computeTokens();
            long p = item.hitCache();
            sumC  += c;
            sumC2 += c * c;
            sumCp += c * p;
            sumP  += p;
        }

        Map<String, Double> vars = new HashMap<>();
        vars.put("sum_c",  (double) sumC);
        vars.put("sum_c2", (double) sumC2);
        vars.put("sum_cp", (double) sumCp);
        vars.put("sum_p",  (double) sumP);
        vars.put("n",      (double) n);
        return formula.evaluate(vars);
    }

    /** The parsed formula, for inspection. */
    public String formulaString() {
        return formula.toString();
    }
}
