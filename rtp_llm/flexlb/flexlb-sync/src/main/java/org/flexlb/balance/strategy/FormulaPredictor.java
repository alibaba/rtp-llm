package org.flexlb.balance.strategy;

import org.flexlb.balance.scheduler.BatchItem;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collections;
import java.util.LinkedHashMap;
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
 *       exposes explicit {@code total*}/{@code max*} variables and evaluates
 *       {@code sum(expr)} over the batch items when per-request distribution is needed</li>
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
    private final String sourceFormula;

    /**
     * Monotonic version counter incremented on every {@link #setParameter}.
     * Folded into the cache key so that a stale write racing with a clear
     * can never be observed by a post-invalidation query (TOCTOU fix).
     */
    private volatile long cacheVersion = 0;

    /**
     * LRU cache for {@link #predictBatchMs} results, keyed by a hash of batch item token stats.
     * Invalidated on {@link #setParameter}.
     */
    private final Map<Long, Double> resultCache = Collections.synchronizedMap(
            new LinkedHashMap<>(64, 0.75f, true) {
                @Override
                protected boolean removeEldestEntry(Map.Entry<Long, Double> eldest) {
                    return size() > 256;
                }
            });

    /**
     * Create a predictor with the given formula string.
     *
     * @param formulaString the cost formula expression
     */
    public FormulaPredictor(String formulaString) {
        this.sourceFormula = formulaString;
        this.formula = PrefillTimeFormula.parse(formulaString);
        logger.info("formula predictor created, formula: {}", formulaString);
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
        long key = computeCacheKey(items);
        Double cached = resultCache.get(key);
        if (cached != null) {
            return cached;
        }
        double result = predictBatchMsUncached(items);
        resultCache.put(key, result);
        return result;
    }

    @Override
    public double predictBatchMsUncached(List<BatchItem> items) {
        if (items.isEmpty()) {
            return 0.0;
        }
        PrefillTimeVariableBindings.EvaluationVariables vars =
                PrefillTimeVariableBindings.batchVariables(items);
        return (double) formula.evaluate(vars.topLevelVars(), vars.itemVars());
    }

    /**
     * Compute a hash key from the batch items' token statistics.
     * Only {@code seqLen} and {@code hitCache} participate, because these are the
     * only inputs that affect the formula result (via {@code inputTokens},
     * {@code hitCacheTokens}, {@code computeTokens}, batch totals/maxima, and {@code batchSize}).
     */
    private long computeCacheKey(List<BatchItem> items) {
        long hash = cacheVersion;          // 纳入版本号
        hash = hash * 31 + items.size();
        for (BatchItem item : items) {
            hash = hash * 31 + item.seqLen();
            hash = hash * 31 + item.hitCache();
        }
        return hash;
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
        cacheVersion++;        // 使旧版本写入的条目无法被新版本查询命中
        resultCache.clear();
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

    String immutableFormulaKey() {
        return hasParameters() ? null : sourceFormula;
    }

    /** The parsed formula, for inspection. */
    public String formulaString() {
        return formula.toString();
    }
}
