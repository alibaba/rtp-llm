package org.flexlb.balance.strategy;

import org.flexlb.balance.scheduler.BatchItem;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Set;

/**
 * Builds variable bindings for {@link PrefillTimeFormula} evaluation.
 *
 * <p>Uses a {@link ThreadLocal} pool of {@code double[]} arrays and a reusable
 * {@code ArrayList} to eliminate per-call allocations on the hot evaluation path.
 * The returned {@link EvaluationVariables} references arrays owned by the ThreadLocal;
 * callers must consume them before the next call on the same thread.
 */
final class PrefillTimeVariableBindings {

    private static final String BATCH_SIZE = "batchSize";
    private static final Set<String> BATCH_VAR_NAMES = Set.of(
            "totalInputTokens", "totalHitCacheTokens", "totalComputeTokens",
            "maxInputTokens", "maxComputeTokens");
    private static final Set<String> REQUEST_VAR_NAMES = Set.of(
            "inputTokens", "hitCacheTokens", "computeTokens", "hasHitCache");

    private static final ThreadLocal<BindingContext> BINDING_CTX = ThreadLocal.withInitial(BindingContext::new);

    private PrefillTimeVariableBindings() {
    }

    static boolean supports(String name) {
        return BATCH_SIZE.equals(name) || BATCH_VAR_NAMES.contains(name) || REQUEST_VAR_NAMES.contains(name);
    }

    static boolean isBatchScoped(String name) {
        return BATCH_SIZE.equals(name) || BATCH_VAR_NAMES.contains(name);
    }

    static EvaluationVariables singleRequestVariables(long totalTokens, long hitCacheTokens) {
        BindingContext ctx = BINDING_CTX.get();
        ctx.reset();
        fillRequestVars(ctx.topLevelVars, totalTokens, hitCacheTokens);
        ctx.topLevelVars[PrefillTimeFormula.IDX_BATCH_SIZE] = 1.0;
        long inputTokens = (long) ctx.topLevelVars[PrefillTimeFormula.IDX_INPUT_TOKENS];
        long boundedHitCacheTokens = (long) ctx.topLevelVars[PrefillTimeFormula.IDX_HIT_CACHE_TOKENS];
        fillBatchVars(ctx.topLevelVars, inputTokens, boundedHitCacheTokens,
                inputTokens, inputTokens - boundedHitCacheTokens);

        double[] item = ctx.acquireArray();
        fillRequestVars(item, totalTokens, hitCacheTokens);
        ctx.itemVars.add(item);

        return new EvaluationVariables(ctx.topLevelVars, ctx.itemVars);
    }

    static EvaluationVariables batchVariables(List<BatchItem> items) {
        BindingContext ctx = BINDING_CTX.get();
        ctx.reset();
        long totalInputTokens = 0L;
        long totalHitCacheTokens = 0L;
        long maxInputTokens = 0L;
        long maxComputeTokens = 0L;
        for (BatchItem item : items) {
            double[] itemArray = ctx.acquireArray();
            fillRequestVars(itemArray, item.seqLen(), item.hitCache());
            ctx.itemVars.add(itemArray);

            long inputTokens = (long) itemArray[PrefillTimeFormula.IDX_INPUT_TOKENS];
            long hitCacheTokens = (long) itemArray[PrefillTimeFormula.IDX_HIT_CACHE_TOKENS];
            long computeTokens = inputTokens - hitCacheTokens;
            totalInputTokens += inputTokens;
            totalHitCacheTokens += hitCacheTokens;
            maxInputTokens = Math.max(maxInputTokens, inputTokens);
            maxComputeTokens = Math.max(maxComputeTokens, computeTokens);
        }
        ctx.topLevelVars[PrefillTimeFormula.IDX_BATCH_SIZE] = items.size();
        fillBatchVars(ctx.topLevelVars, totalInputTokens, totalHitCacheTokens,
                maxInputTokens, maxComputeTokens);
        return new EvaluationVariables(ctx.topLevelVars, ctx.itemVars);
    }

    /**
     * Build variable bindings for a batch consisting of existing queue items
     * plus a new request that hasn't been enqueued yet.
     *
     * <p>Equivalent to {@link #batchVariables(List)} applied to {@code items}
     * with a synthetic item appended for {@code extraSeqLen}/{@code extraCacheHit},
     * but avoids allocating a {@link BatchItem} wrapper. The synthetic item
     * participates in all aggregations, and {@code batchSize = items.size() + 1}.
     */
    static EvaluationVariables batchVariables(List<BatchItem> items, long extraSeqLen, long extraCacheHit) {
        BindingContext ctx = BINDING_CTX.get();
        ctx.reset();
        long totalInputTokens = 0L;
        long totalHitCacheTokens = 0L;
        long maxInputTokens = 0L;
        long maxComputeTokens = 0L;
        for (BatchItem item : items) {
            double[] itemArray = ctx.acquireArray();
            fillRequestVars(itemArray, item.seqLen(), item.hitCache());
            ctx.itemVars.add(itemArray);

            long inputTokens = (long) itemArray[PrefillTimeFormula.IDX_INPUT_TOKENS];
            long hitCacheTokens = (long) itemArray[PrefillTimeFormula.IDX_HIT_CACHE_TOKENS];
            long computeTokens = inputTokens - hitCacheTokens;
            totalInputTokens += inputTokens;
            totalHitCacheTokens += hitCacheTokens;
            maxInputTokens = Math.max(maxInputTokens, inputTokens);
            maxComputeTokens = Math.max(maxComputeTokens, computeTokens);
        }
        // Append the virtual item for the new request
        double[] extraArray = ctx.acquireArray();
        fillRequestVars(extraArray, extraSeqLen, extraCacheHit);
        ctx.itemVars.add(extraArray);
        long extraInputTokens = (long) extraArray[PrefillTimeFormula.IDX_INPUT_TOKENS];
        long extraHitCacheTokens = (long) extraArray[PrefillTimeFormula.IDX_HIT_CACHE_TOKENS];
        long extraComputeTokens = extraInputTokens - extraHitCacheTokens;
        totalInputTokens += extraInputTokens;
        totalHitCacheTokens += extraHitCacheTokens;
        maxInputTokens = Math.max(maxInputTokens, extraInputTokens);
        maxComputeTokens = Math.max(maxComputeTokens, extraComputeTokens);

        ctx.topLevelVars[PrefillTimeFormula.IDX_BATCH_SIZE] = items.size() + 1;
        fillBatchVars(ctx.topLevelVars, totalInputTokens, totalHitCacheTokens,
                maxInputTokens, maxComputeTokens);
        return new EvaluationVariables(ctx.topLevelVars, ctx.itemVars);
    }

    private static void fillBatchVars(double[] vars,
                                      long totalInputTokens,
                                      long totalHitCacheTokens,
                                      long maxInputTokens,
                                      long maxComputeTokens) {
        vars[PrefillTimeFormula.IDX_TOTAL_INPUT_TOKENS] = totalInputTokens;
        vars[PrefillTimeFormula.IDX_TOTAL_HIT_CACHE_TOKENS] = totalHitCacheTokens;
        vars[PrefillTimeFormula.IDX_TOTAL_COMPUTE_TOKENS] = totalInputTokens - totalHitCacheTokens;
        vars[PrefillTimeFormula.IDX_MAX_INPUT_TOKENS] = maxInputTokens;
        vars[PrefillTimeFormula.IDX_MAX_COMPUTE_TOKENS] = maxComputeTokens;
    }

    /**
     * Fill the given array with the four per-request variables.
     * The array is expected to be already zeroed by {@link BindingContext#acquireArray}.
     */
    private static void fillRequestVars(double[] vars, long totalTokens, long hitCacheTokens) {
        long inputTokens = Math.max(0L, totalTokens);
        long boundedHitCacheTokens = Math.max(0L, Math.min(hitCacheTokens, inputTokens));
        long computeTokens = inputTokens - boundedHitCacheTokens;
        double hasHitCache = boundedHitCacheTokens > 0 ? 1.0 : 0.0;
        vars[PrefillTimeFormula.IDX_INPUT_TOKENS] = inputTokens;
        vars[PrefillTimeFormula.IDX_HIT_CACHE_TOKENS] = boundedHitCacheTokens;
        vars[PrefillTimeFormula.IDX_COMPUTE_TOKENS] = computeTokens;
        vars[PrefillTimeFormula.IDX_HAS_HIT_CACHE] = hasHitCache;
    }

    /**
     * Thread-local container for reusable {@code double[]} arrays.
     * <ul>
     *   <li>{@code topLevelVars} — dedicated array for top-level variables (batchSize etc.)</li>
     *   <li>{@code itemVars} — reusable ArrayList of per-request variable arrays</li>
     *   <li>{@code arrayPool} — backing pool of {@code double[]} instances, grown on demand</li>
     * </ul>
     * After {@link #reset()}, all arrays are zeroed and ready for reuse.
     * The pool grows to the maximum batch size seen and never shrinks.
     */
    private static final class BindingContext {
        final double[] topLevelVars = new double[PrefillTimeFormula.VAR_COUNT];
        final List<double[]> itemVars = new ArrayList<>();
        final List<double[]> arrayPool = new ArrayList<>();
        int poolIndex = 0;

        void reset() {
            Arrays.fill(topLevelVars, 0.0);
            itemVars.clear();
            poolIndex = 0;
        }

        double[] acquireArray() {
            if (poolIndex < arrayPool.size()) {
                double[] a = arrayPool.get(poolIndex++);
                Arrays.fill(a, 0.0);
                return a;
            }
            double[] a = new double[PrefillTimeFormula.VAR_COUNT];
            arrayPool.add(a);
            poolIndex++;
            return a;
        }
    }

    record EvaluationVariables(double[] topLevelVars,
                               List<double[]> itemVars) {
    }
}
