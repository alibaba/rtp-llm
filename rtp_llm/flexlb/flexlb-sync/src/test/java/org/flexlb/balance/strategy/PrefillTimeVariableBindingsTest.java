package org.flexlb.balance.strategy;

import org.flexlb.balance.scheduler.BatchItem;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.DebugInfo;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

/**
 * Tests for {@link PrefillTimeVariableBindings}, focusing on the
 * {@code batchVariables(items, extraSeqLen, extraCacheHit)} overload that
 * appends a virtual item for a not-yet-enqueued request.
 */
class PrefillTimeVariableBindingsTest {

    /**
     * The overload must produce identical variable bindings to the regular
     * {@link PrefillTimeVariableBindings#batchVariables(List)} applied to a
     * merged list (existing items + a synthetic item for the extra request).
     *
     * <p>Because both calls share a {@link ThreadLocal} {@code BindingContext},
     * the first result is snapshotted into independent arrays before the
     * second call overwrites the thread-local state.
     */
    @Test
    void batchVariablesExtraMatchesMergedList() {
        List<BatchItem> items = new ArrayList<>(List.of(
                batchItem(500, 200),
                batchItem(300, 100)));
        long extraSeqLen = 1000;
        long extraCacheHit = 400;

        // 1. Call the overload and snapshot (ThreadLocal arrays are reused)
        PrefillTimeVariableBindings.EvaluationVariables overloadResult =
                PrefillTimeVariableBindings.batchVariables(items, extraSeqLen, extraCacheHit);
        double[] overloadTopLevel = Arrays.copyOf(overloadResult.topLevelVars(),
                overloadResult.topLevelVars().length);
        List<double[]> overloadItems = new ArrayList<>();
        for (double[] arr : overloadResult.itemVars()) {
            overloadItems.add(Arrays.copyOf(arr, arr.length));
        }

        // 2. Build merged list: items + synthetic item for the extra request
        List<BatchItem> merged = new ArrayList<>(items);
        merged.add(batchItem(extraSeqLen, extraCacheHit));

        // 3. Call the regular method
        PrefillTimeVariableBindings.EvaluationVariables mergedResult =
                PrefillTimeVariableBindings.batchVariables(merged);

        // 4. Compare — batchSize, totals, maxima, and per-item vars must all match
        assertArrayEquals(overloadTopLevel, mergedResult.topLevelVars(), 0.001,
                "top-level variables must match between overload and merged-list");
        assertEquals(overloadItems.size(), mergedResult.itemVars().size(),
                "item count must match");
        for (int i = 0; i < overloadItems.size(); i++) {
            assertArrayEquals(overloadItems.get(i), mergedResult.itemVars().get(i), 0.001,
                    "item " + i + " variables must match");
        }
    }

    /**
     * With a single existing item, the overload produces a 2-item batch
     * whose batchSize is 2.
     */
    @Test
    void batchVariablesExtraWithOneItemProducesBatchSizeTwo() {
        List<BatchItem> items = new ArrayList<>(List.of(batchItem(500, 200)));

        PrefillTimeVariableBindings.EvaluationVariables vars =
                PrefillTimeVariableBindings.batchVariables(items, 300, 0);

        assertEquals(2.0, vars.topLevelVars()[PrefillTimeFormula.IDX_BATCH_SIZE], 0.001);
        assertEquals(2, vars.itemVars().size());
    }

    /**
     * With no existing items, the overload degenerates to a single-request
     * batch (batchSize = 1, one item var entry).
     */
    @Test
    void batchVariablesExtraEmptyItemsProducesSingleRequestBatch() {
        PrefillTimeVariableBindings.EvaluationVariables vars =
                PrefillTimeVariableBindings.batchVariables(List.of(), 500, 200);

        assertEquals(1.0, vars.topLevelVars()[PrefillTimeFormula.IDX_BATCH_SIZE], 0.001);
        assertEquals(1, vars.itemVars().size());
    }

    // ---- helpers ----

    private static BatchItem batchItem(long seqLen, long hitCacheLen) {
        Request request = new Request();
        request.setRequestId(1L);
        request.setSeqLen(seqLen);

        BalanceContext ctx = new BalanceContext();
        ctx.setRequest(request);

        ServerStatus prefill = new ServerStatus();
        DebugInfo debugInfo = new DebugInfo();
        debugInfo.setHitCacheLen(hitCacheLen);
        prefill.setDebugInfo(debugInfo);

        return new BatchItem(ctx, null, null, prefill, null, null, null, 0, System.currentTimeMillis());
    }
}
