package org.flexlb.balance.strategy;

import org.flexlb.balance.scheduler.BatchItem;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.DebugInfo;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FormulaPredictorTest {

    // ---- formula parsing ----

    @Test
    void parseRejectsUnknownVariable() {
        assertThrows(IllegalArgumentException.class, () ->
                new FormulaPredictor("unknown_var + 5"));
    }

    @Test
    void parseRejectsMalformed() {
        assertThrows(IllegalArgumentException.class, () ->
                new FormulaPredictor("sum(computeTokens) +"));
    }

    @Test
    void parseRejectsShortLegacyVariables() {
        assertThrows(IllegalArgumentException.class, () ->
                new FormulaPredictor("c + p + sum_c + n"));
    }

    @Test
    void parseRejectsBatchScopedVariablesInsideSum() {
        assertThrows(IllegalArgumentException.class, () ->
                new FormulaPredictor("sum(totalComputeTokens)"));
        assertThrows(IllegalArgumentException.class, () ->
                new FormulaPredictor("sum(batchSize)"));
    }

    // ---- estimateMs (single request) ----

    @Test
    void estimateMsEmptyFormula() {
        // "0" → always 0
        FormulaPredictor p = new FormulaPredictor("0");
        assertEquals(0, p.estimateMs(1000, 0));
        assertEquals(0, p.estimateMs(1000, 500));
    }

    @Test
    void estimateMsConstantTerm() {
        // "50" → always 50
        FormulaPredictor p = new FormulaPredictor("50");
        assertEquals(50, p.estimateMs(100, 0));
        assertEquals(50, p.estimateMs(0, 0));
    }

    @Test
    void estimateMsLinearInComputeTokens() {
        FormulaPredictor p = new FormulaPredictor("2*computeTokens");
        assertEquals(2000, p.estimateMs(1500, 500));
        assertEquals(600, p.estimateMs(300, 0));
    }

    @Test
    void estimateMsQuadraticInComputeTokens() {
        FormulaPredictor p = new FormulaPredictor("0.1*computeTokens^2");
        assertEquals(1000, p.estimateMs(100, 0));
    }

    @Test
    void estimateMsInteractionTerm() {
        FormulaPredictor p = new FormulaPredictor("0.5*computeTokens*hitCacheTokens");
        assertEquals(40000, p.estimateMs(600, 400));
    }

    @Test
    void estimateMsSumFunctionInSingleMode() {
        FormulaPredictor p = new FormulaPredictor(
                "sum(computeTokens) + 0.3*sum(hitCacheTokens)");
        assertEquals(360, p.estimateMs(500, 200));
    }

    @Test
    void estimateMsHitCacheRequestCount() {
        FormulaPredictor p = new FormulaPredictor(
                "sum(hitCacheTokens) + 100*sum(hasHitCache)");

        assertEquals(300, p.estimateMs(500, 200));
        assertEquals(0, p.estimateMs(500, 0));
    }

    @Test
    void estimateMsReadablePositivePartFormula() {
        FormulaPredictor p = new FormulaPredictor(
                "max(computeTokens - 2048, 0) + 2*max(computeTokens - 24576, 0)"
                        + " + sum(max(computeTokens - 2048, 0))"
                        + " + 3*sum(max(computeTokens - 24576, 0))");

        // tokens=30000, hitCacheTokens=1000, computeTokens=29000, positive parts=(26952,4424).
        assertEquals(76024, p.estimateMs(30000, 1000));
        assertEquals(0, p.estimateMs(2048, 0));
    }

    @Test
    void estimateMsReadableTokenVariables() {
        FormulaPredictor p = new FormulaPredictor(
                "inputTokens - hitCacheTokens + computeTokens + 10*hasHitCache");

        assertEquals(610, p.estimateMs(500, 200));
        assertEquals(1000, p.estimateMs(500, 0));
    }

    @Test
    void estimateMsFullFormula() {
        // inputTokens=500, hitCacheTokens=200, computeTokens=300
        // = 10 + 30 + 900 + 60 + 100 + 5 = 1105
        FormulaPredictor p = new FormulaPredictor(
                "10 + 0.1*sum(computeTokens)"
                        + " + 0.01*sum(computeTokens^2)"
                        + " + 0.001*sum(computeTokens * hitCacheTokens)"
                        + " + 0.5*sum(hitCacheTokens)"
                        + " + 5*batchSize");
        assertEquals(1105, p.estimateMs(500, 200));
    }

    @Test
    void estimateMsHitTokensCannotExceedTotal() {
        FormulaPredictor p = new FormulaPredictor("2*computeTokens");
        assertEquals(0, p.estimateMs(100, 500));
    }

    @Test
    void estimateMsLargeValuesNoOverflow() {
        FormulaPredictor p = new FormulaPredictor(
                "100 + sum(computeTokens)"
                        + " + 0.001*sum(computeTokens^2)"
                        + " + 0.0001*sum(computeTokens * hitCacheTokens)"
                        + " + 0.5*sum(hitCacheTokens)"
                        + " + 10*batchSize");
        long result = p.estimateMs(100_000, 50_000);
        assertTrue(result >= 0, "Should not overflow or produce negative values");
    }

    // ---- predictBatchMs ----

    @Test
    void predictBatchMsEmptyListReturnsZero() {
        FormulaPredictor p = new FormulaPredictor("10 + sum(computeTokens) + 5*batchSize");
        assertEquals(0, (long) p.predictBatchMs(List.of()));
    }

    @Test
    void predictBatchMsSingleItemMatchesEstimateMs() {
        FormulaPredictor p = new FormulaPredictor(
                "10 + 0.1*sum(computeTokens)"
                        + " + 0.01*sum(computeTokens^2)"
                        + " + 0.001*sum(computeTokens * hitCacheTokens)"
                        + " + 0.5*sum(hitCacheTokens)"
                        + " + 5*batchSize");
        long single = p.estimateMs(500, 200);

        BatchItem item = batchItem(500, 200);
        long batch = (long) p.predictBatchMs(List.of(item));

        assertEquals(single, batch);
    }

    @Test
    void predictBatchMsMultipleItems() {
        // item1: inputTokens=500, hitCacheTokens=200, computeTokens=300
        // item2: inputTokens=300, hitCacheTokens=100, computeTokens=200
        // sum(computeTokens)=500, sum(computeTokens^2)=130000,
        // sum(computeTokens * hitCacheTokens)=80000, sum(hitCacheTokens)=300, batchSize=2
        // = 10 + 0.1*500 + 0.01*130000 + 0.001*80000 + 0.5*300 + 5*2
        // = 10 + 50 + 1300 + 80 + 150 + 10 = 1600
        FormulaPredictor p = new FormulaPredictor(
                "10 + 0.1*sum(computeTokens)"
                        + " + 0.01*sum(computeTokens^2)"
                        + " + 0.001*sum(computeTokens * hitCacheTokens)"
                        + " + 0.5*sum(hitCacheTokens)"
                        + " + 5*batchSize");

        BatchItem item1 = batchItem(500, 200);
        BatchItem item2 = batchItem(300, 100);
        long result = (long) p.predictBatchMs(List.of(item1, item2));

        assertEquals(1600, result);
    }

    @Test
    void predictBatchMsExposesExplicitBatchTotalsAndMaxima() {
        FormulaPredictor p = new FormulaPredictor(
                "batchSize + totalInputTokens + totalHitCacheTokens + totalComputeTokens"
                        + " + maxInputTokens + maxComputeTokens");

        BatchItem item1 = batchItem(500, 200);
        BatchItem item2 = batchItem(300, 100);

        // 2 + 800 + 300 + 500 + 500 + 300
        assertEquals(2402, p.predictBatchMs(List.of(item1, item2)));
        // Single-request mode binds the same explicit batch variables.
        assertEquals(1801, p.estimateMs(500, 200));
    }

    @Test
    void batchTotalSquareIsNotPerRequestSquareSum() {
        FormulaPredictor p = new FormulaPredictor(
                "totalComputeTokens^2 - sum(computeTokens^2)");

        BatchItem item1 = batchItem(500, 200); // compute=300
        BatchItem item2 = batchItem(300, 100); // compute=200

        // (300 + 200)^2 - (300^2 + 200^2) = 120000.
        assertEquals(120000, p.predictBatchMs(List.of(item1, item2)));
    }

    @Test
    void predictBatchMsAggregatesHitCacheRequestCount() {
        FormulaPredictor p = new FormulaPredictor(
                "sum(hitCacheTokens) + 100*sum(hasHitCache)");

        BatchItem item1 = batchItem(500, 200);
        BatchItem item2 = batchItem(300, 0);
        BatchItem item3 = batchItem(400, 400);
        long result = (long) p.predictBatchMs(List.of(item1, item2, item3));

        assertEquals(800, result);
    }

    @Test
    void predictBatchMsAggregatesReadablePositivePartFormula() {
        FormulaPredictor p = new FormulaPredictor(
                "sum(max(computeTokens - 2048, 0))"
                        + " + 2*sum(max(computeTokens - 24576, 0))");

        BatchItem item1 = batchItem(30000, 1000); // computeTokens=29000, positive parts=(26952,4424)
        BatchItem item2 = batchItem(4096, 0);     // computeTokens=4096, positive parts=(2048,0)
        long result = (long) p.predictBatchMs(List.of(item1, item2));

        assertEquals(37848, result);
    }

    @Test
    void predictBatchMsRecommendedFormulaUsesBatchBoundedCacheTerms() {
        FormulaPredictor p = new FormulaPredictor(
                "174.374677211 + 52.642812003*log(batchSize + 1)"
                        + " + 0.000746856881262*sum(2048*log(1 + exp((computeTokens - 8192)/2048)))"
                        + " + 0.0074536400604*sum(4096*log(1 + exp((computeTokens - 24576)/4096)))"
                        + " + 5.73664292066e-05*sum(8192*log(1 + exp((computeTokens - 65536)/8192)))"
                        + " + 0.00111135741393*sum(8192*log(1 + exp((computeTokens - 81920)/8192)))"
                        + " + 0.00424878987222*sum((hitCacheTokens/(inputTokens + 1))"
                        + " * (4096*log(1 + exp((computeTokens - 24576)/4096))))"
                        + " + 0.000489415479845*sum((log(hitCacheTokens + 1)/max(log(inputTokens + 1), 1))"
                        + " * (4096*log(1 + exp((computeTokens - 24576)/4096))))"
                        + " + 18.7646922156*(sum(hasHitCache)/batchSize)"
                        + " + 4.59475450657*(sum(hitCacheTokens/(inputTokens + 1))/batchSize)"
                        + " - 41.7583481006*(sum(log(hitCacheTokens + 1)/max(log(inputTokens + 1), 1))/batchSize)"
                        + " - 5.4218960925*(sum(hitCacheTokens/(hitCacheTokens + 4096))/batchSize)");

        List<BatchItem> fullHitBatch = new ArrayList<>();
        for (int i = 0; i < 64; i++) {
            fullHitBatch.add(batchItem(102400, 101376));
        }

        assertEquals(187, p.predictBatchMs(fullHitBatch.subList(0, 1)));
        assertEquals(246, p.predictBatchMs(fullHitBatch.subList(0, 5)));
        assertEquals(383, p.predictBatchMs(fullHitBatch));
        assertEquals(886, p.predictBatchMs(List.of(batchItem(102400, 0))));
    }

    @Test
    void predictBatchMsSumEvaluatesExpressionPerRequest() {
        FormulaPredictor p = new FormulaPredictor(
                "sum(max(computeTokens - 2048, 0))");

        BatchItem item1 = batchItem(3000, 0); // max(3000-2048,0)=952
        BatchItem item2 = batchItem(1000, 0); // max(1000-2048,0)=0

        assertEquals(952, p.predictBatchMs(List.of(item1, item2)));
    }

    @Test
    void predictBatchMsBatchSizeAffectsResult() {
        FormulaPredictor p = new FormulaPredictor("10*batchSize");

        BatchItem item = batchItem(100, 0);
        assertEquals(10, p.predictBatchMs(List.of(item)));
        assertEquals(20, p.predictBatchMs(List.of(item, item)));
        assertEquals(30, p.predictBatchMs(List.of(item, item, item)));
    }

    @Test
    void predictBatchMsZeroCacheHits() {
        FormulaPredictor p = new FormulaPredictor("sum(computeTokens)");
        BatchItem item = batchItem(500, 0);
        assertEquals(500, p.predictBatchMs(List.of(item)));
    }

    @Test
    void predictBatchMsAllCached() {
        FormulaPredictor p = new FormulaPredictor("sum(computeTokens)");
        BatchItem item = batchItem(500, 500);
        assertEquals(0, p.predictBatchMs(List.of(item)));
    }

    @Test
    void predictBatchMsLargeBatch() {
        FormulaPredictor p = new FormulaPredictor(
                "100 + 0.5*sum(computeTokens) + 0.1*sum(hitCacheTokens) + 3*batchSize");
        List<BatchItem> items = new ArrayList<>();
        for (int i = 0; i < 100; i++) {
            items.add(batchItem(1000, 200));
        }
        long result = (long) p.predictBatchMs(items);
        assertTrue(result > 0, "Large batch should produce positive prediction");
    }

    // ---- power operator ----

    @Test
    void powerOperatorRightAssociative() {
        // 2^3^2 = 2^(3^2) = 2^9 = 512
        FormulaPredictor p = new FormulaPredictor("2^3^2");
        assertEquals(512, p.estimateMs(0, 0));
    }

    // ---- functions ----

    @Test
    void sqrtFunction() {
        FormulaPredictor p = new FormulaPredictor("sqrt(100)");
        assertEquals(10, p.estimateMs(0, 0));
    }

    @Test
    void maxFunction() {
        FormulaPredictor p = new FormulaPredictor("max(sum(computeTokens), 50)");
        assertEquals(100, p.estimateMs(100, 0));
        assertEquals(50, p.estimateMs(30, 0));
    }

    @Test
    void nestedFunctions() {
        FormulaPredictor p = new FormulaPredictor(
                "sqrt(pow(sum(computeTokens), 2) + pow(sum(hitCacheTokens), 2))");
        // inputTokens=7, hitCacheTokens=4, computeTokens=3, sqrt(9+16) = 5
        assertEquals(5, p.estimateMs(7, 4));
    }

    // ---- parentheses ----

    @Test
    void parenthesesOverridePrecedence() {
        FormulaPredictor p = new FormulaPredictor("(2 + 3) * 4");
        assertEquals(20, p.estimateMs(0, 0));
    }

    // ---- learn (interface stub) ----

    @Test
    @DisplayName("learn method accepts batch items, predicted and actual time without error")
    void learnAcceptsBatchInfo() {
        FormulaPredictor p = new FormulaPredictor("100");
        List<BatchItem> items = List.of(
                batchItem(100, 20),
                batchItem(200, 50)
        );
        p.learn(items, 150, 300);  // should not throw
    }

    // ---- param() learnable parameters ----

    @Test
    @DisplayName("param() basic parsing returns initial value")
    void paramBasicParsing() {
        FormulaPredictor p = new FormulaPredictor("param(w0, 100)");
        assertEquals(100, p.estimateMs(0, 0));
        assertEquals(100, p.estimateMs(500, 200));
    }

    @Test
    @DisplayName("param() in expression with variables")
    void paramInExpression() {
        // param(w0, 10) + param(w1, 0.5) * computeTokens
        // inputTokens=100, hitCache=0, computeTokens=100 → 10 + 0.5*100 = 60
        FormulaPredictor p = new FormulaPredictor("param(w0, 10) + param(w1, 0.5) * computeTokens");
        assertEquals(60, p.estimateMs(100, 0));
    }

    @Test
    @DisplayName("setParameter updates parameter value at runtime")
    void paramUpdateValue() {
        FormulaPredictor p = new FormulaPredictor("param(w0, 10) + param(w1, 0.5) * computeTokens");
        assertEquals(60, p.estimateMs(100, 0));
        p.setParameter("w1", 1.0);
        // 10 + 1.0*100 = 110
        assertEquals(110, p.estimateMs(100, 0));
    }

    @Test
    @DisplayName("parameterNames returns all parameter names")
    void parameterNamesListing() {
        FormulaPredictor p = new FormulaPredictor("param(w0, 1) + param(w1, 2) + param(w2, 3)");
        assertEquals(Set.of("w0", "w1", "w2"), p.parameterNames());
    }

    @Test
    @DisplayName("getParameters returns all parameter values")
    void getParametersMap() {
        FormulaPredictor p = new FormulaPredictor("param(w0, 1) + param(w1, 2) + param(w2, 3)");
        assertEquals(Map.of("w0", 1.0, "w1", 2.0, "w2", 3.0), p.getParameters());
    }

    @Test
    @DisplayName("same parameter name reused across formula shares one ParameterNode")
    void paramSameNameReused() {
        // param(w0, 1) * computeTokens + param(w0, 1) * hitCacheTokens
        // inputTokens=100, hitCache=50, computeTokens=50 → 1*50 + 1*50 = 100
        FormulaPredictor p = new FormulaPredictor("param(w0, 1) * computeTokens + param(w0, 1) * hitCacheTokens");
        assertEquals(1, p.parameterNames().size());
        assertTrue(p.parameterNames().contains("w0"));
        assertEquals(100, p.estimateMs(100, 50));
        // setParameter("w0", 2) → 2*50 + 2*50 = 200
        p.setParameter("w0", 2.0);
        assertEquals(200, p.estimateMs(100, 50));
    }

    @Test
    @DisplayName("formula without param() has no parameters")
    void noParametersFormula() {
        FormulaPredictor p = new FormulaPredictor("sum(computeTokens)");
        assertFalse(p.hasParameters());
        assertTrue(p.parameterNames().isEmpty());
    }

    @Test
    @DisplayName("param() works in batch mode with sum()")
    void paramInBatchMode() {
        // param(w0, 10) + param(w1, 0.5) * sum(computeTokens)
        // item1: (500,200) → computeTokens=300
        // item2: (300,100) → computeTokens=200
        // sum(computeTokens) = 500 → 10 + 0.5*500 = 260
        FormulaPredictor p = new FormulaPredictor("param(w0, 10) + param(w1, 0.5) * sum(computeTokens)");
        BatchItem item1 = batchItem(500, 200);
        BatchItem item2 = batchItem(300, 100);
        assertEquals(260, p.predictBatchMs(List.of(item1, item2)));
    }

    @Test
    @DisplayName("getParameter on unknown parameter throws IllegalArgumentException")
    void unknownParameterThrows() {
        FormulaPredictor p = new FormulaPredictor("param(w0, 100)");
        assertThrows(IllegalArgumentException.class, () -> p.getParameter("nonexistent"));
    }

    @Test
    @DisplayName("param() initial value can be an expression")
    void paramInitialValueExpression() {
        // param(w0, 2+3) * computeTokens → 5 * 100 = 500
        FormulaPredictor p = new FormulaPredictor("param(w0, 2+3) * computeTokens");
        assertEquals(5.0, p.getParameter("w0"));
        assertEquals(500, p.estimateMs(100, 0));
    }

    // ---- cache behaviour ----

    @Test
    @DisplayName("predictBatchMs cache invalidated on setParameter")
    void predictBatchMsCacheInvalidatedOnSetParameter() {
        FormulaPredictor p = new FormulaPredictor(
                "param(w0, 10) + param(w1, 0.5) * sum(computeTokens)");
        BatchItem item = batchItem(100, 0);
        // 10 + 0.5*100 = 60
        assertEquals(60, (long) p.predictBatchMs(List.of(item)));
        p.setParameter("w1", 1.0);
        // 10 + 1.0*100 = 110 — 缓存必须已失效
        assertEquals(110, (long) p.predictBatchMs(List.of(item)));
    }

    @Test
    @DisplayName("predictBatchMs cache hit returns same result")
    void predictBatchMsCacheHitReturnsSameResult() {
        FormulaPredictor p = new FormulaPredictor("50 + sum(computeTokens)");
        BatchItem item1 = batchItem(100, 0);
        BatchItem item2 = batchItem(200, 50);
        // 50 + (100 + 150) = 300
        double first = p.predictBatchMs(List.of(item1, item2));
        double second = p.predictBatchMs(List.of(item1, item2));
        assertEquals(first, second, 0.001);
        assertEquals(300, (long) first);
    }

    // ---- batch-level prediction with extra request (not yet enqueued) ----

    @Test
    @DisplayName("predictBatchMs(emptyList, seqLen, cacheHit) matches estimateMs(seqLen, cacheHit)")
    void predictBatchMsEmptyItemsWithExtraMatchesEstimateMs() {
        FormulaPredictor p = new FormulaPredictor(
                "10 + 0.1*sum(computeTokens) + 0.01*sum(computeTokens^2) + 5*batchSize");
        long seqLen = 500;
        long cacheHit = 200;

        assertEquals(p.estimateMs(seqLen, cacheHit),
                (long) p.predictBatchMs(List.of(), seqLen, cacheHit));
    }

    @Test
    @DisplayName("predictBatchMs(items, seqLen, cacheHit) exceeds estimateMs(seqLen, cacheHit)")
    void predictBatchMsWithExtraItemsExceedsSingleEstimate() {
        // Formula where a larger batch strictly increases the result.
        // estimateMs(300, 0): batchSize=1, compute=300 → 10*1 + 300 = 310
        // predictBatchMs([item(500,200)], 300, 0): batchSize=2,
        //   sum(compute) = 300 + 300 = 600 → 10*2 + 600 = 620 > 310
        FormulaPredictor p = new FormulaPredictor("10*batchSize + sum(computeTokens)");
        BatchItem existing = batchItem(500, 200);

        long singleEstimate = p.estimateMs(300, 0);
        long batchEstimate = (long) p.predictBatchMs(List.of(existing), 300, 0);

        assertTrue(batchEstimate > singleEstimate,
                "batch estimate " + batchEstimate + " should exceed single " + singleEstimate);
    }

    @Test
    @DisplayName("predictBatchMs(items, seqLen, cacheHit) matches batchVariables of merged list")
    void predictBatchMsWithExtraMatchesMergedBatch() {
        FormulaPredictor p = new FormulaPredictor(
                "10 + 0.1*sum(computeTokens) + 0.5*sum(hitCacheTokens) + 5*batchSize");
        BatchItem existing = batchItem(500, 200);
        long newSeqLen = 300;
        long newCacheHit = 100;

        // Merged batch: existing + new request as a real BatchItem
        long merged = (long) p.predictBatchMs(List.of(existing, batchItem(newSeqLen, newCacheHit)));
        // Overload: existing items + extra request
        long withExtra = (long) p.predictBatchMs(List.of(existing), newSeqLen, newCacheHit);

        assertEquals(merged, withExtra);
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
