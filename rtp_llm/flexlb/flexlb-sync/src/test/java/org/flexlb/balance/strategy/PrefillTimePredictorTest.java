package org.flexlb.balance.strategy;

import org.flexlb.balance.scheduler.BatchItem;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.DebugInfo;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PrefillTimePredictorTest {

    // ---- formula parsing ----

    @Test
    void parseRejectsUnknownVariable() {
        assertThrows(IllegalArgumentException.class, () ->
                new PrefillTimePredictor("unknown_var + 5"));
    }

    @Test
    void parseRejectsMalformed() {
        assertThrows(IllegalArgumentException.class, () ->
                new PrefillTimePredictor("sum(computeTokens) +"));
    }

    @Test
    void parseRejectsShortLegacyVariables() {
        assertThrows(IllegalArgumentException.class, () ->
                new PrefillTimePredictor("c + p + sum_c + n"));
    }

    // ---- estimateMs (single request) ----

    @Test
    void estimateMsEmptyFormula() {
        // "0" → always 0
        PrefillTimePredictor p = new PrefillTimePredictor("0");
        assertEquals(0, p.estimateMs(1000, 0));
        assertEquals(0, p.estimateMs(1000, 500));
    }

    @Test
    void estimateMsConstantTerm() {
        // "50" → always 50
        PrefillTimePredictor p = new PrefillTimePredictor("50");
        assertEquals(50, p.estimateMs(100, 0));
        assertEquals(50, p.estimateMs(0, 0));
    }

    @Test
    void estimateMsLinearInComputeTokens() {
        PrefillTimePredictor p = new PrefillTimePredictor("2*computeTokens");
        assertEquals(2000, p.estimateMs(1500, 500));
        assertEquals(600, p.estimateMs(300, 0));
    }

    @Test
    void estimateMsQuadraticInComputeTokens() {
        PrefillTimePredictor p = new PrefillTimePredictor("0.1*computeTokens^2");
        assertEquals(1000, p.estimateMs(100, 0));
    }

    @Test
    void estimateMsInteractionTerm() {
        PrefillTimePredictor p = new PrefillTimePredictor("0.5*computeTokens*hitCacheTokens");
        assertEquals(40000, p.estimateMs(600, 400));
    }

    @Test
    void estimateMsSumFunctionInSingleMode() {
        PrefillTimePredictor p = new PrefillTimePredictor(
                "sum(computeTokens) + 0.3*sum(hitCacheTokens)");
        assertEquals(360, p.estimateMs(500, 200));
    }

    @Test
    void estimateMsHitCacheRequestCount() {
        PrefillTimePredictor p = new PrefillTimePredictor(
                "sum(hitCacheTokens) + 100*sum(hasHitCache)");

        assertEquals(300, p.estimateMs(500, 200));
        assertEquals(0, p.estimateMs(500, 0));
    }

    @Test
    void estimateMsReadablePositivePartFormula() {
        PrefillTimePredictor p = new PrefillTimePredictor(
                "max(computeTokens - 2048, 0) + 2*max(computeTokens - 24576, 0)"
                        + " + sum(max(computeTokens - 2048, 0))"
                        + " + 3*sum(max(computeTokens - 24576, 0))");

        // tokens=30000, hitCacheTokens=1000, computeTokens=29000, positive parts=(26952,4424).
        assertEquals(76024, p.estimateMs(30000, 1000));
        assertEquals(0, p.estimateMs(2048, 0));
    }

    @Test
    void estimateMsReadableTokenVariables() {
        PrefillTimePredictor p = new PrefillTimePredictor(
                "inputTokens - hitCacheTokens + computeTokens + 10*hasHitCache");

        assertEquals(610, p.estimateMs(500, 200));
        assertEquals(1000, p.estimateMs(500, 0));
    }

    @Test
    void estimateMsFullFormula() {
        // inputTokens=500, hitCacheTokens=200, computeTokens=300
        // = 10 + 30 + 900 + 60 + 100 + 5 = 1105
        PrefillTimePredictor p = new PrefillTimePredictor(
                "10 + 0.1*sum(computeTokens)"
                        + " + 0.01*sum(computeTokens^2)"
                        + " + 0.001*sum(computeTokens * hitCacheTokens)"
                        + " + 0.5*sum(hitCacheTokens)"
                        + " + 5*batchSize");
        assertEquals(1105, p.estimateMs(500, 200));
    }

    @Test
    void estimateMsHitTokensCannotExceedTotal() {
        PrefillTimePredictor p = new PrefillTimePredictor("2*computeTokens");
        assertEquals(0, p.estimateMs(100, 500));
    }

    @Test
    void estimateMsLargeValuesNoOverflow() {
        PrefillTimePredictor p = new PrefillTimePredictor(
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
        PrefillTimePredictor p = new PrefillTimePredictor("10 + sum(computeTokens) + 5*batchSize");
        assertEquals(0, p.predictBatchMs(List.of()));
    }

    @Test
    void predictBatchMsSingleItemMatchesEstimateMs() {
        PrefillTimePredictor p = new PrefillTimePredictor(
                "10 + 0.1*sum(computeTokens)"
                        + " + 0.01*sum(computeTokens^2)"
                        + " + 0.001*sum(computeTokens * hitCacheTokens)"
                        + " + 0.5*sum(hitCacheTokens)"
                        + " + 5*batchSize");
        long single = p.estimateMs(500, 200);

        BatchItem item = batchItem(500, 200);
        long batch = p.predictBatchMs(List.of(item));

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
        PrefillTimePredictor p = new PrefillTimePredictor(
                "10 + 0.1*sum(computeTokens)"
                        + " + 0.01*sum(computeTokens^2)"
                        + " + 0.001*sum(computeTokens * hitCacheTokens)"
                        + " + 0.5*sum(hitCacheTokens)"
                        + " + 5*batchSize");

        BatchItem item1 = batchItem(500, 200);
        BatchItem item2 = batchItem(300, 100);
        long result = p.predictBatchMs(List.of(item1, item2));

        assertEquals(1600, result);
    }

    @Test
    void predictBatchMsAggregatesHitCacheRequestCount() {
        PrefillTimePredictor p = new PrefillTimePredictor(
                "sum(hitCacheTokens) + 100*sum(hasHitCache)");

        BatchItem item1 = batchItem(500, 200);
        BatchItem item2 = batchItem(300, 0);
        BatchItem item3 = batchItem(400, 400);
        long result = p.predictBatchMs(List.of(item1, item2, item3));

        assertEquals(800, result);
    }

    @Test
    void predictBatchMsAggregatesReadablePositivePartFormula() {
        PrefillTimePredictor p = new PrefillTimePredictor(
                "sum(max(computeTokens - 2048, 0))"
                        + " + 2*sum(max(computeTokens - 24576, 0))");

        BatchItem item1 = batchItem(30000, 1000); // computeTokens=29000, positive parts=(26952,4424)
        BatchItem item2 = batchItem(4096, 0);     // computeTokens=4096, positive parts=(2048,0)
        long result = p.predictBatchMs(List.of(item1, item2));

        assertEquals(37848, result);
    }

    @Test
    void predictBatchMsRecommendedFormulaUsesBatchBoundedCacheTerms() {
        PrefillTimePredictor p = new PrefillTimePredictor(
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
        PrefillTimePredictor p = new PrefillTimePredictor(
                "sum(max(computeTokens - 2048, 0))");

        BatchItem item1 = batchItem(3000, 0); // max(3000-2048,0)=952
        BatchItem item2 = batchItem(1000, 0); // max(1000-2048,0)=0

        assertEquals(952, p.predictBatchMs(List.of(item1, item2)));
    }

    @Test
    void predictBatchMsBatchSizeAffectsResult() {
        PrefillTimePredictor p = new PrefillTimePredictor("10*batchSize");

        BatchItem item = batchItem(100, 0);
        assertEquals(10, p.predictBatchMs(List.of(item)));
        assertEquals(20, p.predictBatchMs(List.of(item, item)));
        assertEquals(30, p.predictBatchMs(List.of(item, item, item)));
    }

    @Test
    void predictBatchMsZeroCacheHits() {
        PrefillTimePredictor p = new PrefillTimePredictor("sum(computeTokens)");
        BatchItem item = batchItem(500, 0);
        assertEquals(500, p.predictBatchMs(List.of(item)));
    }

    @Test
    void predictBatchMsAllCached() {
        PrefillTimePredictor p = new PrefillTimePredictor("sum(computeTokens)");
        BatchItem item = batchItem(500, 500);
        assertEquals(0, p.predictBatchMs(List.of(item)));
    }

    @Test
    void predictBatchMsLargeBatch() {
        PrefillTimePredictor p = new PrefillTimePredictor(
                "100 + 0.5*sum(computeTokens) + 0.1*sum(hitCacheTokens) + 3*batchSize");
        List<BatchItem> items = new ArrayList<>();
        for (int i = 0; i < 100; i++) {
            items.add(batchItem(1000, 200));
        }
        long result = p.predictBatchMs(items);
        assertTrue(result > 0, "Large batch should produce positive prediction");
    }

    // ---- power operator ----

    @Test
    void powerOperatorRightAssociative() {
        // 2^3^2 = 2^(3^2) = 2^9 = 512
        PrefillTimePredictor p = new PrefillTimePredictor("2^3^2");
        assertEquals(512, p.estimateMs(0, 0));
    }

    // ---- functions ----

    @Test
    void sqrtFunction() {
        PrefillTimePredictor p = new PrefillTimePredictor("sqrt(100)");
        assertEquals(10, p.estimateMs(0, 0));
    }

    @Test
    void maxFunction() {
        PrefillTimePredictor p = new PrefillTimePredictor("max(sum(computeTokens), 50)");
        assertEquals(100, p.estimateMs(100, 0));
        assertEquals(50, p.estimateMs(30, 0));
    }

    @Test
    void nestedFunctions() {
        PrefillTimePredictor p = new PrefillTimePredictor(
                "sqrt(pow(sum(computeTokens), 2) + pow(sum(hitCacheTokens), 2))");
        // inputTokens=7, hitCacheTokens=4, computeTokens=3, sqrt(9+16) = 5
        assertEquals(5, p.estimateMs(7, 4));
    }

    // ---- parentheses ----

    @Test
    void parenthesesOverridePrecedence() {
        PrefillTimePredictor p = new PrefillTimePredictor("(2 + 3) * 4");
        assertEquals(20, p.estimateMs(0, 0));
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
