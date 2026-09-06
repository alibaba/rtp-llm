package org.flexlb.balance.strategy;

import org.flexlb.balance.scheduler.BatchItem;
import org.flexlb.balance.scheduler.SchedulingTestConfig;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.DebugInfo;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
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

    // ---- production DSv4 fit (explicit injection) ----

    /**
     * Production DSv4 prefill execution-time fit, injected explicitly by
     * the test (verbatim the same expression the harness injects into
     * generated FLEXLB_CONFIG documents and
     * data/config/master_fixed_window.json). The production code default is
     * the upstream legacy "1 ms/token" expression, so tests verifying the
     * production-fit caliber construct their predictor with an explicit
     * expression, never the code default.
     */
    private static final String DSV4_PRODUCTION_EXPRESSION =
            "max(196, -68.612174288157 + 0.993068319341 * (max(0, 287.3980926717 + 2.30134977837751 * batchSize + "
            + "0.158123254797307 * sum(hitCacheTokens / 1024.) + 0.575522710053703 * sum(computeTokens / 1024.) + "
            + "0.0517623430739831 * sum(computeTokens / 1024. * computeTokens / 1024.) + 0.0395308136993267 * "
            + "sum(hitCacheTokens / 1024. * computeTokens / 1024.) + 0.0104363634681015 * sum(hitCacheTokens / 1024. * "
            + "hitCacheTokens / 1024.) + 0.575522710053703 * max(sum(computeTokens / 1024.) - 16, 0) + 2.82077211814514 "
            + "* max(sum(computeTokens / 1024.) - 32, 0) - 0.0254671429192862 * max(sum(computeTokens / 1024.) - 64, 0) "
            + "+ 2.15779213792494 * max(sum(computeTokens / 1024.) - 96, 0) + 0.247806025472364 * "
            + "max(sum(hitCacheTokens / 1024.) - 32, 0) - 0.444522654549492 * max(sum(hitCacheTokens / 1024.) - 64, 0) "
            + "- 0.427317020061895 * max(sum(hitCacheTokens / 1024.) - 128, 0) + 0.347029077528455 * "
            + "max(sum(hitCacheTokens / 1024.) - 256, 0) - 0.298742307762735 * max(sum(hitCacheTokens / 1024.) - 384, "
            + "0) + 2.30134977837751 * max(batchSize - 8, 0) - 3.54884859699154 * max(batchSize - 16, 0) - "
            + "11.3438560779984 * max(batchSize - 24, 0) + 0.879751992138183 * sum(max(computeTokens / 1024. - 2, 0)) + "
            + "0.636364578079591 * sum(max(computeTokens / 1024. - 4, 0)) - 0.0513345988517118 * sum(max(computeTokens "
            + "/ 1024. - 8, 0)) - 0.332584389129357 * sum(max(hitCacheTokens / 1024. - 2, 0)) + 0.305819761192588 * "
            + "sum(max(hitCacheTokens / 1024. - 4, 0)) - 0.287610979974721 * sum(max(hitCacheTokens / 1024. - 8, 0)) + "
            + "0.191310200712013 * sum(max(hitCacheTokens / 1024. - 12, 0)) + 0.0130251644478961 * max(batchSize - 8, "
            + "0) * sum(hitCacheTokens / 1024.) + 0.00981382840761646 * max(batchSize - 16, 0) * sum(hitCacheTokens / "
            + "1024.) - 0.0299132587297009 * max(batchSize - 24, 0) * sum(hitCacheTokens / 1024.) + 0.0447455122487382 "
            + "* max(batchSize - 8, 0) * sum(computeTokens / 1024.) + 0.0104635312001851 * max(batchSize - 16, 0) * "
            + "sum(computeTokens / 1024.) + 0.0542737877321807 * max(batchSize - 24, 0) * sum(computeTokens / 1024.))))";

    @Test
    void dsv4ProductionFitPredictsProductionScalePrefillLatency() {
        FormulaPredictor p = new FormulaPredictor(DSV4_PRODUCTION_EXPRESSION);

        // All-miss single requests, verified against the production DSv4 fit:
        // 512 -> ~219 ms, 32768 -> ~342 ms, 49152 -> ~494 ms. The legacy
        // 1 ms/token default predicted 32768 -> 32768 ms (96x too slow).
        long v512 = p.estimateMs(512, 0);
        long v32k = p.estimateMs(32768, 0);
        long v48k = p.estimateMs(49152, 0);
        assertTrue(v512 >= 210 && v512 <= 230,
                "512 all-miss expected ~219ms, got " + v512);
        assertTrue(v32k >= 330 && v32k <= 350,
                "32768 all-miss expected ~342ms, got " + v32k);
        assertTrue(v48k >= 480 && v48k <= 510,
                "49152 all-miss expected ~494ms, got " + v48k);

        // Cache hits shorten the prediction: 32768 with half cached ~274ms.
        long v32kHalf = p.estimateMs(32768, 16384);
        assertTrue(v32kHalf >= 260 && v32kHalf <= 290,
                "32768 half-cached expected ~274ms, got " + v32kHalf);
        assertTrue(v32kHalf < v32k, "cache hits must reduce the prediction");
    }

    @Test
    void dsv4ProductionFitEvaluatesInBatchModeWithBatchSizeStairs() {
        FormulaPredictor p = new FormulaPredictor(DSV4_PRODUCTION_EXPRESSION);

        long single = p.estimateMs(512, 0);

        List<BatchItem> items = new ArrayList<>();
        for (int i = 0; i < 32; i++) {
            items.add(batchItem(512, 0));
        }
        long batch = (long) p.predictBatchMs(items);

        // batchSize stairs engage only in batch mode: a 32-request batch of
        // 512-token all-miss requests predicts above a single request but
        // stays in the same order of magnitude (continuous-batching fit).
        assertTrue(batch > single,
                "batch of 32 should exceed a single request, got " + batch);
        assertTrue(batch < single * 4,
                "batch penalty should stay moderate, got " + batch);
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
    @DisplayName("learn method accepts immutable batch features without error")
    void learnAcceptsBatchInfo() {
        FormulaPredictor p = new FormulaPredictor("100");
        List<BatchItem> items = List.of(
                batchItem(100, 20),
                batchItem(200, 50)
        );
        p.learn(PrefillBatchFeatures.from(items), 150, 300);
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
    @DisplayName("same parameter name reused across formula shares one ParameterNode")
    void paramSameNameReused() {
        // param(w0, 1) * computeTokens + param(w0, 1) * hitCacheTokens
        // inputTokens=100, hitCache=50, computeTokens=50 → 1*50 + 1*50 = 100
        FormulaPredictor p = new FormulaPredictor("param(w0, 1) * computeTokens + param(w0, 1) * hitCacheTokens");
        assertEquals(100, p.estimateMs(100, 50));
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
    @DisplayName("param() initial value can be an expression")
    void paramInitialValueExpression() {
        // param(w0, 2+3) * computeTokens → 5 * 100 = 500
        FormulaPredictor p = new FormulaPredictor("param(w0, 2+3) * computeTokens");
        assertEquals(500, p.estimateMs(100, 0));
    }

    // ---- cache behaviour ----

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

    // ---- helpers ----

    private static BatchItem batchItem(long seqLen, long hitCacheLen) {
        Request request = new Request();
        request.setRequestId("1");
        request.setSeqLen(seqLen);

        BalanceContext ctx = new BalanceContext();
        ctx.setConfig(SchedulingTestConfig.batchConfig());
        ctx.setRequest(request);

        ServerStatus prefill = new ServerStatus();
        DebugInfo debugInfo = new DebugInfo();
        debugInfo.setHitCacheLen(hitCacheLen);
        prefill.setDebugInfo(debugInfo);

        return new BatchItem(ctx, null, null, prefill, null, null, null, System.currentTimeMillis());
    }
}
