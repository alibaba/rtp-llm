package org.flexlb.balance.prediction;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

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
        assertEquals(0, p.evaluator().estimateMs(1000, 0));
        assertEquals(0, p.evaluator().estimateMs(1000, 500));
    }

    @Test
    void estimateMsConstantTerm() {
        // "50" → always 50
        FormulaPredictor p = new FormulaPredictor("50");
        assertEquals(50, p.evaluator().estimateMs(100, 0));
        assertEquals(50, p.evaluator().estimateMs(0, 0));
    }

    @Test
    void estimateMsLinearInComputeTokens() {
        FormulaPredictor p = new FormulaPredictor("2*computeTokens");
        assertEquals(2000, p.evaluator().estimateMs(1500, 500));
        assertEquals(600, p.evaluator().estimateMs(300, 0));
    }

    @Test
    void estimateMsQuadraticInComputeTokens() {
        FormulaPredictor p = new FormulaPredictor("0.1*computeTokens^2");
        assertEquals(1000, p.evaluator().estimateMs(100, 0));
    }

    @Test
    void estimateMsInteractionTerm() {
        FormulaPredictor p = new FormulaPredictor("0.5*computeTokens*hitCacheTokens");
        assertEquals(40000, p.evaluator().estimateMs(600, 400));
    }

    @Test
    void estimateMsSumFunctionInSingleMode() {
        FormulaPredictor p = new FormulaPredictor(
                "sum(computeTokens) + 0.3*sum(hitCacheTokens)");
        assertEquals(360, p.evaluator().estimateMs(500, 200));
    }

    @Test
    void estimateMsHitCacheRequestCount() {
        FormulaPredictor p = new FormulaPredictor(
                "sum(hitCacheTokens) + 100*sum(hasHitCache)");

        assertEquals(300, p.evaluator().estimateMs(500, 200));
        assertEquals(0, p.evaluator().estimateMs(500, 0));
    }

    @Test
    void estimateMsReadablePositivePartFormula() {
        FormulaPredictor p = new FormulaPredictor(
                "max(computeTokens - 2048, 0) + 2*max(computeTokens - 24576, 0)"
                        + " + sum(max(computeTokens - 2048, 0))"
                        + " + 3*sum(max(computeTokens - 24576, 0))");

        // tokens=30000, hitCacheTokens=1000, computeTokens=29000, positive parts=(26952,4424).
        assertEquals(76024, p.evaluator().estimateMs(30000, 1000));
        assertEquals(0, p.evaluator().estimateMs(2048, 0));
    }

    @Test
    void estimateMsReadableTokenVariables() {
        FormulaPredictor p = new FormulaPredictor(
                "inputTokens - hitCacheTokens + computeTokens + 10*hasHitCache");

        assertEquals(610, p.evaluator().estimateMs(500, 200));
        assertEquals(1000, p.evaluator().estimateMs(500, 0));
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
        assertEquals(1105, p.evaluator().estimateMs(500, 200));
    }

    @Test
    void estimateMsHitTokensCannotExceedTotal() {
        FormulaPredictor p = new FormulaPredictor("2*computeTokens");
        assertEquals(0, p.evaluator().estimateMs(100, 500));
    }

    @Test
    void estimateMsLargeValuesNoOverflow() {
        FormulaPredictor p = new FormulaPredictor(
                "100 + sum(computeTokens)"
                        + " + 0.001*sum(computeTokens^2)"
                        + " + 0.0001*sum(computeTokens * hitCacheTokens)"
                        + " + 0.5*sum(hitCacheTokens)"
                        + " + 10*batchSize");
        long result = p.evaluator().estimateMs(100_000, 50_000);
        assertTrue(result >= 0, "Should not overflow or produce negative values");
    }

    // ---- predictBatchMs ----

    @Test
    void predictBatchMsEmptyBatchReturnsZero() {
        FormulaPredictor p = new FormulaPredictor("10 + sum(computeTokens) + 5*batchSize");
        assertEquals(0, (long) p.evaluator().predictBatchMs(batchFeatures()));
    }

    @Test
    void predictBatchMsSingleItemMatchesEstimateMs() {
        FormulaPredictor p = new FormulaPredictor(
                "10 + 0.1*sum(computeTokens)"
                        + " + 0.01*sum(computeTokens^2)"
                        + " + 0.001*sum(computeTokens * hitCacheTokens)"
                        + " + 0.5*sum(hitCacheTokens)"
                        + " + 5*batchSize");
        long single = p.evaluator().estimateMs(500, 200);

        PrefillBatchFeatures features = batchFeatures(item(500, 200));
        long batch = (long) p.evaluator().predictBatchMs(features);

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

        PrefillBatchFeatures features = batchFeatures(
                item(500, 200), item(300, 100));
        long result = (long) p.evaluator().predictBatchMs(features);

        assertEquals(1600, result);
    }

    @Test
    void predictBatchMsExposesExplicitBatchTotalsAndMaxima() {
        FormulaPredictor p = new FormulaPredictor(
                "batchSize + totalInputTokens + totalHitCacheTokens + totalComputeTokens"
                        + " + maxInputTokens + maxComputeTokens");

        PrefillBatchFeatures features = batchFeatures(
                item(500, 200), item(300, 100));

        // 2 + 800 + 300 + 500 + 500 + 300
        assertEquals(2402, p.evaluator().predictBatchMs(features));
        // Single-request mode binds the same explicit batch variables.
        assertEquals(1801, p.evaluator().estimateMs(500, 200));
    }

    @Test
    void batchTotalSquareIsNotPerRequestSquareSum() {
        FormulaPredictor p = new FormulaPredictor(
                "totalComputeTokens^2 - sum(computeTokens^2)");

        PrefillBatchFeatures features = batchFeatures(
                item(500, 200), // compute=300
                item(300, 100)); // compute=200

        // (300 + 200)^2 - (300^2 + 200^2) = 120000.
        assertEquals(120000, p.evaluator().predictBatchMs(features));
    }

    @Test
    void predictBatchMsAggregatesHitCacheRequestCount() {
        FormulaPredictor p = new FormulaPredictor(
                "sum(hitCacheTokens) + 100*sum(hasHitCache)");

        PrefillBatchFeatures features = batchFeatures(
                item(500, 200), item(300, 0), item(400, 400));
        long result = (long) p.evaluator().predictBatchMs(features);

        assertEquals(800, result);
    }

    @Test
    void predictBatchMsAggregatesReadablePositivePartFormula() {
        FormulaPredictor p = new FormulaPredictor(
                "sum(max(computeTokens - 2048, 0))"
                        + " + 2*sum(max(computeTokens - 24576, 0))");

        PrefillBatchFeatures features = batchFeatures(
                item(30000, 1000), // computeTokens=29000, positive parts=(26952,4424)
                item(4096, 0));    // computeTokens=4096, positive parts=(2048,0)
        long result = (long) p.evaluator().predictBatchMs(features);

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

        List<PrefillBatchFeatures.Item> fullHitBatch = new ArrayList<>();
        for (int i = 0; i < 64; i++) {
            fullHitBatch.add(item(102400, 101376));
        }

        assertEquals(187, p.evaluator().predictBatchMs(batchFeatures(fullHitBatch.subList(0, 1))));
        assertEquals(246, p.evaluator().predictBatchMs(batchFeatures(fullHitBatch.subList(0, 5))));
        assertEquals(383, p.evaluator().predictBatchMs(batchFeatures(fullHitBatch)));
        assertEquals(886, p.evaluator().predictBatchMs(batchFeatures(item(102400, 0))));
    }

    @Test
    void predictBatchMsSumEvaluatesExpressionPerRequest() {
        FormulaPredictor p = new FormulaPredictor(
                "sum(max(computeTokens - 2048, 0))");

        PrefillBatchFeatures features = batchFeatures(
                item(3000, 0), // max(3000-2048,0)=952
                item(1000, 0)); // max(1000-2048,0)=0

        assertEquals(952, p.evaluator().predictBatchMs(features));
    }

    @Test
    void predictBatchMsBatchSizeAffectsResult() {
        FormulaPredictor p = new FormulaPredictor("10*batchSize");

        PrefillBatchFeatures.Item item = item(100, 0);
        assertEquals(10, p.evaluator().predictBatchMs(batchFeatures(item)));
        assertEquals(20, p.evaluator().predictBatchMs(batchFeatures(item, item)));
        assertEquals(30, p.evaluator().predictBatchMs(batchFeatures(item, item, item)));
    }

    @Test
    void predictBatchMsZeroCacheHits() {
        FormulaPredictor p = new FormulaPredictor("sum(computeTokens)");
        assertEquals(500, p.evaluator().predictBatchMs(batchFeatures(item(500, 0))));
    }

    @Test
    void predictBatchMsAllCached() {
        FormulaPredictor p = new FormulaPredictor("sum(computeTokens)");
        assertEquals(0, p.evaluator().predictBatchMs(batchFeatures(item(500, 500))));
    }

    @Test
    void predictBatchMsLargeBatch() {
        FormulaPredictor p = new FormulaPredictor(
                "100 + 0.5*sum(computeTokens) + 0.1*sum(hitCacheTokens) + 3*batchSize");
        List<PrefillBatchFeatures.Item> items = new ArrayList<>();
        for (int i = 0; i < 100; i++) {
            items.add(item(1000, 200));
        }
        long result = (long) p.evaluator().predictBatchMs(batchFeatures(items));
        assertTrue(result > 0, "Large batch should produce positive prediction");
    }

    // ---- power operator ----

    @Test
    void powerOperatorRightAssociative() {
        // 2^3^2 = 2^(3^2) = 2^9 = 512
        FormulaPredictor p = new FormulaPredictor("2^3^2");
        assertEquals(512, p.evaluator().estimateMs(0, 0));
    }

    // ---- functions ----

    @Test
    void sqrtFunction() {
        FormulaPredictor p = new FormulaPredictor("sqrt(100)");
        assertEquals(10, p.evaluator().estimateMs(0, 0));
    }

    @Test
    void maxFunction() {
        FormulaPredictor p = new FormulaPredictor("max(sum(computeTokens), 50)");
        assertEquals(100, p.evaluator().estimateMs(100, 0));
        assertEquals(50, p.evaluator().estimateMs(30, 0));
    }

    @Test
    void nestedFunctions() {
        FormulaPredictor p = new FormulaPredictor(
                "sqrt(pow(sum(computeTokens), 2) + pow(sum(hitCacheTokens), 2))");
        // inputTokens=7, hitCacheTokens=4, computeTokens=3, sqrt(9+16) = 5
        assertEquals(5, p.evaluator().estimateMs(7, 4));
    }

    // ---- parentheses ----

    @Test
    void parenthesesOverridePrecedence() {
        FormulaPredictor p = new FormulaPredictor("(2 + 3) * 4");
        assertEquals(20, p.evaluator().estimateMs(0, 0));
    }

    // ---- learn (interface stub) ----

    @Test
    @DisplayName("learn method accepts immutable batch features without error")
    void learnAcceptsBatchInfo() {
        FormulaPredictor p = new FormulaPredictor("100");
        PrefillBatchFeatures features = batchFeatures(
                item(100, 20), item(200, 50));
        p.learn(features, 150, 300);
    }

    // ---- param() learnable parameters ----

    @Test
    @DisplayName("param() basic parsing returns initial value")
    void paramBasicParsing() {
        FormulaPredictor p = new FormulaPredictor("param(w0, 100)");
        assertEquals(100, p.evaluator().estimateMs(0, 0));
        assertEquals(100, p.evaluator().estimateMs(500, 200));
    }

    @Test
    @DisplayName("param() in expression with variables")
    void paramInExpression() {
        // param(w0, 10) + param(w1, 0.5) * computeTokens
        // inputTokens=100, hitCache=0, computeTokens=100 → 10 + 0.5*100 = 60
        FormulaPredictor p = new FormulaPredictor("param(w0, 10) + param(w1, 0.5) * computeTokens");
        assertEquals(60, p.evaluator().estimateMs(100, 0));
    }

    @Test
    @DisplayName("same parameter name reused across formula shares one ParameterNode")
    void paramSameNameReused() {
        // param(w0, 1) * computeTokens + param(w0, 1) * hitCacheTokens
        // inputTokens=100, hitCache=50, computeTokens=50 → 1*50 + 1*50 = 100
        FormulaPredictor p = new FormulaPredictor("param(w0, 1) * computeTokens + param(w0, 1) * hitCacheTokens");
        assertEquals(100, p.evaluator().estimateMs(100, 50));
    }

    @Test
    @DisplayName("param() works in batch mode with sum()")
    void paramInBatchMode() {
        // param(w0, 10) + param(w1, 0.5) * sum(computeTokens)
        // item1: (500,200) → computeTokens=300
        // item2: (300,100) → computeTokens=200
        // sum(computeTokens) = 500 → 10 + 0.5*500 = 260
        FormulaPredictor p = new FormulaPredictor("param(w0, 10) + param(w1, 0.5) * sum(computeTokens)");
        PrefillBatchFeatures features = batchFeatures(
                item(500, 200), item(300, 100));
        assertEquals(260, p.evaluator().predictBatchMs(features));
    }

    @Test
    @DisplayName("param() initial value can be an expression")
    void paramInitialValueExpression() {
        // param(w0, 2+3) * computeTokens → 5 * 100 = 500
        FormulaPredictor p = new FormulaPredictor("param(w0, 2+3) * computeTokens");
        assertEquals(500, p.evaluator().estimateMs(100, 0));
    }

    // ---- concurrency ----

    @Test
    @DisplayName("thread-local formula cursors isolate concurrent single and batch evaluations")
    void formulaCursorsAreConcurrentSafe() throws Exception {
        FormulaPredictor single = new FormulaPredictor(
                "2*computeTokens + 3*hitCacheTokens + batchSize");
        FormulaPredictor batch = new FormulaPredictor(
                "batchSize + sum(2*computeTokens + 3*hitCacheTokens)");
        int threadCount = 8;
        ExecutorService executor = Executors.newFixedThreadPool(threadCount);
        CountDownLatch start = new CountDownLatch(1);
        List<Future<?>> futures = new ArrayList<>();
        try {
            for (int thread = 0; thread < threadCount; thread++) {
                long seqLen = 1_000L + thread * 37L;
                long hitCache = 100L + thread * 11L;
                PrefillBatchFeatures features = batchFeatures(
                        item(seqLen, hitCache),
                        item(seqLen / 2L, hitCache / 2L));
                long expectedSingle = 2L * (seqLen - hitCache) + 3L * hitCache + 1L;
                long batchExpected = features.batchSize();
                for (PrefillBatchFeatures.Item item : features.items()) {
                    batchExpected += 2L * (item.seqLen() - item.hitCache())
                            + 3L * item.hitCache();
                }
                long expectedBatch = batchExpected;
                futures.add(executor.submit(() -> {
                    start.await();
                    for (int iteration = 0; iteration < 2_000; iteration++) {
                        assertEquals(expectedSingle, single.evaluator().estimateMs(seqLen, hitCache));
                        assertEquals(expectedBatch, (long) batch.evaluator().predictBatchMs(features));
                    }
                    return null;
                }));
            }
            start.countDown();
            for (Future<?> future : futures) {
                future.get(10, TimeUnit.SECONDS);
            }
        } finally {
            executor.shutdownNow();
            assertTrue(executor.awaitTermination(5, TimeUnit.SECONDS));
        }
    }

    @Test
    @DisplayName("failed evaluation does not poison the reusable cursor")
    void failedEvaluationDoesNotPoisonReusableCursor() {
        PrefillTimeFormula direct = PrefillTimeFormula.parse("computeTokens");

        assertThrows(NullPointerException.class, () -> direct.evaluate(null, List.of()));

        FormulaPredictor first = new FormulaPredictor(
                "sum(computeTokens) + 2*sum(hitCacheTokens)");
        FormulaPredictor second = new FormulaPredictor(
                "sum(max(computeTokens - 10, 0))");
        assertEquals(500, first.evaluator().estimateMs(400, 100));
        assertEquals(290, second.evaluator().estimateMs(400, 100));
        assertEquals(500, first.evaluator().estimateMs(400, 100));
    }

    // ---- helpers ----

    private static PrefillBatchFeatures batchFeatures(
            PrefillBatchFeatures.Item... items) {
        return new PrefillBatchFeatures(List.of(items));
    }

    private static PrefillBatchFeatures batchFeatures(
            List<PrefillBatchFeatures.Item> items) {
        return new PrefillBatchFeatures(items);
    }

    private static PrefillBatchFeatures.Item item(long seqLen, long hitCache) {
        return new PrefillBatchFeatures.Item(seqLen, hitCache);
    }
}
