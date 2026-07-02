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
                new PrefillTimePredictor("sum_c +"));
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
        // "2*c" → time = 2*c
        PrefillTimePredictor p = new PrefillTimePredictor("2*c");
        // c = 1500-500 = 1000 → 2000
        assertEquals(2000, p.estimateMs(1500, 500));
        // c = 300-0 = 300 → 600
        assertEquals(600, p.estimateMs(300, 0));
    }

    @Test
    void estimateMsQuadraticInComputeTokens() {
        // "0.1*c^2" → time = 0.1 * c²
        PrefillTimePredictor p = new PrefillTimePredictor("0.1*c^2");
        // c=100 → 0.1*10000 = 1000
        assertEquals(1000, p.estimateMs(100, 0));
    }

    @Test
    void estimateMsInteractionTerm() {
        // "0.5*c*p" → time = 0.5 * c * hitTokens
        PrefillTimePredictor p = new PrefillTimePredictor("0.5*c*p");
        // c=200, hit=400 → 0.5*200*400 = 40000
        assertEquals(40000, p.estimateMs(600, 400));
    }

    @Test
    void estimateMsSumVariablesInSingleMode() {
        // In single-request mode, sum_c == c and sum_p == p
        PrefillTimePredictor p = new PrefillTimePredictor("sum_c + 0.3*sum_p");
        // c=300, p=200 → 300 + 0.3*200 = 360
        assertEquals(360, p.estimateMs(500, 200));
    }

    @Test
    void estimateMsFullFormula() {
        // "10 + 0.1*c + 0.01*c^2 + 0.001*c*p + 0.5*p + 5"
        // total=500, hit=200 → c=300
        // = 10 + 30 + 900 + 60 + 100 + 5 = 1105
        PrefillTimePredictor p = new PrefillTimePredictor("10 + 0.1*sum_c + 0.01*sum_c2 + 0.001*sum_cp + 0.5*sum_p + 5*n");
        assertEquals(1105, p.estimateMs(500, 200));
    }

    @Test
    void estimateMsHitTokensCannotExceedTotal() {
        // c = max(0, total-hit), so if hit > total, c = 0
        PrefillTimePredictor p = new PrefillTimePredictor("2*c");
        assertEquals(0, p.estimateMs(100, 500));
    }

    @Test
    void estimateMsLargeValuesNoOverflow() {
        PrefillTimePredictor p = new PrefillTimePredictor("100 + sum_c + 0.001*sum_c2 + 0.0001*sum_cp + 0.5*sum_p + 10*n");
        long result = p.estimateMs(100_000, 50_000);
        assertTrue(result >= 0, "Should not overflow or produce negative values");
    }

    // ---- predictBatchMs ----

    @Test
    void predictBatchMsEmptyListReturnsZero() {
        PrefillTimePredictor p = new PrefillTimePredictor("10 + sum_c + 5*n");
        assertEquals(0, p.predictBatchMs(List.of()));
    }

    @Test
    void predictBatchMsSingleItemMatchesEstimateMs() {
        PrefillTimePredictor p = new PrefillTimePredictor("10 + 0.1*sum_c + 0.01*sum_c2 + 0.001*sum_cp + 0.5*sum_p + 5*n");
        long single = p.estimateMs(500, 200);

        BatchItem item = batchItem(500, 200);
        long batch = p.predictBatchMs(List.of(item));

        assertEquals(single, batch);
    }

    @Test
    void predictBatchMsMultipleItems() {
        // "10 + 0.1*sum_c + 0.01*sum_c2 + 0.001*sum_cp + 0.5*sum_p + 5*n"
        // item1: seq=500, hit=200 → c=300, c²=90000, cp=60000
        // item2: seq=300, hit=100 → c=200, c²=40000, cp=20000
        // sum_c=500, sum_c2=130000, sum_cp=80000, sum_p=300, n=2
        // = 10 + 0.1*500 + 0.01*130000 + 0.001*80000 + 0.5*300 + 5*2
        // = 10 + 50 + 1300 + 80 + 150 + 10 = 1600
        PrefillTimePredictor p = new PrefillTimePredictor("10 + 0.1*sum_c + 0.01*sum_c2 + 0.001*sum_cp + 0.5*sum_p + 5*n");

        BatchItem item1 = batchItem(500, 200);
        BatchItem item2 = batchItem(300, 100);
        long result = p.predictBatchMs(List.of(item1, item2));

        assertEquals(1600, result);
    }

    @Test
    void predictBatchMsBatchSizeAffectsResult() {
        // "10*n" → time depends only on batch size
        PrefillTimePredictor p = new PrefillTimePredictor("10*n");

        BatchItem item = batchItem(100, 0);
        assertEquals(10, p.predictBatchMs(List.of(item)));
        assertEquals(20, p.predictBatchMs(List.of(item, item)));
        assertEquals(30, p.predictBatchMs(List.of(item, item, item)));
    }

    @Test
    void predictBatchMsZeroCacheHits() {
        // "sum_c" → time = sum of compute tokens
        PrefillTimePredictor p = new PrefillTimePredictor("sum_c");
        BatchItem item = batchItem(500, 0);
        assertEquals(500, p.predictBatchMs(List.of(item)));
    }

    @Test
    void predictBatchMsAllCached() {
        // "sum_c" → all cached → c=0 for each item → 0
        PrefillTimePredictor p = new PrefillTimePredictor("sum_c");
        BatchItem item = batchItem(500, 500);
        assertEquals(0, p.predictBatchMs(List.of(item)));
    }

    @Test
    void predictBatchMsLargeBatch() {
        PrefillTimePredictor p = new PrefillTimePredictor("100 + 0.5*sum_c + 0.1*sum_p + 3*n");
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
        PrefillTimePredictor p = new PrefillTimePredictor("max(sum_c, 50)");
        assertEquals(100, p.estimateMs(100, 0));  // c=100 → max(100,50)=100
        assertEquals(50, p.estimateMs(30, 0));    // c=30  → max(30,50)=50
    }

    @Test
    void nestedFunctions() {
        PrefillTimePredictor p = new PrefillTimePredictor("sqrt(pow(sum_c, 2) + pow(sum_p, 2))");
        // total=7, hit=4 → c=3, p=4 → sqrt(9+16) = 5
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
