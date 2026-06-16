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
import static org.junit.jupiter.api.Assertions.assertTrue;

class PrefillTimePredictorTest {

    // ---- estimateMs (single request) ----

    @Test
    void estimateMsZeroCoefficientsReturnsZero() {
        PrefillTimePredictor predictor = new PrefillTimePredictor(0, 0, 0, 0, 0, 0);
        assertEquals(0, predictor.estimateMs(1000, 0));
        assertEquals(0, predictor.estimateMs(1000, 500));
    }

    @Test
    void estimateMsConstantTermOnly() {
        // α₀ = 50, others 0 → always 50
        PrefillTimePredictor predictor = new PrefillTimePredictor(50, 0, 0, 0, 0, 0);
        assertEquals(50, predictor.estimateMs(100, 0));
        assertEquals(50, predictor.estimateMs(0, 0));
    }

    @Test
    void estimateMsLinearInComputeTokens() {
        // α₁ = 2, others 0 → time = 2 * c
        // c = totalTokens - hitTokens
        PrefillTimePredictor predictor = new PrefillTimePredictor(0, 2, 0, 0, 0, 0);
        assertEquals(2000, predictor.estimateMs(1500, 500));  // c=1000 → 2*1000
        assertEquals(600, predictor.estimateMs(300, 0));      // c=300 → 2*300
    }

    @Test
    void estimateMsQuadraticInComputeTokens() {
        // α₂ = 0.1, others 0 → time = 0.1 * c²
        PrefillTimePredictor predictor = new PrefillTimePredictor(0, 0, 0.1, 0, 0, 0);
        assertEquals(1000, predictor.estimateMs(100, 0));  // c=100 → 0.1*10000
    }

    @Test
    void estimateMsInteractionTerm() {
        // α₃ = 0.5, others 0 → time = 0.5 * c * hitTokens
        PrefillTimePredictor predictor = new PrefillTimePredictor(0, 0, 0, 0.5, 0, 0);
        assertEquals(40000, predictor.estimateMs(600, 400)); // c=200, hit=400 → 0.5*200*400
    }

    @Test
    void estimateMsLinearInHitTokens() {
        // α₄ = 3, others 0 → time = 3 * hitTokens
        PrefillTimePredictor predictor = new PrefillTimePredictor(0, 0, 0, 0, 3, 0);
        assertEquals(300, predictor.estimateMs(500, 100)); // hitTokens never negative
    }

    @Test
    void estimateMsBatchSizeTerm() {
        // α₅ = 10, others 0 → time = 10 (bs=1 in single mode)
        PrefillTimePredictor predictor = new PrefillTimePredictor(0, 0, 0, 0, 0, 10);
        assertEquals(10, predictor.estimateMs(1000, 0));
    }

    @Test
    void estimateMsFullFormula() {
        // α₀=10, α₁=0.1, α₂=0.01, α₃=0.001, α₄=0.5, α₅=5
        // total=500, hit=200 → c=300
        // result = 10 + 0.1*300 + 0.01*90000 + 0.001*300*200 + 0.5*200 + 5
        // = 10 + 30 + 900 + 60 + 100 + 5 = 1105
        PrefillTimePredictor predictor = new PrefillTimePredictor(10, 0.1, 0.01, 0.001, 0.5, 5);
        assertEquals(1105, predictor.estimateMs(500, 200));
    }

    @Test
    void estimateMsHitTokensCannotExceedTotal() {
        // c = max(0, total - hit), so if hit > total, c = 0
        PrefillTimePredictor predictor = new PrefillTimePredictor(0, 2, 0, 0, 0, 0);
        assertEquals(0, predictor.estimateMs(100, 500)); // c = max(0, 100-500) = 0
    }

    @Test
    void estimateMsLargeValuesNoOverflow() {
        PrefillTimePredictor predictor = new PrefillTimePredictor(100, 1, 0.001, 0.0001, 0.5, 10);
        long result = predictor.estimateMs(100_000, 50_000);
        assertTrue(result >= 0, "Should not overflow or produce negative values");
    }

    // ---- predictBatchMs ----

    @Test
    void predictBatchMsEmptyListReturnsZero() {
        PrefillTimePredictor predictor = new PrefillTimePredictor(10, 1, 0, 0, 0, 5);
        assertEquals(0, predictor.predictBatchMs(List.of()));
    }

    @Test
    void predictBatchMsSingleItemMatchesEstimateMs() {
        // For a single item, predictBatchMs should be close to estimateMs
        // (α₅ contributes per-item vs per-call, so α₅ * bs differs: estimateMs uses α₅*1,
        // predictBatchMs uses α₅*bs=α₅*1 — same for single item)
        PrefillTimePredictor predictor = new PrefillTimePredictor(10, 0.1, 0.01, 0.001, 0.5, 5);
        long single = predictor.estimateMs(500, 200);

        BatchItem item = batchItem(500, 200);
        long batch = predictor.predictBatchMs(List.of(item));

        assertEquals(single, batch);
    }

    @Test
    void predictBatchMsMultipleItems() {
        // α₀=10, α₁=0.1, α₂=0.01, α₃=0.001, α₄=0.5, α₅=5
        // item1: seq=500, hit=200 → c=300
        // item2: seq=300, hit=100 → c=200
        // Σc=500, Σc²*p need to be recomputed
        // Let me compute: for item1: a2*c²=0.01*90000=900, a3*c*p=0.001*300*200=60
        // for item2: a2*c²=0.01*40000=400, a3*c*p=0.001*200*100=20
        // Σquadratic = 900+60+400+20 = 1380
        // Σp = 200+100 = 300
        // result = 10 + 0.1*500 + 1380 + 0.5*300 + 5*2
        // = 10 + 50 + 1380 + 150 + 10 = 1600
        PrefillTimePredictor predictor = new PrefillTimePredictor(10, 0.1, 0.01, 0.001, 0.5, 5);

        BatchItem item1 = batchItem(500, 200);
        BatchItem item2 = batchItem(300, 100);
        long result = predictor.predictBatchMs(List.of(item1, item2));

        assertEquals(1600, result);
    }

    @Test
    void predictBatchMsBatchSizeAffectsResult() {
        // α₅=10, others 0 → time depends only on batch size
        PrefillTimePredictor predictor = new PrefillTimePredictor(0, 0, 0, 0, 0, 10);

        BatchItem item = batchItem(100, 0);
        assertEquals(10, predictor.predictBatchMs(List.of(item)));
        assertEquals(20, predictor.predictBatchMs(List.of(item, item)));
        assertEquals(30, predictor.predictBatchMs(List.of(item, item, item)));
    }

    @Test
    void predictBatchMsZeroCacheHits() {
        PrefillTimePredictor predictor = new PrefillTimePredictor(0, 1, 0, 0, 0, 0);
        // c = seqLen - 0 = seqLen, so time = seqLen
        BatchItem item = batchItem(500, 0);
        assertEquals(500, predictor.predictBatchMs(List.of(item)));
    }

    @Test
    void predictBatchMsAllCached() {
        // α₁ = 1 (linear in compute tokens)
        PrefillTimePredictor predictor = new PrefillTimePredictor(0, 1, 0, 0, 0, 0);
        // seq=500, hit=500 → c=0
        BatchItem item = batchItem(500, 500);
        assertEquals(0, predictor.predictBatchMs(List.of(item)));
    }

    @Test
    void predictBatchMsLargeBatch() {
        PrefillTimePredictor predictor = new PrefillTimePredictor(100, 0.5, 0, 0, 0.1, 3);
        List<BatchItem> items = new ArrayList<>();
        for (int i = 0; i < 100; i++) {
            items.add(batchItem(1000, 200));
        }
        long result = predictor.predictBatchMs(items);
        assertTrue(result > 0, "Large batch should produce positive prediction");
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
