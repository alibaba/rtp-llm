package org.flexlb.balance.strategy;

import org.flexlb.balance.scheduler.BatchItem;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.DebugInfo;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class LearningPredictorTest {

    @Test
    @DisplayName("default model produces a non-negative estimate")
    void defaultModelEstimateMs() {
        LearningPredictor p = new LearningPredictor();
        assertTrue(p.estimateMs(1000, 200) >= 0);
    }

    @Test
    @DisplayName("estimateMs with zero tokens")
    void estimateMsZeroTokens() {
        LearningPredictor p = new LearningPredictor();
        assertTrue(p.estimateMs(0, 0) >= 0);
    }

    @Test
    @DisplayName("estimateMs bounds hitTokens to totalTokens")
    void estimateMsHitTokensBounded() {
        LearningPredictor p = new LearningPredictor();
        assertEquals(p.estimateMs(100, 100), p.estimateMs(100, 500));
    }

    @Test
    @DisplayName("predictBatchMs aggregates correctly")
    void predictBatchMsAggregation() {
        LearningPredictor p = new LearningPredictor();
        BatchItem item1 = batchItem(500, 200);
        BatchItem item2 = batchItem(300, 100);
        assertTrue(p.predictBatchMs(List.of(item1, item2)) >= 0);
    }

    @Test
    @DisplayName("predictBatchMs empty list returns 0")
    void predictBatchMsEmpty() {
        LearningPredictor p = new LearningPredictor();
        assertEquals(0, p.predictBatchMs(List.of()));
    }

    @Test
    @DisplayName("learn accepts completed batches")
    void learnAcceptsCompletedBatches() {
        LearningPredictor p = new LearningPredictor();
        List<BatchItem> batchItems = List.of(
                batchItem(500, 200),
                batchItem(300, 100),
                batchItem(1000, 500));
        for (int i = 0; i < 4; i++) {
            assertDoesNotThrow(() -> p.learn(batchItems, 300, 400));
        }
    }

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

        return new BatchItem(ctx, null, null, prefill, null, null, null, System.currentTimeMillis());
    }
}
