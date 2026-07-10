package org.flexlb.balance.strategy;

import org.flexlb.balance.scheduler.BatchItem;
import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.DebugInfo;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.ServerStatus;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

class LearningPredictorTest {

    @Test
    @DisplayName("default weights produce correct estimateMs")
    void defaultWeightsEstimateMs() {
        // w0=50, w1=0.5, w2=0.3
        // totalTokens=1000, hitTokens=200 → computeTokens=800
        // 50 + 0.5*800 + 0.3*200 = 50 + 400 + 60 = 510
        LearningPredictor p = new LearningPredictor();
        assertEquals(510, p.estimateMs(1000, 200));
    }

    @Test
    @DisplayName("estimateMs with zero tokens")
    void estimateMsZeroTokens() {
        LearningPredictor p = new LearningPredictor();
        // 50 + 0 + 0 = 50
        assertEquals(50, p.estimateMs(0, 0));
    }

    @Test
    @DisplayName("estimateMs bounds hitTokens to totalTokens")
    void estimateMsHitTokensBounded() {
        LearningPredictor p = new LearningPredictor();
        // hitTokens > totalTokens → bounded to totalTokens, computeTokens=0
        // 50 + 0 + 0.3*100 = 80
        assertEquals(80, p.estimateMs(100, 500));
    }

    @Test
    @DisplayName("predictBatchMs aggregates correctly")
    void predictBatchMsAggregation() {
        LearningPredictor p = new LearningPredictor();
        // item1: (500,200) → compute=300, hit=200
        // item2: (300,100) → compute=200, hit=100
        // sumCompute=500, sumHit=300
        // 50 + 0.5*500 + 0.3*300 = 50 + 250 + 90 = 390
        BatchItem item1 = batchItem(500, 200);
        BatchItem item2 = batchItem(300, 100);
        assertEquals(390, p.predictBatchMs(List.of(item1, item2)));
    }

    @Test
    @DisplayName("predictBatchMs empty list returns 0")
    void predictBatchMsEmpty() {
        LearningPredictor p = new LearningPredictor();
        assertEquals(0, p.predictBatchMs(List.of()));
    }

    @Test
    @DisplayName("learn does not throw")
    void learnDoesNotThrow() {
        LearningPredictor p = new LearningPredictor();
        BatchItem item = batchItem(500, 200);
        assertDoesNotThrow(() -> p.learn(List.of(item), 300, 350));
    }

    @Test
    @DisplayName("getParameter returns weight values")
    void getParameterValues() {
        LearningPredictor p = new LearningPredictor();
        assertEquals(50.0, p.getParameter("w0"));
        assertEquals(0.5, p.getParameter("w1"));
        assertEquals(0.3, p.getParameter("w2"));
    }

    @Test
    @DisplayName("setParameter updates weight")
    void setParameterUpdatesWeight() {
        LearningPredictor p = new LearningPredictor();
        p.setParameter("w1", 1.0);
        assertEquals(1.0, p.getParameter("w1"));
        // 50 + 1.0*800 + 0.3*200 = 50 + 800 + 60 = 910
        assertEquals(910, p.estimateMs(1000, 200));
    }

    @Test
    @DisplayName("getParameter on unknown name throws")
    void unknownParameterThrows() {
        LearningPredictor p = new LearningPredictor();
        assertThrows(IllegalArgumentException.class, () -> p.getParameter("unknown"));
    }

    @Test
    @DisplayName("parameterNames returns all three weights")
    void parameterNames() {
        LearningPredictor p = new LearningPredictor();
        assertEquals(Set.of("w0", "w1", "w2"), p.parameterNames());
    }

    @Test
    @DisplayName("getParameters returns all weights as map")
    void getParametersMap() {
        LearningPredictor p = new LearningPredictor();
        Map<String, Double> params = p.getParameters();
        assertEquals(50.0, params.get("w0"));
        assertEquals(0.5, params.get("w1"));
        assertEquals(0.3, params.get("w2"));
    }

    @Test
    @DisplayName("hasParameters returns true")
    void hasParameters() {
        LearningPredictor p = new LearningPredictor();
        assertTrue(p.hasParameters());
    }

    @Test
    @DisplayName("formulaString contains weight values")
    void formulaStringContainsWeights() {
        LearningPredictor p = new LearningPredictor();
        String desc = p.formulaString();
        assertTrue(desc.contains("w0"));
        assertTrue(desc.contains("w1"));
        assertTrue(desc.contains("w2"));
        assertTrue(desc.contains("computeTokens"));
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

        return new BatchItem(ctx, null, null, prefill, null, null, null, 0, System.currentTimeMillis());
    }
}
