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
    @DisplayName("default weights produce zero estimateMs")
    void defaultWeightsEstimateMs() {
        // Exponential model: default weights all 0
        // output = w6/coff1 + w7/coff2 * exp(0/coff3) = 0 + 0 * 1.0 = 0
        LearningPredictor p = new LearningPredictor();
        assertEquals(0, p.estimateMs(1000, 200));
    }

    @Test
    @DisplayName("estimateMs with zero tokens")
    void estimateMsZeroTokens() {
        LearningPredictor p = new LearningPredictor();
        assertEquals(0, p.estimateMs(0, 0));
    }

    @Test
    @DisplayName("estimateMs bounds hitTokens to totalTokens")
    void estimateMsHitTokensBounded() {
        LearningPredictor p = new LearningPredictor();
        assertEquals(0, p.estimateMs(100, 500));
    }

    @Test
    @DisplayName("predictBatchMs aggregates correctly")
    void predictBatchMsAggregation() {
        LearningPredictor p = new LearningPredictor();
        BatchItem item1 = batchItem(500, 200);
        BatchItem item2 = batchItem(300, 100);
        // Default weights all 0 → prediction 0
        assertEquals(0, p.predictBatchMs(List.of(item1, item2)));
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
        List<BatchItem> batchItems = List.of(
                batchItem(500, 200),
                batchItem(300, 100),
                batchItem(1000, 500));
        for (int i = 0; i < 400; i++) {
            final int round = i;
            assertDoesNotThrow(() -> p.learn(batchItems, 0, 100L + round));
        }
        System.out.println(p.formulaString());
    }

    @Test
    @DisplayName("getParameter returns weight values")
    void getParameterValues() {
        LearningPredictor p = new LearningPredictor();
        // Exponential model: 8 weights, all default to 0
        assertEquals(0.0, p.getParameter("w0"));
        assertEquals(0.0, p.getParameter("w1"));
        assertEquals(0.0, p.getParameter("w2"));
        assertEquals(0.0, p.getParameter("w3"));
        assertEquals(0.0, p.getParameter("w4"));
        assertEquals(0.0, p.getParameter("w5"));
        assertEquals(0.0, p.getParameter("w6"));
        assertEquals(0.0, p.getParameter("w7"));
    }

    @Test
    @DisplayName("setParameter updates weight")
    void setParameterUpdatesWeight() {
        LearningPredictor p = new LearningPredictor();
        p.setParameter("w6", 300.0);
        p.setParameter("w7", 360.0);
        assertEquals(300.0, p.getParameter("w6"));
        assertEquals(360.0, p.getParameter("w7"));
        // w6=300, w7=360, other weights=0:
        // sum=0, linearExp=exp(0/1700)=1.0
        // output = 300/0.55 + 360/1.2 * 1.0 = 545.45 + 300 = 845.45 → 845
        assertEquals(845, p.estimateMs(1000, 200));
    }

    @Test
    @DisplayName("getParameter on unknown name throws")
    void unknownParameterThrows() {
        LearningPredictor p = new LearningPredictor();
        assertThrows(IllegalArgumentException.class, () -> p.getParameter("unknown"));
    }

    @Test
    @DisplayName("parameterNames returns all six weights")
    void parameterNames() {
        LearningPredictor p = new LearningPredictor();
        assertEquals(Set.of("w0", "w1", "w2", "w3", "w4", "w5"), p.parameterNames());
    }

    @Test
    @DisplayName("getParameters returns all weights as map")
    void getParametersMap() {
        LearningPredictor p = new LearningPredictor();
        Map<String, Double> params = p.getParameters();
        // Exponential model has 8 weights (w0-w7), all default to 0
        assertEquals(8, params.size());
        assertEquals(0.0, params.get("w0"));
        assertEquals(0.0, params.get("w1"));
        assertEquals(0.0, params.get("w2"));
        assertEquals(0.0, params.get("w3"));
        assertEquals(0.0, params.get("w4"));
        assertEquals(0.0, params.get("w5"));
        assertEquals(0.0, params.get("w6"));
        assertEquals(0.0, params.get("w7"));
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
        p.setParameter("w0", 300.0);
        p.setParameter("w1", -5.0);
        String desc = p.formulaString();
        assertTrue(desc.contains("w0"));
        assertTrue(desc.contains("w1"));
        assertTrue(desc.contains("w3"));
        assertTrue(desc.contains("300.0"));
        assertTrue(desc.contains("-5.0"));
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
