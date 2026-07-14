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
        // weights = {300, -5, 0, 5, 0, 0}
        // seq=1000, hit=200, compute=(1000-200)/1024=0.78125
        // y = 300*1 + (-5)*1 + 0*reuse + 5*0.78125 + 0 + 0 = 295 + 3.90625 = 298.90625
        LearningPredictor p = new LearningPredictor();
        assertEquals(298, p.estimateMs(1000, 200));
    }

    @Test
    @DisplayName("estimateMs with zero tokens")
    void estimateMsZeroTokens() {
        LearningPredictor p = new LearningPredictor();
        // seq=0, hit=0, compute=0
        // y = 300*1 + (-5)*1 + 0 + 5*0 + 0 + 0 = 295
        assertEquals(295, p.estimateMs(0, 0));
    }

    @Test
    @DisplayName("estimateMs bounds hitTokens to totalTokens")
    void estimateMsHitTokensBounded() {
        LearningPredictor p = new LearningPredictor();
        // hit=min(500,100)=100, compute=(100-100)/1024=0
        // y = 300*1 + (-5)*1 + 0 + 5*0 + 0 + 0 = 295
        assertEquals(295, p.estimateMs(100, 500));
    }

    @Test
    @DisplayName("predictBatchMs aggregates correctly")
    void predictBatchMsAggregation() {
        LearningPredictor p = new LearningPredictor();
        // item1: seq=500, hit=200 → reuse=200/1024, compute=300/1024
        // item2: seq=300, hit=100 → reuse=100/1024, compute=200/1024
        // y = 300*1 + (-5)*2 + 0*sumReuse + 5*sumCompute + 0 + 0
        //   = 300 - 10 + 5*(500/1024) = 290 + 2.4414 = 292.4414
        BatchItem item1 = batchItem(500, 200);
        BatchItem item2 = batchItem(300, 100);
        assertEquals(292, p.predictBatchMs(List.of(item1, item2)));
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
        assertEquals(300.0, p.getParameter("w0"));
        assertEquals(-5.0, p.getParameter("w1"));
        assertEquals(0.0, p.getParameter("w2"));
        assertEquals(5.0, p.getParameter("w3"));
        assertEquals(0.0, p.getParameter("w4"));
        assertEquals(0.0, p.getParameter("w5"));
    }

    @Test
    @DisplayName("setParameter updates weight")
    void setParameterUpdatesWeight() {
        LearningPredictor p = new LearningPredictor();
        p.setParameter("w1", 1.0);
        assertEquals(1.0, p.getParameter("w1"));
        // weights = {300, 1.0, 0, 5, 0, 0}
        // seq=1000, hit=200, compute=800/1024=0.78125
        // y = 300*1 + 1.0*1 + 0 + 5*0.78125 + 0 + 0 = 304.90625
        assertEquals(304, p.estimateMs(1000, 200));
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
        assertEquals(300.0, params.get("w0"));
        assertEquals(-5.0, params.get("w1"));
        assertEquals(0.0, params.get("w2"));
        assertEquals(5.0, params.get("w3"));
        assertEquals(0.0, params.get("w4"));
        assertEquals(0.0, params.get("w5"));
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
