package org.flexlb.balance.strategy;

import org.flexlb.balance.prediction.FormulaPredictor;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class PrefillAdaptiveBudgetTest {
    @ParameterizedTest @CsvSource({"1,100", "7,100", "70,2000", "700,2000"})
    void tokenBudgetTracksWorkerSpeedAndWaitTarget(int tokensPerMs, long budgetMs) {
        var evaluator = new FormulaPredictor("sum(computeTokens) / " + tokensPerMs).evaluator();
        long budget = PrefillAdaptiveBudget.tokensFor(evaluator, budgetMs);
        assertTrue(evaluator.estimateMs(budget, 0) <= budgetMs);
        assertTrue(evaluator.estimateMs(budget + 1, 0) > budgetMs);
        assertTrue(budget >= budgetMs * tokensPerMs);
    }

    @Test void coldStartAlwaysAllowsOneRequestAndSearchIsBounded() {
        assertEquals(1, PrefillAdaptiveBudget.tokensFor(new FormulaPredictor("100000 + sum(computeTokens)").evaluator(), 1));
        assertTrue(PrefillAdaptiveBudget.tokensFor(new FormulaPredictor("0").evaluator(), 2_000) <= 1L << 30);
    }

    @Test void fixedExecutionOverheadDoesNotConsumeTheComputeCollectionWindow() {
        var plain = new FormulaPredictor("sum(computeTokens) / 70").evaluator();
        var overhead = new FormulaPredictor("350 + sum(computeTokens) / 70").evaluator();
        assertEquals(PrefillAdaptiveBudget.tokensFor(plain, 200), PrefillAdaptiveBudget.tokensFor(overhead, 200));
        assertTrue(PrefillAdaptiveBudget.tokensFor(overhead, 200) >= 14_000);
    }

    @Test void defaultLearningModelCanCollectMultipleWarmRequestsAtShortSlo() {
        var evaluator = new org.flexlb.balance.prediction.LearningPredictor().evaluator();
        assertTrue(evaluator.estimateMs(1, 0) > 200, "reproduce the cold-start intercept above the service window");
        long tokens = PrefillAdaptiveBudget.tokensFor(evaluator, 200);
        assertTrue(tokens >= 2 * 1024, "warm requests must be able to share a collection window");
        assertTrue(evaluator.estimateMs(tokens, 0) - evaluator.estimateMs(0, 0) <= 200);
        assertTrue(evaluator.estimateMs(tokens + 1, 0) - evaluator.estimateMs(0, 0) > 200);
    }

    @Test void zeroWindowIsRejected() {
        assertThrows(IllegalArgumentException.class, () -> PrefillAdaptiveBudget.tokensFor(
                new FormulaPredictor("sum(computeTokens)").evaluator(), 0));
    }
}
