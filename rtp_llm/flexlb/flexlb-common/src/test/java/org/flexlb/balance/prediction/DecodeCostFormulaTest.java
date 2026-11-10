package org.flexlb.balance.prediction;

import org.junit.jupiter.api.Test;

import java.util.concurrent.CyclicBarrier;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertSame;

class DecodeCostFormulaTest {
    @Test
    void evaluatesAllVariablesAndZeroCapacity() {
        var evaluator =
                DecodeCostFormula.parse(
                                "running_size + max_running_size + kvcache_used + kvcache_capacity"
                                        + " + kvcache_used_ratio * 100")
                        .evaluator();
        assertEquals(162.0, evaluator.evaluate(2, 10, 25, 100));
        assertEquals(37.0, evaluator.evaluate(2, 10, 25, 0));
    }

    @Test
    void changingTheFormulaDoesNotOverwriteAnEarlierEvaluator() {
        var formula = DecodeCostFormula.parse("running_size * 2");
        var first = formula.evaluator();
        assertSame(first, formula.evaluator());
        var second = DecodeCostFormula.parse("sqrt(kvcache_used)").evaluator();
        assertEquals(3.0, second.evaluate(7, 10, 9, 100));
        assertEquals(14.0, first.evaluate(7, 10, 9, 100));
        assertSame(first, formula.evaluator());
        assertEquals(14.0, formula.evaluator().evaluate(7, 10, 9, 100));
    }

    @Test
    void concurrentScoringPassesKeepIndependentVariables() throws Exception {
        var formula = DecodeCostFormula.parse("running_size + kvcache_used_ratio * 100");
        var barrier = new CyclicBarrier(2);
        try (var executor = Executors.newFixedThreadPool(2)) {
            var first =
                    executor.submit(
                            () -> {
                                var evaluator = formula.evaluator();
                                barrier.await(5, TimeUnit.SECONDS);
                                for (int i = 0; i < 10_000; i++) {
                                    assertEquals(26.0, evaluator.evaluate(1, 10, 25, 100));
                                }
                                return null;
                            });
            var second =
                    executor.submit(
                            () -> {
                                var evaluator = formula.evaluator();
                                barrier.await(5, TimeUnit.SECONDS);
                                for (int i = 0; i < 10_000; i++) {
                                    assertEquals(52.0, evaluator.evaluate(2, 10, 50, 100));
                                }
                                return null;
                            });
            first.get(10, TimeUnit.SECONDS);
            second.get(10, TimeUnit.SECONDS);
        }
    }
}
