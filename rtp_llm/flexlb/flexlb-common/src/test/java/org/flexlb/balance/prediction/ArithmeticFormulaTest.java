package org.flexlb.balance.prediction;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.AbstractList;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.function.DoubleBinaryOperator;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ArithmeticFormulaTest {

    @Test
    void evaluatesScalarVariablesWithCallerDefinedIndices() {
        ArithmeticFormula formula = ArithmeticFormula.parse(
                "running_size / max_running_size + cache_ratio^2",
                Map.of("running_size", 2, "max_running_size", 0, "cache_ratio", 1));

        assertEquals(0.75, formula.evaluateAsDouble(new double[]{8, 0.5, 4}, null));
        assertTrue(formula.referencesVariable("running_size"));
        assertFalse(formula.referencesVariable("running"));
    }

    @Test
    void preservesFunctionsScientificNotationAndOperatorPrecedence() {
        ArithmeticFormula functions = ArithmeticFormula.parse(
                "sqrt(9) + log(exp(2)) + abs(-4) + min(7,5) + max(2,3) + pow(2,3) + 1e-3*1000",
                Map.of());
        assertEquals(26.0, functions.evaluateAsDouble(new double[0], null), 1e-12);
        assertEquals(512.0, ArithmeticFormula.parse("2^3^2", Map.of())
                .evaluateAsDouble(new double[0], null));
        // Keep the established prefill grammar: unary signs bind before exponentiation.
        assertEquals(4.0, ArithmeticFormula.parse("-2^2", Map.of())
                .evaluateAsDouble(new double[0], null));
    }

    @Test
    void recordsVariablesInParameterInitializersButNotParameterNames() {
        ArithmeticFormula formula = ArithmeticFormula.parse(
                "param(weight, signal + 2) + param(weight, signal + 2)", Map.of("signal", 3));

        assertTrue(formula.referencesVariable("signal"));
        assertFalse(formula.referencesVariable("weight"));
        assertEquals(4.0, formula.evaluateAsDouble(new double[]{0, 0, 0, 99}, null),
                "parameter initial values retain their parse-time zero-variable bindings");
    }

    @Test
    void rejectsInconsistentParameterDefaults() {
        assertThrows(IllegalArgumentException.class,
                () -> ArithmeticFormula.parse("param(weight, 1) + param(weight, 2)", Map.of()));
    }

    @ParameterizedTest
    @ValueSource(strings = {"", "x +", "x y", "unknown + 1", "sqrt()", "max(1)",
            "unknown(1)", "param(weight, unknown)", "param(weight, 1, 2)", "1.2.3"})
    void rejectsMalformedExpressionsAndUnknownNames(String expression) {
        assertThrows(IllegalArgumentException.class,
                () -> ArithmeticFormula.parse(expression, Map.of("x", 0)));
    }

    @ParameterizedTest
    @ValueSource(strings = {"sum(x)", "param(weight, sum(x))", "max(1, sum(x))"})
    void scalarModeRejectsBatchAggregates(String expression) {
        assertThrows(IllegalArgumentException.class,
                () -> ArithmeticFormula.parse(expression, Map.of("x", 0)));
    }

    @Test
    void aggregateModeRetainsPerItemAndBatchScopes() {
        ArithmeticFormula formula = ArithmeticFormula.parse(
                "batchSize + sum(x^2) + x", Map.of("x", 0, "batchSize", 1), Set.of("batchSize"), true);

        assertEquals(15.0, formula.evaluateAsDouble(new double[]{3, 2},
                List.of(new double[]{1, 0}, new double[]{3, 0})));
        assertTrue(formula.referencesVariable("x"));
        assertTrue(formula.referencesVariable("batchSize"));
    }

    @ParameterizedTest
    @ValueSource(strings = {"sum(batchSize)", "sum(param(weight, batchSize))", "sum(sum(batchSize))"})
    void aggregateModeRejectsExcludedVariablesThroughoutTheAggregate(String expression) {
        assertThrows(IllegalArgumentException.class,
                () -> ArithmeticFormula.parse(expression, Map.of("batchSize", 0), Set.of("batchSize"), true));
    }

    @Test
    void aggregateFallbackRetainsSingleItemAndNestedSumBehavior() {
        ArithmeticFormula formula = ArithmeticFormula.parse(
                "sum(sum(x))", Map.of("x", 0), Set.of(), true);

        assertEquals(7.0, formula.evaluateAsDouble(new double[]{7}, null));
        assertEquals(7.0, formula.evaluateAsDouble(new double[]{7}, List.of()));
        assertEquals(5.0, formula.evaluateAsDouble(new double[]{7},
                List.of(new double[]{2}, new double[]{3})));
    }

    @Test
    void distinctAndRepeatedAggregatesShareOneTraversal() {
        ArithmeticFormula formula = aggregateFormula("sum(max(x, 2)) + sum(max((x), 2)) + sum(x)");
        List<double[]> values = List.of(new double[]{1}, new double[]{3});
        int[] reads = {0};
        List<double[]> counted = new AbstractList<>() {
            @Override
            public double[] get(int index) {
                reads[0]++;
                return values.get(index);
            }

            @Override
            public int size() {
                return values.size();
            }
        };

        assertEquals(14.0, formula.evaluateAsDouble(new double[]{0}, counted));
        assertEquals(2, reads[0], "all aggregates share one traversal of the two items");
    }

    @Test
    void fusedAggregatesKeepIndependentFloatingPointAccumulationOrder() {
        ArithmeticFormula formula = ArithmeticFormula.parse("sum(x) + sum(y) + sum(x*y)",
                Map.of("x", 0, "y", 1), Set.of(), true);
        List<double[]> items = new LinkedList<>();
        double sumX = 0, sumY = 0, sumProduct = 0;
        for (double[] item : List.of(new double[]{1e16, 1}, new double[]{1, -3},
                new double[]{-1e16, 1}, new double[]{0.25, 0.125}, new double[]{-0.0, 0.0})) {
            items.add(item);
            sumX += item[0];
            sumY += item[1];
            sumProduct += item[0]*item[1];
            assertBitsEqual(sumX + sumY + sumProduct, formula.evaluateAsDouble(new double[0], items));
        }
    }

    @Test
    void nestedSumsKeepTheirItemScopeWhenOuterAggregatesAreShared() {
        ArithmeticFormula formula = aggregateFormula(
                "sum(x) + sum(x) + sum(sum(x) + sum(x)) + sum(sum(x) + sum(x))");

        assertEquals(30.0, formula.evaluateAsDouble(new double[]{7},
                List.of(new double[]{2}, new double[]{3})));
        assertEquals(42.0, formula.evaluateAsDouble(new double[]{7}, null));
        assertEquals(42.0, formula.evaluateAsDouble(new double[]{7}, List.of()));
    }

    @Test
    void aggregateResultsDoNotLeakAcrossInputsFormulasOrFailures() {
        ArithmeticFormula first = aggregateFormula("sum(x) + sum(x) + x");
        ArithmeticFormula second = aggregateFormula("sum(x*x) + sum(x*x)");
        double[] item = {2};
        List<double[]> items = List.of(item);

        assertEquals(14.0, first.evaluateAsDouble(new double[]{10}, items));
        assertEquals(8.0, second.evaluateAsDouble(new double[]{0}, items));
        item[0] = 4;
        assertEquals(18.0, first.evaluateAsDouble(new double[]{10}, items));
        assertThrows(ArrayIndexOutOfBoundsException.class,
                () -> first.evaluateAsDouble(new double[0], items));
        item[0] = 7;
        assertEquals(24.0, first.evaluateAsDouble(new double[]{10}, items));
    }

    @Test
    void optimizationsPreserveSignedZeroNonFiniteValuesAndAdditionOrder() {
        ArithmeticFormula repeated = aggregateFormula("sum(x) + sum(x)");
        assertBitsEqual(-0.0, repeated.evaluateAsDouble(new double[]{-0.0}, null));
        assertBitsEqual(-0.0, repeated.evaluateAsDouble(new double[]{-0.0}, List.of()));
        assertBitsEqual(0.0, repeated.evaluateAsDouble(new double[]{-0.0}, List.of(new double[]{-0.0})));
        assertBitsEqual(0.0, repeated.evaluateAsDouble(new double[]{0},
                List.of(new double[]{1e16}, new double[]{1}, new double[]{-1e16})));
        assertTrue(Double.isNaN(aggregateFormula("sum(x) - sum(x)")
                .evaluateAsDouble(new double[]{0}, List.of(new double[]{Double.POSITIVE_INFINITY}))));
        assertBitsEqual(-0.0, ArithmeticFormula.parse("x / (1 / -0.0)", Map.of("x", 0))
                .evaluateAsDouble(new double[]{1}, null));
        assertBitsEqual(0.0, ArithmeticFormula.parse("x + 0.0", Map.of("x", 0))
                .evaluateAsDouble(new double[]{-0.0}, null));
        assertTrue(Double.isNaN(ArithmeticFormula.parse("x * 0", Map.of("x", 0))
                .evaluateAsDouble(new double[]{Double.POSITIVE_INFINITY}, null)));
        assertTrue(Double.isNaN(repeated.evaluateAsDouble(new double[]{Double.NaN}, null)));
        assertEquals(4.0, repeated.evaluateAsDouble(new double[]{2}, null));
    }

    @Test
    void boundFunctionsEvaluateChangingVariables() {
        ArithmeticFormula formula = ArithmeticFormula.parse(
                "sqrt(x) + log(x) + exp(x) + abs(-x) + min(x,2) + max(x,2) + pow(x,2)",
                Map.of("x", 0));
        for (double x : new double[]{0.25, 1, 3}) {
            double expected = Math.sqrt(x) + Math.log(x) + Math.exp(x) + Math.abs(-x)
                    + Math.min(x, 2) + Math.max(x, 2) + Math.pow(x, 2);
            assertBitsEqual(expected, formula.evaluateAsDouble(new double[]{x}, null));
        }
    }

    @Test
    void largeExpressionPreservesArithmeticOrderAndAllFunctions() {
        ArithmeticFormula formula = ArithmeticFormula.parse(
                "((x+y)+z) + sqrt(abs(x)) + log(abs(y)+1) + exp(z)"
                        + " + min(x,y) + max(y,z) + pow(x,2) + x^3 - (-y) / (abs(z)+1)",
                Map.of("x", 0, "y", 1, "z", 2));
        double[] values = {-0.0, 0.0, 0.25, -3, 1e16, Double.POSITIVE_INFINITY, Double.NaN};
        for (double x : values) {
            for (double y : values) {
                for (double z : values) {
                    double expected = ((x+y)+z) + Math.sqrt(Math.abs(x)) + Math.log(Math.abs(y)+1)
                            + Math.exp(z) + Math.min(x,y) + Math.max(y,z) + Math.pow(x,2)
                            + Math.pow(x,3) - (-y) / (Math.abs(z)+1);
                    assertBitsEqual(expected, formula.evaluateAsDouble(new double[]{x,y,z}, null));
                }
            }
        }
    }

    @Test
    void largeExpressionPreservesNegativeZeroAndDoesNotSimplifyInfinityTimesZero() {
        ArithmeticFormula product = aggregateFormula("x*2*3*5*7*11*13*17*19*23");
        assertBitsEqual(-0.0, product.evaluateAsDouble(new double[]{-0.0}, null));
        ArithmeticFormula zero = aggregateFormula("x*2*3*5*7*11*13*17*19*23*0");
        assertTrue(Double.isNaN(zero.evaluateAsDouble(new double[]{Double.POSITIVE_INFINITY}, null)));
    }

    @Test
    void largeAggregateExpressionsPreserveNestedScopesAfterFailure() {
        String itemExpression = "x + x*x + x*x*x + max(x,2) + min(x,3) + abs(x) + sqrt(abs(x))";
        ArithmeticFormula formula = aggregateFormula("(" + itemExpression + ") + sum(" + itemExpression
                + ") + sum(sum(" + itemExpression + ")) + (" + itemExpression + ")");
        double[] item = {2};
        List<double[]> items = List.of(item, new double[]{3});
        for (int i = 0; i < 20; i++) {
            item[0] = i;
            double batch = largeItemValue(i) + largeItemValue(3);
            assertBitsEqual(largeItemValue(7) + batch + batch + largeItemValue(7),
                    formula.evaluateAsDouble(new double[]{7}, items));
        }
        assertBitsEqual(4 * largeItemValue(7), formula.evaluateAsDouble(new double[]{7}, null));
        assertBitsEqual(4 * largeItemValue(7), formula.evaluateAsDouble(new double[]{7}, List.of()));
        assertThrows(ArrayIndexOutOfBoundsException.class,
                () -> formula.evaluateAsDouble(new double[]{7}, List.of(new double[0])));
        assertBitsEqual(4 * largeItemValue(7), formula.evaluateAsDouble(new double[]{7}, null));
    }

    private static double largeItemValue(double x) {
        return x + x*x + x*x*x + Math.max(x,2) + Math.min(x,3) + Math.abs(x) + Math.sqrt(Math.abs(x));
    }

    @Test
    void compiledAggregatesPreserveScalarFallbackAndExactOperationOrder() {
        double[] values = {-0.0, 0.0, 1, -3, 1e308, 1e-308,
                Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, Double.NaN};
        for (String divisor : List.of("1024", "-0.0", "0.0", "1e308", "1e-308")) {
            double d = Double.parseDouble(divisor);
            assertAggregateMatchesReference("x / " + divisor, (x, y) -> x / d, values);
            assertAggregateMatchesReference("x / " + divisor + " * y / 17", (x, y) -> ((x / d) * y) / 17, values);
            assertAggregateMatchesReference("max(x / " + divisor + " - 2, -0.0)",
                    (x, y) -> Math.max(x / d - 2, -0.0), values);
        }
    }

    private static void assertAggregateMatchesReference(String term, DoubleBinaryOperator reference, double[] values) {
        ArithmeticFormula formula = ArithmeticFormula.parse("sum(" + term + ")",
                Map.of("x", 0, "y", 1), Set.of(), true);
        List<double[]> batch = new ArrayList<>();
        double expectedSum = 0.0;
        for (double x : values) {
            for (double y : values) {
                double[] item = {x, y};
                double expected = reference.applyAsDouble(x, y);
                assertBitsEqual(expected, formula.evaluateAsDouble(item, null));
                assertBitsEqual(expected, formula.evaluateAsDouble(item, List.of()));
                assertBitsEqual(0.0 + expected, formula.evaluateAsDouble(new double[]{999, 999}, List.of(item)));
                batch.add(item);
                expectedSum += expected;
                assertBitsEqual(expectedSum, formula.evaluateAsDouble(new double[]{999, 999}, batch));
            }
        }
    }

    @Test
    void compiledAggregatesRespectNestedScopesAndDoNotRetainResultsAfterFailure() {
        ArithmeticFormula formula = aggregateFormula(
                "sum(x/2) + sum(x/2) + sum(sum(x/2))");
        double[] item = {4};
        assertEquals(6.0, formula.evaluateAsDouble(new double[]{100}, List.of(item)));
        item[0] = 8;
        assertEquals(12.0, formula.evaluateAsDouble(new double[]{100}, List.of(item)));
        assertThrows(ArrayIndexOutOfBoundsException.class,
                () -> formula.evaluateAsDouble(new double[]{100}, List.of(new double[0])));
        assertEquals(12.0, formula.evaluateAsDouble(new double[]{100}, List.of(item)));
        assertBitsEqual(-0.0, formula.evaluateAsDouble(new double[]{-0.0}, null));
    }

    @Test
    void compilesArbitraryAggregateBodiesWithoutExpressionShapeSpecialCases() {
        String term = "exp(x/1024) + sqrt(abs(y)) - log(1+abs(y)) + pow(x-y,3)/max(abs(x),1)";
        ArithmeticFormula formula = ArithmeticFormula.parse("sum(" + term + ") + sum(sum(" + term + "))",
                Map.of("x", 0, "y", 1), Set.of(), true);
        List<double[]> items = new ArrayList<>();
        double expected = 0;
        for (int i = 0; i < 32; i++) {
            double x = i * 0.75 - 10, y = i * 1.5 - 20;
            items.add(new double[]{x, y});
            expected += Math.exp(x/1024) + Math.sqrt(Math.abs(y)) - Math.log(1+Math.abs(y))
                    + Math.pow(x-y,3)/Math.max(Math.abs(x),1);
            assertBitsEqual(expected + expected, formula.evaluateAsDouble(new double[0], items));
        }
    }

    @Test
    void liveFormulasRemainValidAfterCompiledCacheEviction() {
        List<ArithmeticFormula> formulas = new ArrayList<>();
        for (int i = 0; i < 160; i++) formulas.add(aggregateFormula("sum(x / " + (i+1) + ") + " + i));
        for (int i = 0; i < formulas.size(); i++) {
            assertBitsEqual(0.0 + 2.0/(i+1) + 3.0/(i+1) + i,
                    formulas.get(i).evaluateAsDouble(new double[0], List.of(new double[]{2}, new double[]{3})));
        }
    }

    @Test
    void oversizedJvmMethodFallsBackWithoutRejectingValidFormula() {
        // Balanced depth avoids testing parser recursion limits instead of JVM code size.
        List<String> terms = new ArrayList<>();
        for (int i = 0; i < 4000; i++) terms.add("(x + " + i + ")");
        while (terms.size() > 1) {
            List<String> next = new ArrayList<>();
            for (int i = 0; i < terms.size(); i += 2) {
                next.add(i+1 == terms.size() ? terms.get(i) : "(" + terms.get(i) + "+" + terms.get(i+1) + ")");
            }
            terms = next;
        }
        ArithmeticFormula formula = aggregateFormula(terms.getFirst());
        assertEquals(8_002_000.0, formula.evaluateAsDouble(new double[]{1}, null));
        assertEquals(8_006_000.0, formula.evaluateAsDouble(new double[]{2}, null));
    }

    @Test
    void oneFormulaCanBeEvaluatedConcurrentlyWithIndependentInputs() throws Exception {
        ArithmeticFormula formula = aggregateFormula("sum(x*x) + sum(x*x) + sum(x)"
                + " + max(x,0) + min(x,0) + abs(x) + sqrt(abs(x)) + x^2 + sum(x/2) + sum(x/2)");
        int threads = 6;
        CountDownLatch ready = new CountDownLatch(threads);
        try (var executor = Executors.newFixedThreadPool(threads)) {
            List<Future<?>> results = new ArrayList<>();
            for (int i = 0; i < threads; i++) {
                int offset = i * 1000;
                results.add(executor.submit(() -> {
                    ready.countDown();
                    assertTrue(ready.await(5, TimeUnit.SECONDS));
                    for (int j = 0; j < 1000; j++) {
                        double x = offset + j;
                        double expected = 2 * (x*x + (x+1)*(x+1)) + x + (x+1);
                        double scaled = x/2 + (x+1)/2;
                        expected = expected + scaled + scaled;
                        assertBitsEqual(expected, formula.evaluateAsDouble(new double[]{0},
                                List.of(new double[]{x}, new double[]{x+1})));
                    }
                    return null;
                }));
            }
            for (Future<?> result : results) {
                result.get(10, TimeUnit.SECONDS);
            }
        }
    }

    private static ArithmeticFormula aggregateFormula(String expression) {
        return ArithmeticFormula.parse(expression, Map.of("x", 0), Set.of(), true);
    }

    private static void assertBitsEqual(double expected, double actual) {
        assertEquals(Double.doubleToLongBits(expected), Double.doubleToLongBits(actual));
    }

    @Test
    void returnsNonFiniteResultsForTheCallerToValidate() {
        assertTrue(Double.isNaN(ArithmeticFormula.parse("sqrt(-1)", Map.of())
                .evaluateAsDouble(new double[0], null)));
        assertEquals(Double.POSITIVE_INFINITY, ArithmeticFormula.parse("1/0", Map.of())
                .evaluateAsDouble(new double[0], null));
    }

    @Test
    void prefillWrapperPreservesAggregateBindingsAndRestrictions() {
        PrefillTimeFormula formula = PrefillTimeFormula.parse("sum(computeTokens) + 0.3*sum(hitCacheTokens)");
        double[] batch = new double[PrefillTimeFormula.VAR_COUNT];
        double[] first = new double[PrefillTimeFormula.VAR_COUNT];
        first[PrefillTimeFormula.IDX_COMPUTE_TOKENS] = 100;
        first[PrefillTimeFormula.IDX_HIT_CACHE_TOKENS] = 50;
        double[] second = new double[PrefillTimeFormula.VAR_COUNT];
        second[PrefillTimeFormula.IDX_COMPUTE_TOKENS] = 200;
        second[PrefillTimeFormula.IDX_HIT_CACHE_TOKENS] = 150;

        assertEquals(360.0, formula.evaluateAsDouble(batch, List.of(first, second)));
        assertEquals(360L, formula.evaluate(batch, List.of(first, second)));
        assertThrows(IllegalArgumentException.class, () -> PrefillTimeFormula.parse("sum(totalComputeTokens)"));
        assertThrows(IllegalArgumentException.class, () -> PrefillTimeFormula.parse("running_size"));
    }
}
