package org.flexlb.balance.prediction;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.List;
import java.util.Map;
import java.util.Set;

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
