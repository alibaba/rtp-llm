package org.flexlb.util.formula;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import java.util.Map;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/**
 * Checks formula parsing and evaluation against supported arithmetic and aggregation semantics.
 */
class ArithmeticFormulaTest {

    @Test
    void evaluatesScalarVariablesWithCallerDefinedIndices() {
        ArithmeticFormula formula =
                ArithmeticFormula.parse(
                        "running_size / max_running_size + cache_ratio^2",
                        Map.of("running_size", 2, "max_running_size", 0, "cache_ratio", 1));

        assertEquals(0.75, formula.evaluateAsDouble(new double[] {8, 0.5, 4}));
        assertTrue(formula.referencesVariable("running_size"));
        assertFalse(formula.referencesVariable("running"));
    }

    @Test
    void preservesFunctionsScientificNotationAndOperatorPrecedence() {
        ArithmeticFormula functions =
                ArithmeticFormula.parse(
                        "sqrt(9) + log(exp(2)) + abs(-4) + min(7,5) + max(2,3) + pow(2,3) + 1e-3*1000",
                        Map.of());
        assertEquals(26.0, functions.evaluateAsDouble(new double[0]), 1e-12);
        assertEquals(
                512.0, ArithmeticFormula.parse("2^3^2", Map.of()).evaluateAsDouble(new double[0]));
        // Keep the established prefill grammar: unary signs bind before exponentiation.
        assertEquals(
                4.0, ArithmeticFormula.parse("-2^2", Map.of()).evaluateAsDouble(new double[0]));
    }

    @Test
    void recordsVariablesInParameterInitializersButNotParameterNames() {
        ArithmeticFormula formula =
                ArithmeticFormula.parse(
                        "param(weight, signal + 2) + param(weight, signal + 2)",
                        Map.of("signal", 3));

        assertTrue(formula.referencesVariable("signal"));
        assertFalse(formula.referencesVariable("weight"));
        assertEquals(
                4.0,
                formula.evaluateAsDouble(new double[] {0, 0, 0, 99}),
                "parameter initial values retain their parse-time zero-variable bindings");
    }

    @Test
    void rejectsInconsistentParameterDefaults() {
        assertThrows(
                IllegalArgumentException.class,
                () -> ArithmeticFormula.parse("param(weight, 1) + param(weight, 2)", Map.of()));
    }

    @ParameterizedTest
    @ValueSource(
            strings = {
                "",
                "x +",
                "x y",
                "unknown + 1",
                "sqrt()",
                "max(1)",
                "unknown(1)",
                "param(weight, unknown)",
                "param(weight, 1, 2)",
                "1.2.3"
            })
    void rejectsMalformedExpressionsAndUnknownNames(String expression) {
        assertThrows(
                IllegalArgumentException.class,
                () -> ArithmeticFormula.parse(expression, Map.of("x", 0)));
    }

    @Test
    void boundFunctionsEvaluateChangingVariables() {
        ArithmeticFormula formula =
                ArithmeticFormula.parse(
                        "sqrt(x) + log(x) + exp(x) + abs(-x) + min(x,2) + max(x,2) + pow(x,2)",
                        Map.of("x", 0));
        for (double x : new double[] {0.25, 1, 3}) {
            double expected =
                    Math.sqrt(x)
                            + Math.log(x)
                            + Math.exp(x)
                            + Math.abs(-x)
                            + Math.min(x, 2)
                            + Math.max(x, 2)
                            + Math.pow(x, 2);
            assertBitsEqual(expected, formula.evaluateAsDouble(new double[] {x}));
        }
    }

    @Test
    void largeExpressionPreservesArithmeticOrderAndAllFunctions() {
        ArithmeticFormula formula =
                ArithmeticFormula.parse(
                        "((x+y)+z) + sqrt(abs(x)) + log(abs(y)+1) + exp(z)"
                                + " + min(x,y) + max(y,z) + pow(x,2) + x^3 - (-y) / (abs(z)+1)",
                        Map.of("x", 0, "y", 1, "z", 2));
        double[] values = {-0.0, 0.0, 0.25, -3, 1e16, Double.POSITIVE_INFINITY, Double.NaN};
        for (double x : values) {
            for (double y : values) {
                for (double z : values) {
                    double expected =
                            ((x + y) + z)
                                    + Math.sqrt(Math.abs(x))
                                    + Math.log(Math.abs(y) + 1)
                                    + Math.exp(z)
                                    + Math.min(x, y)
                                    + Math.max(y, z)
                                    + Math.pow(x, 2)
                                    + Math.pow(x, 3)
                                    - (-y) / (Math.abs(z) + 1);
                    assertBitsEqual(expected, formula.evaluateAsDouble(new double[] {x, y, z}));
                }
            }
        }
    }

    @Test
    void returnsNonFiniteResultsForTheCallerToValidate() {
        assertTrue(
                Double.isNaN(
                        ArithmeticFormula.parse("sqrt(-1)", Map.of())
                                .evaluateAsDouble(new double[0])));
        assertEquals(
                Double.POSITIVE_INFINITY,
                ArithmeticFormula.parse("1/0", Map.of()).evaluateAsDouble(new double[0]));
    }

    private static double largeItemValue(double x) {
        return x
                + x * x
                + x * x * x
                + Math.max(x, 2)
                + Math.min(x, 3)
                + Math.abs(x)
                + Math.sqrt(Math.abs(x));
    }

    private static void assertBitsEqual(double expected, double actual) {
        assertEquals(Double.doubleToLongBits(expected), Double.doubleToLongBits(actual));
    }
}
