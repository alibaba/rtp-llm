package org.flexlb.balance.prediction;

import org.junit.jupiter.api.Test;
import java.util.*;
import static org.junit.jupiter.api.Assertions.*;

class IncrementalArithmeticFormulaTest {
    private final Random random = new Random(20260917);

    @Test
    void randomTreesMatchFullEvaluationAtEveryPrefix() {
        for (int trial = 0; trial < 500; trial++) {
            String text = expression(4);
            var formula = ArithmeticFormula.parse(text, Map.of("x", 0, "y", 1), Set.of(), true);
            var cursor = formula.newAggregation();
            assertNotNull(cursor);
            var items = new ArrayList<double[]>();
            for (int i = 0; i < 32; i++) {
                double[] item = {value(), value()};
                items.add(item);
                cursor.append(item);
                double[] vars = {value(), value()};
                assertEquals(Double.doubleToLongBits(formula.evaluateAsDouble(vars, items)),
                        Double.doubleToLongBits(cursor.evaluate(vars)), text + " prefix=" + items.size());
            }
        }
    }

    @Test
    void emptyCursorCannotAccidentallyReplaceScalarEmptyBatchSemantics() {
        var formula = ArithmeticFormula.parse("sum(x)", Map.of("x", 0), Set.of(), true);
        var cursor = formula.newAggregation();
        assertThrows(IllegalStateException.class, () -> cursor.evaluate(new double[]{42}));
        assertEquals(42, formula.evaluateAsDouble(new double[]{42}, List.of()));
        cursor.append(new double[]{-0.0});
        assertEquals(Double.doubleToLongBits(0.0), Double.doubleToLongBits(cursor.evaluate(new double[]{42})));
    }

    private double value() {
        return switch (random.nextInt(12)) {
            case 0 -> -0.0;
            case 1 -> 0.0;
            case 2 -> Double.NaN;
            case 3 -> Double.POSITIVE_INFINITY;
            case 4 -> Double.NEGATIVE_INFINITY;
            default -> random.nextDouble() * 20 - 5;
        };
    }

    private String expression(int depth) {
        if (depth == 0) return List.of("x", "y", "-0.0", "1e308").get(random.nextInt(4));
        String a = expression(depth - 1);
        return switch (random.nextInt(10)) {
            case 0 -> "sum(" + a + ")";
            case 1 -> "(" + a + ") + (" + a + ")";
            case 2 -> "sqrt(" + a + ")";
            case 3 -> "log(" + a + ")";
            case 4 -> "exp(" + a + ")";
            case 5 -> "max(" + a + "," + expression(depth - 1) + ")";
            default -> "(" + a + ")" + "+-*/^".charAt(random.nextInt(5))
                    + "(" + expression(depth - 1) + ")";
        };
    }
}
