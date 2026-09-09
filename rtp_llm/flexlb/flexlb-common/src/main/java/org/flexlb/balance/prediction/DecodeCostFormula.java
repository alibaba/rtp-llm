package org.flexlb.balance.prediction;

import java.util.Map;

/** Immutable Decode cost expression evaluated against one routing snapshot. */
public final class DecodeCostFormula {

    private static final Map<String, Integer> VARIABLES = Map.of(
            "running_size", 0,
            "max_running_size", 1,
            "kvcache_used", 2,
            "kvcache_capacity", 3,
            "kvcache_used_ratio", 4);
    private static final ThreadLocal<double[]> EVALUATION_VARIABLES =
            ThreadLocal.withInitial(() -> new double[VARIABLES.size()]);

    private final String expression;
    private final ArithmeticFormula formula;

    private DecodeCostFormula(String expression, ArithmeticFormula formula) {
        this.expression = expression;
        this.formula = formula;
    }

    public static DecodeCostFormula parse(String expression) {
        if (expression == null || expression.isBlank()) {
            throw new IllegalArgumentException("Decode cost expression must not be blank");
        }
        return new DecodeCostFormula(expression, ArithmeticFormula.parse(expression, VARIABLES));
    }

    public String expression() {
        return expression;
    }

    public boolean requiresMaxRunningSize() {
        return formula.referencesVariable("max_running_size");
    }

    public double evaluate(long runningSize, long maxRunningSize, long kvUsed, long kvCapacity) {
        double[] vars = EVALUATION_VARIABLES.get();
        vars[0] = runningSize;
        vars[1] = maxRunningSize;
        vars[2] = kvUsed;
        vars[3] = kvCapacity;
        vars[4] = kvCapacity == 0L ? 0.0 : (double) kvUsed / kvCapacity;
        return formula.evaluateAsDouble(vars, null);
    }
}
