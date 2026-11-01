package org.flexlb.balance.prediction;

import java.util.Map;

import static com.google.common.base.Preconditions.checkArgument;

/** Immutable Decode cost expression evaluated against one routing snapshot. */
public final class DecodeCostFormula {

    private static final Map<String, Integer> VARIABLES =
            Map.of(
                    "running_size", 0,
                    "max_running_size", 1,
                    "kvcache_used", 2,
                    "kvcache_capacity", 3,
                    "kvcache_used_ratio", 4);
    private final String expression;
    private final ArithmeticFormula formula;
    private final ThreadLocal<Evaluator> evaluators;

    private DecodeCostFormula(String expression, ArithmeticFormula formula) {
        this.expression = expression;
        this.formula = formula;
        this.evaluators = ThreadLocal.withInitial(() -> new Evaluator(formula));
    }

    public static DecodeCostFormula parse(String expression) {
        checkArgument(
                expression != null && !expression.isBlank(),
                "Decode cost expression must not be blank");
        return new DecodeCostFormula(expression, ArithmeticFormula.parse(expression, VARIABLES));
    }

    public String expression() {
        return expression;
    }

    public boolean requiresMaxRunningSize() {
        return formula.referencesVariable("max_running_size");
    }

    /** Borrow on the calling thread for one synchronous scoring pass; never share across threads. */
    public Evaluator evaluator() {
        return evaluators.get();
    }

    public static final class Evaluator {
        private final ArithmeticFormula formula;
        private final double[] vars = new double[VARIABLES.size()];

        private Evaluator(ArithmeticFormula formula) {
            this.formula = formula;
        }

        public double evaluate(
                long runningSize, long maxRunningSize, long kvUsed, long kvCapacity) {
            vars[0] = runningSize;
            vars[1] = maxRunningSize;
            vars[2] = kvUsed;
            vars[3] = kvCapacity;
            vars[4] = kvCapacity == 0L ? 0.0 : (double) kvUsed / kvCapacity;
            return formula.evaluateAsDouble(vars, null);
        }
    }
}
