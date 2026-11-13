package org.flexlb.util.formula;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Set;

import static com.google.common.base.Preconditions.checkArgument;
import static org.flexlb.util.formula.ArithmeticFormulaAst.Node;

/**
 * Evaluates scalar arithmetic expressions for routing costs.
 * Supports {@code + - * / ^}, {@code sqrt}, {@code log}, {@code exp}, {@code abs},
 * {@code max}, {@code min}, {@code pow}, and named constants via {@code param(name, value)}.
 */
public final class ArithmeticFormula {
    // Bounded reuse also avoids one generated class per equal-model endpoint.
    private static final int MAX_COMPILED_FORMULAS = 128;

    private static final Map<ParseKey, ArithmeticFormula> COMPILED =
            new LinkedHashMap<>(MAX_COMPILED_FORMULAS, 0.75f, true);

    private final Executable compiled;
    private final Set<String> referencedVariables;

    private ArithmeticFormula(Node root, Set<String> referencedVariables) {
        this.compiled = ArithmeticFormulaCompiler.compile(root);
        this.referencedVariables = Set.copyOf(referencedVariables);
    }

    public static ArithmeticFormula parse(String expression, Map<String, Integer> variables) {
        checkArgument(expression != null, "Formula expression is required");
        ParseKey key = new ParseKey(expression, Map.copyOf(variables));
        synchronized (COMPILED) {
            ArithmeticFormula cached = COMPILED.get(key);
            if (cached != null) return cached;
        }
        ArithmeticFormulaParser parser = new ArithmeticFormulaParser(expression, key.variables());
        Node root = parser.parse();
        synchronized (COMPILED) {
            ArithmeticFormula result =
                    COMPILED.computeIfAbsent(
                            key,
                            ignored -> new ArithmeticFormula(root, parser.referencedVariables()));
            if (COMPILED.size() > MAX_COMPILED_FORMULAS) {
                COMPILED.remove(COMPILED.keySet().iterator().next());
            }
            return result;
        }
    }

    /**
     * Includes variables parsed inside a {@code param()} initial value.
     */
    public boolean referencesVariable(String variable) {
        return referencedVariables.contains(variable);
    }

    /**
     * Retain fractional and non-finite results for the caller's validation boundary.
     */
    public double evaluateAsDouble(double[] vars) {
        return compiled.evaluate(vars);
    }

    private record ParseKey(String expression, Map<String, Integer> variables) {}

    /**
     * Evaluates a compiled arithmetic expression against variables in their assigned array positions.
     */
    interface Executable {
        double evaluate(double[] vars);
    }
}
