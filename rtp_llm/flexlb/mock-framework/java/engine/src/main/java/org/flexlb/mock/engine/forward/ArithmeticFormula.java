package org.flexlb.mock.engine.forward;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.flexlb.mock.engine.forward.ArithmeticFormulaAst.Node;

/**
 * Arithmetic expression engine shared by routing cost and prefill-time formulas.
 * Supports {@code + - * / ^}, {@code sqrt}, {@code log}, {@code exp}, {@code abs},
 * {@code max}, {@code min}, {@code pow}, and named constants via {@code param(name, value)}.
 * Batch-aware callers may additionally enable {@code sum(expression)}.
 */
final class ArithmeticFormula {
    // Bounded reuse also avoids one generated class per equal-model endpoint.
    private static final int MAX_COMPILED_FORMULAS = 128;

    private static final Map<Node, Executable> COMPILED =
            new LinkedHashMap<>(MAX_COMPILED_FORMULAS, 0.75f, true);

    private final Executable compiled;

    private ArithmeticFormula(Node root) {
        synchronized (COMPILED) {
            this.compiled =
                    COMPILED.computeIfAbsent(root, ArithmeticFormulaCompiler::compile);
            if (COMPILED.size() > MAX_COMPILED_FORMULAS) {
                COMPILED.remove(COMPILED.keySet().iterator().next());
            }
        }
    }

    /**
     * Parse with explicit variable bindings and aggregate policy.
     * Variables in {@code aggregateExcludedVariables} may only appear outside {@code sum()}.
     *
     * @throws IllegalArgumentException for malformed expressions, unknown names, or invalid variable indices.
     */
    public static ArithmeticFormula parse(
            String expression,
            Map<String, Integer> variables,
            Set<String> aggregateExcludedVariables,
            boolean allowAggregates) {
        if (expression == null) {
            throw new IllegalArgumentException("Formula expression is required");
        }
        ArithmeticFormulaParser parser =
                new ArithmeticFormulaParser(
                        expression, variables, aggregateExcludedVariables, allowAggregates);
        return new ArithmeticFormula(parser.parse());
    }

    /**
     * Retain fractional and non-finite results for the caller's validation boundary.
     * Inputs must remain stable during this call; aggregates may share one traversal.
     */
    public double evaluateAsDouble(double[] vars, List<double[]> itemVars) {
        return compiled.evaluate(vars, itemVars);
    }

    /**
     * Evaluates a compiled expression against scalar variables and optional item variables.
     */
    interface Executable {
        double evaluate(double[] vars, List<double[]> items);
    }
}
