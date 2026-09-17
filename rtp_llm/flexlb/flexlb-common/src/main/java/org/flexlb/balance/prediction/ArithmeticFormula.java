package org.flexlb.balance.prediction;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static org.flexlb.balance.prediction.ArithmeticFormulaAst.Node;

/**
 * Arithmetic expression engine shared by routing cost and prefill-time formulas.
 * Supports {@code + - * / ^}, {@code sqrt}, {@code log}, {@code exp}, {@code abs},
 * {@code max}, {@code min}, {@code pow}, and named constants via {@code param(name, value)}.
 * Batch-aware callers may additionally enable {@code sum(expression)}.
 */
public final class ArithmeticFormula {

    // Bounded reuse also avoids one generated class per equal-model endpoint.
    private static final int MAX_COMPILED_FORMULAS = 128;
    private static final Map<Node, Compiled> COMPILED = new LinkedHashMap<>(MAX_COMPILED_FORMULAS, 0.75f, true);

    private final Compiled compiled;

    private record Compiled(Executable full, ArithmeticFormulaCompiler.Incremental incremental) { }
    private final Set<String> referencedVariables;

    private ArithmeticFormula(Node root, Set<String> referencedVariables) {
        synchronized (COMPILED) {
            this.compiled = COMPILED.computeIfAbsent(root, node -> new Compiled(
                    ArithmeticFormulaCompiler.compile(node), ArithmeticFormulaCompiler.compileIncremental(node)));
            if (COMPILED.size() > MAX_COMPILED_FORMULAS) {
                COMPILED.remove(COMPILED.keySet().iterator().next());
            }
        }
        this.referencedVariables = Set.copyOf(referencedVariables);
    }

    /** Parse a scalar expression; batch aggregates are not available. */
    public static ArithmeticFormula parse(String expression, Map<String, Integer> variables) {
        return parse(expression, variables, Set.of(), false);
    }

    /**
     * Parse with explicit variable bindings and aggregate policy.
     * Variables in {@code aggregateExcludedVariables} may only appear outside {@code sum()}.
     *
     * @throws IllegalArgumentException for malformed expressions, unknown names, or invalid variable indices.
     */
    public static ArithmeticFormula parse(String expression, Map<String, Integer> variables,
                                          Set<String> aggregateExcludedVariables, boolean allowAggregates) {
        if (expression == null) {
            throw new IllegalArgumentException("Formula expression is required");
        }
        ArithmeticFormulaParser parser = new ArithmeticFormulaParser(
                expression, variables, aggregateExcludedVariables, allowAggregates);
        return new ArithmeticFormula(parser.parse(), parser.referencedVariables());
    }

    /** Includes variables parsed inside a {@code param()} initial value. */
    public boolean referencesVariable(String variable) {
        return referencedVariables.contains(variable);
    }

    /**
     * Retain fractional and non-finite results for the caller's validation boundary.
     * Inputs must remain stable during this call; aggregates may share one traversal.
     */
    public double evaluateAsDouble(double[] vars, List<double[]> itemVars) {
        return compiled.full().evaluate(vars, itemVars);
    }

    /** A fresh, caller-owned accumulator for a nonempty append-only batch; null on compiler fallback. */
    public Aggregation newAggregation() {
        return compiled.incremental() == null ? null : new Aggregation(compiled.incremental());
    }

    public static final class Aggregation {
        private final ArithmeticFormulaCompiler.Incremental program;
        private final double[] sums;
        private boolean nonempty;

        private Aggregation(ArithmeticFormulaCompiler.Incremental program) {
            this.program = program;
            sums = new double[program.aggregateCount()];
        }

        public void append(double[] itemVars) {
            program.append(itemVars, sums);
            nonempty = true;
        }

        public double evaluate(double[] batchVars) {
            if (!nonempty) throw new IllegalStateException("Append an item before evaluating a batch");
            return program.evaluate(batchVars, sums);
        }
    }

    interface Executable {
        double evaluate(double[] vars, List<double[]> items);
    }
}
