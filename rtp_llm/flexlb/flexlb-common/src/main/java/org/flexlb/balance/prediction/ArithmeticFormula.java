package org.flexlb.balance.prediction;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;
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
    private static final Map<ParseKey, ArithmeticFormula> COMPILED =
            new LinkedHashMap<>(MAX_COMPILED_FORMULAS, 0.75f, true);

    private record ParseKey(String expression, Map<String, Integer> variables,
                            Set<String> excludedVariables, boolean allowAggregates) { }

    private final Compiled compiled;

    private record Compiled(Executable full, Executable bindings,
                            ArithmeticFormulaCompiler.Incremental incremental) { }
    private final Set<String> referencedVariables;

    private ArithmeticFormula(Node root, Set<String> referencedVariables, boolean allowAggregates) {
        this.compiled = new Compiled(ArithmeticFormulaCompiler.compile(root, false),
                allowAggregates ? ArithmeticFormulaCompiler.compile(root, true) : null,
                ArithmeticFormulaCompiler.compileIncremental(root));
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
        checkArgument(expression != null, "Formula expression is required");
        ParseKey key = new ParseKey(expression,
                Map.copyOf(java.util.Objects.requireNonNull(variables, "variables")),
                Set.copyOf(java.util.Objects.requireNonNull(aggregateExcludedVariables, "aggregateExcludedVariables")),
                allowAggregates);
        synchronized (COMPILED) {
            ArithmeticFormula cached = COMPILED.get(key);
            if (cached != null) return cached;
        }
        ArithmeticFormulaParser parser = new ArithmeticFormulaParser(
                expression, key.variables(), key.excludedVariables(), allowAggregates);
        Node root = parser.parse();
        synchronized (COMPILED) {
            ArithmeticFormula result = COMPILED.computeIfAbsent(key,
                    ignored -> new ArithmeticFormula(root, parser.referencedVariables(), allowAggregates));
            if (COMPILED.size() > MAX_COMPILED_FORMULAS) {
                COMPILED.remove(COMPILED.keySet().iterator().next());
            }
            return result;
        }
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

    /** Stable per-item bindings read directly by the compiled aggregate loop. */
    public interface Variables {
        double variable(int index);
    }

    public double evaluateWithBindings(double[] vars, List<? extends Variables> items) {
        checkState(compiled.bindings() != null, "Batch aggregates are disabled");
        return compiled.bindings().evaluate(vars, items);
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
            checkState(nonempty, "Append an item before evaluating a batch");
            return program.evaluate(batchVars, sums);
        }
    }

    interface Executable {
        double evaluate(double[] vars, List<?> items);
    }
}
