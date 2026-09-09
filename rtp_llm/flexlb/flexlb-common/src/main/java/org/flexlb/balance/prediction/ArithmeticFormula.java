package org.flexlb.balance.prediction;

import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;

/**
 * Arithmetic expression engine shared by routing cost and prefill-time formulas.
 * Supports {@code + - * / ^}, {@code sqrt}, {@code log}, {@code exp}, {@code abs},
 * {@code max}, {@code min}, {@code pow}, and named constants via {@code param(name, value)}.
 * Batch-aware callers may additionally enable {@code sum(expression)}.
 */
public final class ArithmeticFormula {

    private static final ThreadLocal<EvalContext> EVALUATION_CONTEXT =
            ThreadLocal.withInitial(EvalContext::new);

    private static final Map<String, DoubleUnaryOperator> UNARY_FUNCTIONS = Map.of(
            "sqrt",  Math::sqrt,
            "log",   Math::log,
            "exp",   Math::exp,
            "abs",   Math::abs
    );

    private static final Map<String, DoubleBinaryOperator> BINARY_FUNCTIONS = Map.of(
            "max", Math::max,
            "min", Math::min,
            "pow", Math::pow
    );

    private static final Set<String> AGGREGATE_FUNCTIONS = Set.of("sum");

    private final Node root;
    private final Set<String> referencedVariables;

    private ArithmeticFormula(Node root, Set<String> referencedVariables) {
        this.root = root;
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
        Parser parser = new Parser(expression, variables, aggregateExcludedVariables, allowAggregates);
        Node root = parser.parseExpression();
        parser.expectEnd();
        return new ArithmeticFormula(root, parser.referencedVariables);
    }

    /** Includes variables parsed inside a {@code param()} initial value. */
    public boolean referencesVariable(String variable) {
        return referencedVariables.contains(variable);
    }

    /** Retain fractional and non-finite results for the caller's validation boundary. */
    public double evaluateAsDouble(double[] vars, List<double[]> itemVars) {
        EvalContext context = EVALUATION_CONTEXT.get();
        context.reset(vars, itemVars);
        try {
            return root.evaluate(context);
        } finally {
            context.reset(null, null);
        }
    }

    // ---- AST nodes ----

    private interface Node {
        double evaluate(EvalContext ctx);
    }

    private static final class EvalContext {
        double[] vars;
        List<double[]> itemVars;

        private EvalContext() {
        }

        private void reset(double[] vars, List<double[]> itemVars) {
            this.vars = vars;
            this.itemVars = itemVars;
        }
    }

    private record ConstantNode(double value) implements Node {
        @Override
        public double evaluate(EvalContext ctx) {
            return value;
        }
    }

    private record VariableNode(int varIndex) implements Node {
        @Override
        public double evaluate(EvalContext ctx) {
            return ctx.vars[varIndex];
        }
    }

    private record UnaryNode(char op, Node operand) implements Node {
        @Override
        public double evaluate(EvalContext ctx) {
            double v = operand.evaluate(ctx);
            return op == '-' ? -v : v;
        }
    }

    private record BinaryNode(char op, Node left, Node right) implements Node {
        @Override
        public double evaluate(EvalContext ctx) {
            double l = left.evaluate(ctx);
            double r = right.evaluate(ctx);
            return switch (op) {
                case '+' -> l + r;
                case '-' -> l - r;
                case '*' -> l * r;
                case '/' -> l / r;
                case '^' -> Math.pow(l, r);
                default  -> throw new IllegalStateException("Unknown operator: " + op);
            };
        }
    }

    private record UnaryFuncNode(String name, Node arg) implements Node {
        @Override
        public double evaluate(EvalContext ctx) {
            double a = arg.evaluate(ctx);
            return UNARY_FUNCTIONS.get(name).applyAsDouble(a);
        }
    }

    private record BinaryFuncNode(String name, Node left, Node right) implements Node {
        @Override
        public double evaluate(EvalContext ctx) {
            double l = left.evaluate(ctx);
            double r = right.evaluate(ctx);
            return BINARY_FUNCTIONS.get(name).applyAsDouble(l, r);
        }
    }

    private record AggregateFuncNode(Node arg) implements Node {
        @Override
        public double evaluate(EvalContext ctx) {
            List<double[]> itemVars = ctx.itemVars;
            if (itemVars == null || itemVars.isEmpty()) {
                ctx.itemVars = null;
                try {
                    return arg.evaluate(ctx);
                } finally {
                    ctx.itemVars = itemVars;
                }
            }
            double total = 0.0;
            double[] savedVars = ctx.vars;
            ctx.itemVars = null;
            try {
                for (double[] item : itemVars) {
                    ctx.vars = item;
                    total += arg.evaluate(ctx);
                }
            } finally {
                ctx.vars = savedVars;
                ctx.itemVars = itemVars;
            }
            return total;
        }
    }

    private static final class ParameterNode implements Node {
        private final double value;

        ParameterNode(double initialValue) {
            this.value = initialValue;
        }

        @Override
        public double evaluate(EvalContext ctx) {
            return value;
        }

        double value() {
            return value;
        }
    }

    // ---- Recursive-descent parser ----

    private static final class Parser {
        private final String input;
        private final Map<String, ParameterNode> parameters = new LinkedHashMap<>();
        private final Map<String, Integer> variables;
        private final Set<String> aggregateExcludedVariables;
        private final Set<String> referencedVariables = new HashSet<>();
        private final boolean allowAggregates;
        private final int variableCount;
        private int pos;
        private int aggregateDepth;

        Parser(String input, Map<String, Integer> variables,
               Set<String> aggregateExcludedVariables, boolean allowAggregates) {
            this.input = input;
            this.variables = Map.copyOf(Objects.requireNonNull(variables, "variables"));
            this.aggregateExcludedVariables = Set.copyOf(
                    Objects.requireNonNull(aggregateExcludedVariables, "aggregateExcludedVariables"));
            this.allowAggregates = allowAggregates;
            int count = 0;
            for (int index : this.variables.values()) {
                if (index < 0 || index == Integer.MAX_VALUE) {
                    throw new IllegalArgumentException("Variable index is outside its domain: " + index);
                }
                count = Math.max(count, index + 1);
            }
            this.variableCount = count;
        }

        // expression → term (('+' | '-') term)*
        Node parseExpression() {
            Node node = parseTerm();
            while (true) {
                skipWs();
                if (match('+')) {
                    node = new BinaryNode('+', node, parseTerm());
                } else if (match('-')) {
                    node = new BinaryNode('-', node, parseTerm());
                } else {
                    return node;
                }
            }
        }

        // term → factor (('*' | '/') factor)*
        Node parseTerm() {
            Node node = parseFactor();
            while (true) {
                skipWs();
                if (match('*')) {
                    node = new BinaryNode('*', node, parseFactor());
                } else if (match('/')) {
                    node = new BinaryNode('/', node, parseFactor());
                } else {
                    return node;
                }
            }
        }

        // factor → unary ('^' factor)*    right-associative
        Node parseFactor() {
            Node node = parseUnary();
            while (true) {
                skipWs();
                if (match('^')) {
                    Node right = parseFactor();  // right-assoc: a^b^c = a^(b^c)
                    node = new BinaryNode('^', node, right);
                } else {
                    return node;
                }
            }
        }

        // unary → ('+' | '-') unary | primary
        Node parseUnary() {
            skipWs();
            if (match('+')) {
                return new UnaryNode('+', parseUnary());
            }
            if (match('-')) {
                return new UnaryNode('-', parseUnary());
            }
            return parsePrimary();
        }

        // primary → '(' expression ')' | function_call | param_call | number | variable
        Node parsePrimary() {
            skipWs();
            if (match('(')) {
                Node node = parseExpression();
                skipWs();
                if (!match(')')) {
                    throw error("Expected ')'");
                }
                return node;
            }
            if (hasNext() && Character.isLetter(peek())) {
                String name = parseIdentifier();
                skipWs();
                if (match('(')) {
                    if (name.equals("param")) {
                        return parseParamCall();
                    }
                    return parseFuncCall(name);
                }
                if (name.equals("param")) {
                    throw error("'param' must be used as param(name, initialValue)");
                }
                if (aggregateDepth > 0 && aggregateExcludedVariables.contains(name)) {
                    throw error("Batch-scoped variable cannot be used inside sum(): " + name);
                }
                Integer idx = variables.get(name);
                if (idx == null) {
                    throw error("Unknown variable: " + name);
                }
                referencedVariables.add(name);
                return new VariableNode(idx);
            }
            if (hasNext() && (Character.isDigit(peek()) || peek() == '.')) {
                return parseNumber();
            }
            throw error("Expected number, variable, or '('");
        }

        // param(name, initialValue) → ParameterNode
        Node parseParamCall() {
            skipWs();
            if (!hasNext() || !(Character.isLetter(peek()) || peek() == '_')) {
                throw error("Expected parameter name in param()");
            }
            String paramName = parseIdentifier();
            skipWs();
            if (!match(',')) {
                throw error("Expected ',' after parameter name in param()");
            }
            skipWs();
            Node initialValueNode = parseExpression();
            EvalContext initialValueContext = new EvalContext();
            initialValueContext.reset(new double[variableCount], null);
            double initialValue = initialValueNode.evaluate(initialValueContext);
            skipWs();
            if (!match(')')) {
                throw error("Expected ')' after param() arguments");
            }
            ParameterNode existing = parameters.get(paramName);
            if (existing != null) {
                if (existing.value() != initialValue) {
                    throw error("Inconsistent initial value for parameter '" + paramName
                            + "': " + existing.value() + " vs " + initialValue);
                }
                return existing;
            }
            ParameterNode node = new ParameterNode(initialValue);
            parameters.put(paramName, node);
            return node;
        }

        Node parseFuncCall(String name) {
            if (!UNARY_FUNCTIONS.containsKey(name)
                    && !BINARY_FUNCTIONS.containsKey(name)
                    && !(allowAggregates && AGGREGATE_FUNCTIONS.contains(name))) {
                throw error("Unknown function: " + name);
            }
            skipWs();
            boolean aggregate = AGGREGATE_FUNCTIONS.contains(name);
            if (aggregate) {
                aggregateDepth++;
            }
            Node arg0;
            try {
                arg0 = parseExpression();
            } finally {
                if (aggregate) {
                    aggregateDepth--;
                }
            }
            if (aggregate) {
                skipWs();
                if (!match(')')) {
                    throw error("Expected ')' after aggregate function argument");
                }
                return new AggregateFuncNode(arg0);
            }
            if (BINARY_FUNCTIONS.containsKey(name)) {
                skipWs();
                if (!match(',')) {
                    throw error("Expected ',' in binary function '" + name + "'");
                }
                skipWs();
                Node arg1 = parseExpression();
                skipWs();
                if (!match(')')) {
                    throw error("Expected ')' after function arguments");
                }
                return new BinaryFuncNode(name, arg0, arg1);
            }
            skipWs();
            if (!match(')')) {
                throw error("Expected ')' after function argument");
            }
            return new UnaryFuncNode(name, arg0);
        }

        Node parseNumber() {
            int start = pos;
            while (hasNext() && (Character.isDigit(peek()) || peek() == '.')) {
                pos++;
            }
            if (hasNext() && (peek() == 'e' || peek() == 'E')) {
                pos++;
                if (hasNext() && (peek() == '+' || peek() == '-')) {
                    pos++;
                }
                while (hasNext() && Character.isDigit(peek())) {
                    pos++;
                }
            }
            try {
                return new ConstantNode(Double.parseDouble(input.substring(start, pos)));
            } catch (NumberFormatException e) {
                throw error("Invalid number");
            }
        }

        String parseIdentifier() {
            int start = pos;
            while (hasNext() && (Character.isLetterOrDigit(peek()) || peek() == '_')) {
                pos++;
            }
            return input.substring(start, pos);
        }

        void expectEnd() {
            skipWs();
            if (hasNext()) {
                throw error("Unexpected token");
            }
        }

        // ---- helpers ----

        boolean match(char expected) {
            if (hasNext() && peek() == expected) {
                pos++;
                return true;
            }
            return false;
        }

        char peek() {
            return input.charAt(pos);
        }

        boolean hasNext() {
            return pos < input.length();
        }

        void skipWs() {
            while (hasNext() && Character.isWhitespace(peek())) {
                pos++;
            }
        }

        IllegalArgumentException error(String msg) {
            return new IllegalArgumentException(
                    msg + " at pos " + pos + " in: " + input);
        }
    }
}
