package org.flexlb.balance.strategy;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;

/**
 * Configurable prefill-time formula engine.
 *
 * <h3>Syntax</h3>
 * <ul>
 *   <li><b>Operators</b>: {@code + - * / ^} (power, right-associative)</li>
 *   <li><b>Functions</b>: {@code sqrt(x) log(x) exp(x) abs(x) max(a,b) min(a,b) pow(a,b)}</li>
 *   <li><b>Batch aggregate</b>: {@code sum(expr)} evaluates {@code expr} per request and sums it</li>
 *   <li><b>Numbers</b>: decimal ({@code 3.14}) or scientific ({@code 1.2e-8})</li>
 *   <li><b>Parentheses</b>: {@code ( expr )}</li>
 * </ul>
 *
 * <h3>Variables</h3>
 * <table>
 *   <tr><th>Symbol</th><th>Meaning</th></tr>
 *   <tr><td>{@code inputTokens}</td><td>request input tokens</td></tr>
 *   <tr><td>{@code hitCacheTokens}</td><td>observed reusable KV-cache tokens</td></tr>
 *   <tr><td>{@code computeTokens}</td><td>{@code inputTokens - hitCacheTokens}</td></tr>
 *   <tr><td>{@code hasHitCache}</td><td>1 if {@code hitCacheTokens > 0}, otherwise 0</td></tr>
 *   <tr><td>{@code batchSize}</td><td>number of requests in the batch</td></tr>
 *   <tr><td>{@code totalInputTokens}</td><td>sum of input tokens across the batch</td></tr>
 *   <tr><td>{@code totalHitCacheTokens}</td><td>sum of cache-hit tokens across the batch</td></tr>
 *   <tr><td>{@code totalComputeTokens}</td><td>{@code totalInputTokens - totalHitCacheTokens}</td></tr>
 *   <tr><td>{@code maxInputTokens}</td><td>maximum request input length in the batch</td></tr>
 *   <tr><td>{@code maxComputeTokens}</td><td>maximum request compute length in the batch</td></tr>
 * </table>
 * <p>Batch-scoped variables ({@code batchSize}, {@code total*}, and {@code max*}) must be used
 * outside {@code sum(expr)}. Use the explicit {@code total*} variables for nonlinear batch-total terms. For example,
 * {@code totalComputeTokens^2} squares the batch total, while
 * {@code sum(computeTokens^2)} sums per-request squares. These expressions are intentionally
 * different and must not be substituted for each other. Use {@code sum(expr)} only when the
 * per-request distribution is part of the model.
 *
 * <h3>Example</h3>
 * <pre>{@code
 *   "param(base, 100) + param(batch, 2)*batchSize
 *       + param(compute, 0.01)*totalComputeTokens
 *       + param(compute2, 1e-8)*totalComputeTokens^2
 *       + param(maxCompute, 0.001)*maxComputeTokens
 *       + param(distribution, 1e-8)*sum(computeTokens^2)"
 * }</pre>
 */
public final class PrefillTimeFormula {

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

    static final int IDX_BATCH_SIZE = 0;
    static final int IDX_INPUT_TOKENS = 1;
    static final int IDX_HIT_CACHE_TOKENS = 2;
    static final int IDX_COMPUTE_TOKENS = 3;
    static final int IDX_HAS_HIT_CACHE = 4;
    static final int IDX_TOTAL_INPUT_TOKENS = 5;
    static final int IDX_TOTAL_HIT_CACHE_TOKENS = 6;
    static final int IDX_TOTAL_COMPUTE_TOKENS = 7;
    static final int IDX_MAX_INPUT_TOKENS = 8;
    static final int IDX_MAX_COMPUTE_TOKENS = 9;
    static final int VAR_COUNT = 10;

    private static final Map<String, Integer> VAR_INDEX_MAP = Map.of(
            "batchSize", IDX_BATCH_SIZE,
            "inputTokens", IDX_INPUT_TOKENS,
            "hitCacheTokens", IDX_HIT_CACHE_TOKENS,
            "computeTokens", IDX_COMPUTE_TOKENS,
            "hasHitCache", IDX_HAS_HIT_CACHE,
            "totalInputTokens", IDX_TOTAL_INPUT_TOKENS,
            "totalHitCacheTokens", IDX_TOTAL_HIT_CACHE_TOKENS,
            "totalComputeTokens", IDX_TOTAL_COMPUTE_TOKENS,
            "maxInputTokens", IDX_MAX_INPUT_TOKENS,
            "maxComputeTokens", IDX_MAX_COMPUTE_TOKENS
    );

    private final Node root;

    private PrefillTimeFormula(Node root) {
        this.root = root;
    }

    /**
     * Parse a formula string.
     *
     * @throws IllegalArgumentException if the formula is malformed or references unknown variables.
     */
    public static PrefillTimeFormula parse(String formula) {
        Map<String, ParameterNode> parameters = new LinkedHashMap<>();
        Parser parser = new Parser(formula, parameters);
        Node root = parser.parseExpression();
        parser.expectEnd();
        return new PrefillTimeFormula(root);
    }

    /**
     * Evaluate the formula with aggregate-aware per-request bindings.
     * {@code sum(expr)} evaluates {@code expr} for each array in {@code itemVars}.
     */
    public long evaluate(double[] vars, List<double[]> itemVars) {
        return (long) root.evaluate(new EvalContext(vars, itemVars));
    }

    // ---- AST nodes ----

    private interface Node {
        double evaluate(EvalContext ctx);
    }

    private static final class EvalContext {
        double[] vars;
        List<double[]> itemVars;

        EvalContext(double[] vars, List<double[]> itemVars) {
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
        private final Map<String, ParameterNode> parameters;
        private int pos;
        private int aggregateDepth;

        Parser(String input, Map<String, ParameterNode> parameters) {
            this.input = input;
            this.parameters = parameters;
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
                if (aggregateDepth > 0 && PrefillTimeVariableBindings.isBatchScoped(name)) {
                    throw error("Batch-scoped variable cannot be used inside sum(): " + name);
                }
                Integer idx = VAR_INDEX_MAP.get(name);
                if (idx == null) {
                    throw error("Unknown variable: " + name);
                }
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
            double initialValue = initialValueNode.evaluate(new EvalContext(new double[VAR_COUNT], null));
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
                    && !AGGREGATE_FUNCTIONS.contains(name)) {
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
