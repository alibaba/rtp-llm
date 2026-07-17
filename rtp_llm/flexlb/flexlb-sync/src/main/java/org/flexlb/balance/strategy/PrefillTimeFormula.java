package org.flexlb.balance.strategy;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleUnaryOperator;
import java.util.function.Predicate;

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
 * </table>
 * <p>Use {@code sum(expr)} for batch aggregates, for example
 * {@code sum(computeTokens)}, {@code sum(computeTokens^2)}, or
 * {@code sum(computeTokens * hitCacheTokens)}.
 *
 * <h3>Example</h3>
 * <pre>{@code
 *   "174.374677211 + 52.642812003*log(batchSize + 1)
 *       + 0.000746856881262*sum(2048*log(1 + exp((computeTokens - 8192)/2048)))
 *       + 0.0074536400604*sum(4096*log(1 + exp((computeTokens - 24576)/4096)))
 *       + 18.7646922156*(sum(hasHitCache)/batchSize)
 *       - 41.7583481006*(sum(log(hitCacheTokens + 1)/max(log(inputTokens + 1), 1))/batchSize)"
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
    static final int VAR_COUNT = 5;

    private static final Map<String, Integer> VAR_INDEX_MAP = Map.of(
            "batchSize", IDX_BATCH_SIZE,
            "inputTokens", IDX_INPUT_TOKENS,
            "hitCacheTokens", IDX_HIT_CACHE_TOKENS,
            "computeTokens", IDX_COMPUTE_TOKENS,
            "hasHitCache", IDX_HAS_HIT_CACHE
    );

    private final String source;
    private final Node root;
    private final Map<String, ParameterNode> parameters;

    private PrefillTimeFormula(String source, Node root, Map<String, ParameterNode> parameters) {
        this.source = source;
        this.root = root;
        this.parameters = parameters;
    }

    /**
     * Parse a formula string.
     *
     * @throws IllegalArgumentException if the formula is malformed or references unknown variables.
     */
    public static PrefillTimeFormula parse(String formula) {
        return parse(formula, PrefillTimeVariableBindings::supports);
    }

    static PrefillTimeFormula parse(String formula, Predicate<String> supportedVariable) {
        Map<String, ParameterNode> parameters = new LinkedHashMap<>();
        Parser parser = new Parser(formula, supportedVariable, parameters);
        Node root = parser.parseExpression();
        parser.expectEnd();
        return new PrefillTimeFormula(formula, root, parameters);
    }

    /**
     * Evaluate the formula with the given variable bindings.
     * Array slots not set default to 0.0.
     */
    public long evaluate(double[] vars) {
        return evaluate(vars, null);
    }

    /**
     * Evaluate the formula with aggregate-aware per-request bindings.
     * {@code sum(expr)} evaluates {@code expr} for each array in {@code itemVars}.
     */
    public long evaluate(double[] vars, List<double[]> itemVars) {
        return (long) root.evaluate(new EvalContext(vars, itemVars));
    }

    @Override
    public String toString() {
        return source;
    }

    // ---- parameter management ----

    public double getParameter(String name) {
        ParameterNode node = parameters.get(name);
        if (node == null) {
            throw new IllegalArgumentException("Unknown parameter: " + name);
        }
        return node.value();
    }

    public void setParameter(String name, double value) {
        ParameterNode node = parameters.get(name);
        if (node == null) {
            throw new IllegalArgumentException("Unknown parameter: " + name);
        }
        node.setValue(value);
    }

    public Set<String> parameterNames() {
        return Collections.unmodifiableSet(parameters.keySet());
    }

    public Map<String, Double> getParameters() {
        Map<String, Double> result = new LinkedHashMap<>();
        parameters.forEach((name, node) -> result.put(name, node.value()));
        return result;
    }

    public boolean hasParameters() {
        return !parameters.isEmpty();
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
        private final String name;
        private volatile double value;

        ParameterNode(String name, double initialValue) {
            this.name = name;
            this.value = initialValue;
        }

        @Override
        public double evaluate(EvalContext ctx) {
            return value;
        }

        public String name() {
            return name;
        }

        public double value() {
            return value;
        }

        public void setValue(double value) {
            this.value = value;
        }

        @Override
        public String toString() {
            return "param(" + name + ", " + value + ")";
        }
    }

    // ---- Recursive-descent parser ----

    private static final class Parser {
        private final String input;
        private final Predicate<String> supportedVariable;
        private final Map<String, ParameterNode> parameters;
        private int pos;

        Parser(String input, Predicate<String> supportedVariable, Map<String, ParameterNode> parameters) {
            this.input = input;
            this.supportedVariable = supportedVariable;
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
                if (!supportedVariable.test(name)) {
                    throw error("Unknown variable: " + name);
                }
                Integer idx = VAR_INDEX_MAP.get(name);
                if (idx == null) {
                    throw error("Variable not in index map: " + name);
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
            ParameterNode node = new ParameterNode(paramName, initialValue);
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
            Node arg0 = parseExpression();
            if (AGGREGATE_FUNCTIONS.contains(name)) {
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
