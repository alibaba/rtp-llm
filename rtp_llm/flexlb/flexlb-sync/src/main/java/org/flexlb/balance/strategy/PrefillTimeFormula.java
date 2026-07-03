package org.flexlb.balance.strategy;

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
 *   "213.058760744 + 0.000420120401621*sum(max(computeTokens - 2048, 0))
 *       + 0.00817215761679*sum(max(computeTokens - 24576, 0))
 *       - 0.000373217058264*sum(hitCacheTokens)
 *       - 10.6141559328*sum(hasHitCache)
 *       + 2.84762280669e-08*sum(computeTokens * hitCacheTokens)"
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

    private final String source;
    private final Node root;

    private PrefillTimeFormula(String source, Node root) {
        this.source = source;
        this.root = root;
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
        Parser parser = new Parser(formula, supportedVariable);
        Node root = parser.parseExpression();
        parser.expectEnd();
        return new PrefillTimeFormula(formula, root);
    }

    /**
     * Evaluate the formula with the given variable bindings.
     * Missing keys default to 0.
     */
    public long evaluate(Map<String, Double> vars) {
        return evaluate(vars, null);
    }

    /**
     * Evaluate the formula with aggregate-aware per-request bindings.
     * {@code sum(expr)} evaluates {@code expr} for each map in {@code itemVars}.
     */
    public long evaluate(Map<String, Double> vars, List<Map<String, Double>> itemVars) {
        return (long) root.evaluate(new EvalContext(vars, itemVars));
    }

    @Override
    public String toString() {
        return source;
    }

    // ---- AST nodes ----

    private interface Node {
        double evaluate(EvalContext ctx);
    }

    private record EvalContext(Map<String, Double> vars, List<Map<String, Double>> itemVars) {
        EvalContext withVars(Map<String, Double> nextVars) {
            return new EvalContext(nextVars, null);
        }
    }

    private record ConstantNode(double value) implements Node {
        @Override
        public double evaluate(EvalContext ctx) {
            return value;
        }
    }

    private record VariableNode(String name) implements Node {
        @Override
        public double evaluate(EvalContext ctx) {
            Double v = ctx.vars().get(name);
            return v != null ? v : 0.0;
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
            List<Map<String, Double>> itemVars = ctx.itemVars();
            if (itemVars == null || itemVars.isEmpty()) {
                return arg.evaluate(ctx.withVars(ctx.vars()));
            }
            double total = 0.0;
            for (Map<String, Double> item : itemVars) {
                total += arg.evaluate(ctx.withVars(item));
            }
            return total;
        }
    }

    // ---- Recursive-descent parser ----

    private static final class Parser {
        private final String input;
        private final Predicate<String> supportedVariable;
        private int pos;

        Parser(String input, Predicate<String> supportedVariable) {
            this.input = input;
            this.supportedVariable = supportedVariable;
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

        // primary → '(' expression ')' | function_call | number | variable
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
                    return parseFuncCall(name);
                }
                if (!supportedVariable.test(name)) {
                    throw error("Unknown variable: " + name);
                }
                return new VariableNode(name);
            }
            if (hasNext() && (Character.isDigit(peek()) || peek() == '.')) {
                return parseNumber();
            }
            throw error("Expected number, variable, or '('");
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
