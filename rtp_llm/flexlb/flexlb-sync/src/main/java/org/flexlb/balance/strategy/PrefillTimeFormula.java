package org.flexlb.balance.strategy;

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
 *   <li><b>Numbers</b>: decimal ({@code 3.14}) or scientific ({@code 1.2e-8})</li>
 *   <li><b>Parentheses</b>: {@code ( expr )}</li>
 * </ul>
 *
 * <h3>Variables</h3>
 * <table>
 *   <tr><th>Symbol</th><th>Meaning</th></tr>
 *   <tr><td>{@code c}</td><td>compute tokens = inputLen − hitCacheTokens</td></tr>
 *   <tr><td>{@code p}</td><td>cache-hit tokens</td></tr>
 *   <tr><td>{@code sum_c}</td><td>Σ cᵢ (batch aggregate)</td></tr>
 *   <tr><td>{@code sum_c2}</td><td>Σ cᵢ²</td></tr>
 *   <tr><td>{@code sum_cp}</td><td>Σ (cᵢ·pᵢ)</td></tr>
 *   <tr><td>{@code sum_p}</td><td>Σ pᵢ</td></tr>
 *   <tr><td>{@code n}</td><td>batch size</td></tr>
 * </table>
 *
 * <h3>Example</h3>
 * <pre>{@code
 *   "205 + 1.2e-8*sum_c2 + 1.2e-8*sum_cp + 5*n"
 * }</pre>
 */
public final class PrefillTimeFormula {

    private static final Set<String> KNOWN_VARS = Set.of(
            "c", "p", "sum_c", "sum_c2", "sum_cp", "sum_p", "n");

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
        Parser parser = new Parser(formula);
        Node root = parser.parseExpression();
        parser.expectEnd();
        return new PrefillTimeFormula(formula, root);
    }

    /**
     * Evaluate the formula with the given variable bindings.
     * Missing keys default to 0.
     */
    public long evaluate(Map<String, Double> vars) {
        return (long) root.evaluate(vars);
    }

    @Override
    public String toString() {
        return source;
    }

    // ---- AST nodes ----

    private interface Node {
        double evaluate(Map<String, Double> vars);
    }

    private record ConstantNode(double value) implements Node {
        @Override
        public double evaluate(Map<String, Double> vars) {
            return value;
        }
    }

    private record VariableNode(String name) implements Node {
        @Override
        public double evaluate(Map<String, Double> vars) {
            Double v = vars.get(name);
            return v != null ? v : 0.0;
        }
    }

    private record UnaryNode(char op, Node operand) implements Node {
        @Override
        public double evaluate(Map<String, Double> vars) {
            double v = operand.evaluate(vars);
            return op == '-' ? -v : v;
        }
    }

    private record BinaryNode(char op, Node left, Node right) implements Node {
        @Override
        public double evaluate(Map<String, Double> vars) {
            double l = left.evaluate(vars);
            double r = right.evaluate(vars);
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
        public double evaluate(Map<String, Double> vars) {
            double a = arg.evaluate(vars);
            return UNARY_FUNCTIONS.get(name).applyAsDouble(a);
        }
    }

    private record BinaryFuncNode(String name, Node left, Node right) implements Node {
        @Override
        public double evaluate(Map<String, Double> vars) {
            double l = left.evaluate(vars);
            double r = right.evaluate(vars);
            return BINARY_FUNCTIONS.get(name).applyAsDouble(l, r);
        }
    }

    // ---- Recursive-descent parser ----

    private static final class Parser {
        private final String input;
        private int pos;

        Parser(String input) {
            this.input = input;
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
                if (!KNOWN_VARS.contains(name)) {
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
            if (!UNARY_FUNCTIONS.containsKey(name) && !BINARY_FUNCTIONS.containsKey(name)) {
                throw error("Unknown function: " + name);
            }
            skipWs();
            Node arg0 = parseExpression();
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
